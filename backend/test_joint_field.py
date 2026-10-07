import unittest
import numpy as np
import pandas as pd
from joint_field import solve_field_joint
from joint_simulation import nominal_rows
from research_bridge import ResearchRayTracer
from main import parse_beacon_ids,app,TASK_STORE,_run_realdatabackground
from fastapi.testclient import TestClient

def fixture(n=3, layers=40):
    profile=pd.DataFrame({'Depth (m)':np.linspace(0,3500,layers),'SV (m/s)':np.full(layers,1500.)})
    theta=np.linspace(0,2*np.pi,64,endpoint=False)
    ship=np.c_[1800*np.cos(theta),1200*np.sin(theta),-10+np.cos(3*theta)]
    truth=np.array([[250*np.cos(k*2*np.pi/n),200*np.sin(k*2*np.pi/n),-1500-20*k] for k in range(n)])
    groups={}
    for k,bid in enumerate([5,2,6,11][:n]):
        times=np.arange(len(ship))*20.+k+np.where(np.arange(len(ship))>=32,6000.,0.)
        delay=np.linalg.norm(ship-truth[k],axis=1)/1500.
        order=np.roll(np.arange(len(ship))[::-1],k)
        groups[bid]=dict(ship=ship[order],times=times[order],delays=delay[order],source_times=[f'UTC-{bid}-{i}' for i in order])
    return groups,profile,{bid:b for bid,b in zip([5,2,6,11][:n],truth)}

class JointFieldTests(unittest.TestCase):
    def test_residual_uses_complete_prediction_and_seconds_to_ms(self):
        groups,profile,truth=fixture()
        tracer=ResearchRayTracer(profile)
        all_times=np.concatenate([g['times'] for g in groups.values()]);t0=np.median(all_times)
        for bid,g in groups.items():
            travel,_=tracer.varying_offset(g['ship'],truth[bid],.2)
            g['delays']=travel+.001+.00002*(g['times']-t0)/3600+({5:0,2:.0001,6:-.0001}[bid])
        records,rows,summary=solve_field_joint(groups,profile,gnss_noise_m=0)
        for row in rows:
            g=groups[row['beacon_id']];index=summary['beacon_ids'].index(row['beacon_id'])
            travel,_=tracer.varying_offset(g['ship'],np.array([row['x'],row['y'],row['z']]),summary['sound_speed_offset_mps'])
            fitted=travel+summary['common_delay_s']+summary['relative_delays_s'][index]+summary['delay_rate_s_per_hour']*(g['times']-summary['time_reference_s'])/3600
            expected=1000*(g['delays']-fitted)
            np.testing.assert_allclose(row['residuals_ms'],expected,atol=1e-9,rtol=0)
            self.assertAlmostEqual(row['rmse_time_ms'],np.sqrt(np.mean(expected**2)),places=9)

    def test_general_n_free_coordinates_and_original_observation_order(self):
        for n in (1,3,4):
            with self.subTest(n=n):
                groups,profile,truth=fixture(n)
                records,rows,solution=solve_field_joint(groups,profile,gnss_noise_m=0)
                self.assertEqual(solution['joint_parameter_count'],4*n+2)
                self.assertEqual(solution['augmented_rank'],4*n+2)
                self.assertEqual(solution['reference_beacon_id'],5)
                self.assertNotIn('position_errors_m',solution)
                self.assertNotIn('rmse_3d_m',solution)
                self.assertLess(solution['solver_remaining_scaled_step'],1e-5)
                for r in records:
                    np.testing.assert_allclose([r['x'],r['y'],r['z']],truth[r['beacon_id']],atol=.001)
                for row in rows:
                    original=groups[row['beacon_id']]
                    self.assertEqual([r['source_time'] for r in row['observations']],original['source_times'])
                    np.testing.assert_array_equal([r['time_s'] for r in row['observations']],original['times'])
                    self.assertGreater(np.ptp(original['times']),6000)
                    self.assertEqual(len(row['residuals_ms']),len(original['times']))
                    self.assertLess(max(abs(np.asarray(row['residuals_ms']))),.001)

    def test_measured_profile_layers_not_decimated(self):
        groups,profile,_=fixture(3,3243)
        from unittest.mock import patch
        import joint_field
        original=joint_field.ResearchRayTracer
        captured=[]
        def tracer(p):
            captured.append(p.copy());return original(p)
        with patch.object(joint_field,'ResearchRayTracer',side_effect=tracer):
            _,_,summary=solve_field_joint(groups,profile)
        pd.testing.assert_frame_equal(captured[0].iloc[:len(profile)].reset_index(drop=True),profile)
        self.assertEqual(summary['measured_ssp_max_depth_m'],3500)
        self.assertEqual(summary['model_ssp_max_depth_m'],6000)

    def test_real_research_mean_and_jacobian_consistency(self):
        from risk_design.joint import JointProblem
        from risk_design.forward import TravelTimeModel
        groups,profile,_=fixture()
        tracer=ResearchRayTracer(profile)
        # Research order B2,B5,B6 versus general-N order B5,B2,B6.
        points=np.array([[140,-220,-1500],[100,200,-1515],[-300,50,-1530.]])
        ship=np.concatenate([groups[bid]['ship'] for bid in [2,5,6]])
        ids=np.repeat([2,5,6],64);times=np.concatenate([groups[bid]['times'] for bid in [2,5,6]])
        theta=np.r_[points.ravel(),.15,.0002,.00001,.00003,-.00004]
        problem=JointProblem(TravelTimeModel(tracer),ship,ids,times,np.full(len(times),.0005**2),t0=float(np.median(times)))
        # B2 and B6 relative delays use B5 as gauge in both models.
        general=np.r_[points[[1,0,2]].ravel(),theta[9:12],theta[12:14]]
        mapped=np.array([{5:0,2:1,6:2}[i] for i in ids])
        y,j=nominal_rows(tracer,ship,mapped,points[[1,0,2]],times,general)
        np.testing.assert_allclose(y,problem.mean(theta),rtol=0,atol=1e-10)
        # Analytic derivative versus the research derivative, reordered axes.
        np.testing.assert_allclose(j[:,[3,4,5,0,1,2,6,7,8,9,10,11,12,13]],problem.derivative(theta),rtol=2e-4,atol=1e-9)

    def test_id_parse_and_upload_discovery(self):
        self.assertEqual(parse_beacon_ids('2，5;6 5'),[2,5,6])
        for text in ('2,wrong,5','0,5','2.5,6'):
            with self.assertRaises(ValueError):parse_beacon_ids(text)
        csv='Time_UTC,Beacon_ID,Delay_s\n2021-05-01T00:00:00Z,2,1\n2021-05-01T00:00:01Z,5,1\n2021-05-01T00:00:02Z,6,1\n'
        response=TestClient(app).post('/api/delay/parse',files={'delay_file':('d.csv',csv,'text/csv')})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['beacon_ids'],[2,5,6])

    def test_invalid_data_is_not_silently_fitted(self):
        groups,profile,_=fixture()
        groups[5]['delays'][0]=np.nan
        with self.assertRaisesRegex(ValueError,'invalid aligned'):solve_field_joint(groups,profile)

    def test_field_api_defaults_to_joint_and_single_mode_is_retained(self):
        import csv,io,json,zipfile
        from geo import enu_matrix,apply_lever_arm_batch
        from datetime import datetime,timedelta,timezone
        count=48;angle=np.arange(count)*2*np.pi/count
        lat=18.7+.01*np.cos(angle);lon=116.5+.015*np.sin(angle);alt=np.full(count,20.)
        ship=apply_lever_arm_batch(enu_matrix(lat,lon,alt,18.7,116.5),[1.585,6.716,-29.05],np.zeros(count),np.zeros(count),np.zeros(count))
        start=datetime(2021,5,1,tzinfo=timezone.utc)
        stamps=[(start+timedelta(seconds=20*i+(3600 if i>=24 else 0))).isoformat() for i in range(count)]
        def csv_bytes(header,rows):
            text=io.StringIO();writer=csv.writer(text);writer.writerow(header);writer.writerows(rows);return text.getvalue().encode()
        gnss=csv_bytes(['Time_UTC','Latitude_deg','Longitude_deg','Altitude_m','Pitch_deg','Roll_deg','Heading_deg'],[[stamps[i],lat[i],lon[i],20,0,0,0] for i in range(count)])
        delays=csv_bytes(['Time_UTC','Beacon_ID','Delay_s'],[[stamps[i],bid,float(np.linalg.norm(ship[i]-np.array([200*k,-100*k,-1500-20*k]))/1500)] for i in range(count) for k,bid in enumerate([2,5,6])])
        ssp=csv_bytes(['Depth_m','Velocity_mps'],[[i*100,1500] for i in range(36)])
        files={'gnss_file':('g.csv',gnss,'text/csv'),'delay_file':('d.csv',delays,'text/csv'),'svp_file':('s.csv',ssp,'text/csv')}
        for mode in ('joint_map','single_map'):
            options={} if mode=='joint_map' else {'solver_type':mode,'beacon_ids':'2，5 6'}
            client=TestClient(app);response=client.post('/api/calculate',files=files,data=options)
            self.assertEqual(response.status_code,200,response.text)
            tid=response.json()['task_id']
            try:
                result=client.get('/api/progress/'+tid).json()
                self.assertEqual(result['status'],'completed',result.get('error'))
                self.assertNotIn('_export',result)
                result=result['result'];self.assertEqual(result['solver_type'],mode)
                self.assertEqual([r['beacon_id'] for r in result['beacon_results']],[2,5,6])
                self.assertEqual(sum(r['n_clean'] for r in result['beacon_results']),144)
                self.assertEqual(sum(p['gap_before'] for p in result['trajectory']),1)
                self.assertEqual(sum(p['gap_before'] for p in result['trajectory_latlon']),1)
                self.assertEqual(len(result['residual_observations']),144)
                self.assertEqual([r['beacon_id'] for r in result['residual_observations'][:3]],[2,5,6])
                self.assertIsNone(result['depth_prior_m'])
                if mode=='joint_map':self.assertEqual(result['joint_solution']['joint_parameter_count'],14)
                else:
                    self.assertIsNone(result['joint_solution'])
                    self.assertTrue(all(r['single_solution']['joint_parameter_count']==6 for r in result['beacon_results']))
                exported=client.post('/api/export/'+tid,json=dict(scope='custom',ids=[5,6],parts=['observations','residuals','raw']))
                self.assertEqual(exported.status_code,200)
                archive=zipfile.ZipFile(io.BytesIO(exported.content))
                payload=json.loads(archive.read('results.json'))
                self.assertEqual([r['beacon_id'] for r in payload['results']],[5,6])
                for row in payload['results']:
                    self.assertEqual(len(row['observations']),48)
                    self.assertTrue(all(o['beacon_id']==row['beacon_id'] for o in row['observations']))
                self.assertEqual(archive.read('original_inputs/delay.csv'),delays)
            finally:TASK_STORE.pop(tid,None)

if __name__=='__main__':unittest.main()
