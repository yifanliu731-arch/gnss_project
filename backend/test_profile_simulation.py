"""Scientific invariants and public API/export checks for profile mode."""
import copy
import hashlib
import io
import json
import unittest
import zipfile
import numpy as np
from profile_data import EXAMPLE, example_payload, load_profiles
from profile_simulation import tracer_for, profile_metrics, truth_observations
from joint_simulation import configuration, generate_observations, run_survey, design_metrics
from closed_survey import SurveyTour
from risk_design.profile_ensemble import profile_mapping
from munk_ssp import generate_munk_ssp_table
from test_joint_simulation import body


def settings(zero=False):
    p=generate_munk_ssp_table(H=2200,dz=50)
    z=p.iloc[:,0].to_numpy();v=p.iloc[:,1].to_numpy()
    background='depth_m,sound_speed_mps\n'+''.join(f'{d:.17g},{c:.17g}\n' for d,c in zip(z,v))
    history='date,depth_m,sound_speed_mps\n'
    for day,amplitude in [('2020-01-01',-.4),('2020-01-02',.7),('2020-01-03',2.),('2020-01-04',.5),('2020-01-05',1.5)]:
        for d,c in zip(z,v):
            history+=f'{day},{d:.17g},{c+(0 if zero else amplitude)*np.exp(-d/700):.17g}\n'
    return dict(background_csv=background,history_csv=history,training_end='2020-01-02',validation_start='2020-01-04',
                validation_date='2020-01-04',validation_scope='all',amplitude=1.)


class ProfileTests(unittest.TestCase):
    def test_example_hash_split_and_second_moment(self):
        manifest=json.loads((EXAMPLE/'provenance.json').read_text())
        for name,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((EXAMPLE/name).read_bytes()).hexdigest(),digest)
        fields=load_profiles(example_payload())
        self.assertEqual([fields['metadata'][k] for k in ('training_count','gap_count','validation_count')],[182,14,170])
        self.assertEqual(len(fields['basis']),32)
        d=fields['delta'][fields['train']]
        np.testing.assert_allclose(fields['basis'].T@fields['basis'],d.T@d/len(d),atol=2e-12,rtol=2e-10)

    def test_invalid_inputs_are_rejected(self):
        s=settings()
        for changes in [dict(training_end='2020-01-04'),dict(validation_date='2020-01-02'),dict(amplitude=-1),
                        dict(background_csv='depth_m,sound_speed_mps\n0,NaN\n2,1500'),
                        dict(history_csv=s['history_csv']+'2020-01-01,0,1500\n')]:
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):load_profiles(dict(s,**changes))

    def test_statistics_follow_uploaded_content_and_split(self):
        s=settings();first=load_profiles(s,fit=False)['metadata']
        self.assertEqual([first[k] for k in ('training_count','gap_count','validation_count','depth_count','max_depth_m')],[2,1,2,45,2200.])
        second=load_profiles(dict(s,training_end='2020-01-01',validation_start='2020-01-03'),fit=False)['metadata']
        self.assertEqual([second[k] for k in ('training_count','gap_count','validation_count')],[1,1,3])
        # Replacing with the sample computes sample statistics; no rows accumulate.
        third=load_profiles(example_payload(),fit=False)['metadata']
        self.assertEqual(third['training_count'],182)

    def test_test_values_do_not_change_basis_or_selection(self):
        s=settings();q=body(3,sound_field_mode='profiles',sound_field=s)
        first=run_survey(q)
        lines=s['history_csv'].splitlines()
        s2=copy.deepcopy(s)
        rewritten=[lines[0]]
        for line in lines[1:]:
            day,z,v=line.split(',')
            rewritten.append(f'{day},{z},{float(v)+(3 if day>="2020-01-04" else 0):.17g}')
        s2['history_csv']='\n'.join(rewritten)+'\n'
        f1=load_profiles(s);f2=load_profiles(s2)
        np.testing.assert_array_equal(f1['basis'],f2['basis'])
        second=run_survey(dict(q,sound_field=s2))
        self.assertEqual(first['search']['selection_sha256'],second['search']['selection_sha256'])
        for metric in ('c1_m2','c3_m2'):
            self.assertEqual(first['all_results'][0][metric],second['all_results'][0][metric])

    def test_finite_difference_and_direct_training_risk(self):
        fields=load_profiles(settings());c=configuration(body(4));tr=tracer_for(fields['depth'],fields['nominal'])
        data=generate_observations(SurveyTour('figure_eight',1500,c['center']),c,tr)
        mode=fields['basis'][0];epsilon=1e-3
        plus=tracer_for(fields['depth'],fields['nominal']+epsilon*mode)
        minus=tracer_for(fields['depth'],fields['nominal']-epsilon*mode)
        analytic=profile_mapping(tr,data['true_ship'],data['ids'],c['beacons'],fields['depth'],mode[None,:])[:,0]
        numeric=np.empty(len(analytic))
        for b,beacon in enumerate(c['beacons']):
            ix=data['ids']==b
            numeric[ix]=(plus.varying_offset(data['true_ship'][ix],beacon,0)[0]-minus.varying_offset(data['true_ship'][ix],beacon,0)[0])/(2*epsilon)
        np.testing.assert_allclose(numeric,analytic,atol=2e-10,rtol=2e-5)
        metrics,gain=design_metrics(data,c,tr,return_gain=True)
        h=profile_mapping(tr,data['true_ship'],data['ids'],c['beacons'],fields['depth'],fields['delta'][fields['train']])
        a=gain@h/np.sqrt(3*len(c['beacons'])*fields['train'].sum())
        expected=np.linalg.eigvalsh(a@a.T)[-1]
        one=profile_metrics(data,c,tr,fields)
        two=profile_metrics(data,c,tr,dict(fields,alpha=2))
        self.assertAlmostEqual(one['worst_bias_m2'],expected,places=10)
        self.assertAlmostEqual(two['worst_bias_m2'],4*one['worst_bias_m2'],places=10)
        self.assertAlmostEqual(one['c2_m2']-one['c1_m2'],float(np.trace(a@a.T)),places=10)
        self.assertGreaterEqual(one['c2_m2']+1e-12,one['c3_m2'])

    def test_no_perturbation_recovers_truth_and_multiple_beacons(self):
        for n in (3,4):
            result=run_survey(body(n,sound_field_mode='profiles',sound_field=settings(zero=True)))
            self.assertEqual(result['field_validation']['failed_fits'],0)
            self.assertEqual(result['plan_a']['c1_m2'],result['plan_a']['c3_m2'])
            self.assertLess(result['plan_a']['rmse_3d_m'],.001)
            self.assertEqual(result['plan_a']['joint_parameter_count'],4*n+2)

    def test_c2_selection_and_criterion_independent_pool(self):
        for mode in ('munk','profiles'):
            q=body(3,sound_field_mode=mode,sound_field=settings())
            q.update(modes=['circle','figure_eight'],size_ratios=[.75,1.25],noise=.0001,temporal_sigma_mps=.3)
            first=run_survey(dict(q,criterion='c1'))
            second=run_survey(dict(q,criterion='c2'))
            self.assertEqual([r['candidate_id'] for r in first['all_results']],[r['candidate_id'] for r in second['all_results']])
            eligible=[r for r in second['all_results'] if r['valid'] and r['gdop']<=q.get('gdop_threshold',2.5)]
            self.assertAlmostEqual(second['plan_a']['c2_m2'],min(r['c2_m2'] for r in eligible),places=12)
            self.assertTrue(all(r['c2_m2']+1e-12>=r['c3_m2']>=r['c1_m2'] for r in eligible))

    def test_failed_test_does_not_replace_selected_route(self):
        from unittest.mock import patch
        q=body(3,sound_field_mode='profiles',sound_field=settings())
        with patch('profile_simulation.js.joint_fit',side_effect=ValueError('Deliberate validation failure')):
            result=run_survey(q)
        self.assertIsNotNone(result['plan_a'])
        self.assertTrue(result['plan_a']['design_valid'])
        self.assertEqual(result['plan_a']['validation_status'],'failed')
        self.assertGreater(result['field_validation']['failed_fits'],0)

    def test_api_and_saved_exports(self):
        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/simulation/profiles/check',json=settings()).status_code,200)
            self.assertEqual(client.post('/api/simulation/profiles/check',json={}).status_code,422)
            raw=client.get('/api/simulation/profile-example/download')
            self.assertEqual(raw.status_code,200)
            with zipfile.ZipFile(io.BytesIO(raw.content)) as z:
                self.assertEqual(z.read('background.csv'),(EXAMPLE/'background.csv').read_bytes())
            q=body(3,sound_field_mode='profiles',sound_field=settings())
            response=client.post('/api/calculate',json=q)
            task=response.json()['task_id'];result=client.get('/api/progress/'+task).json()
            self.assertEqual(result['status'],'completed',result)
            exported=client.post('/api/export/'+task,json=dict(scope='all',parts=['raw','observations','residuals']))
            self.assertEqual(exported.status_code,200,exported.text[:100] if exported.status_code!=200 else '')
            with zipfile.ZipFile(io.BytesIO(exported.content)) as z:
                self.assertEqual(z.read('original_inputs/history.csv').decode(),q['sound_field']['history_csv'])
                self.assertIn('validation_trials.csv',z.namelist())
                self.assertIn('nominal_delay_s',z.read('results/0001/observations.csv').decode())


if __name__=='__main__':unittest.main()
