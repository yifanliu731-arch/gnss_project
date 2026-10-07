"""Regression checks for closed geometry and parameter-dependent scale search."""
import unittest
import numpy as np
from closed_survey import SurveyTour,FAMILIES
from joint_simulation import configuration,initial_specs,run_survey,generate_observations,schedule
from research_bridge import ResearchRayTracer
from munk_ssp import generate_munk_ssp_table

def request(**changes):
    b=dict(array_points=[[500,0,-1500],[-500,0,-1520],[0,500,-1510]],modes=['figure_eight'],
        scale_min_ratio=.5,scale_max_ratio=1.5,scale_count=2,time_budget_min_h=.5,time_budget_max_h=3.,
        ship_speed_kn=3.,sample_freq_hz=.05,noise=0.,gnss_noise_std=0.,temporal_sigma_mps=0.,sv_grad_n=0.,
        refinement_rounds=1,gdop_threshold=100.)
    b.update(changes);return b

class ClosedSurveyTests(unittest.TestCase):
    def test_half_turn_divisions_reach_candidate_angles(self):
        for n,expected in [(1,[0]),(2,[0,90]),(3,[0,60,120]),(4,[0,45,90,135]),(6,list(range(0,180,30)))]:
            c=configuration(request(rotation_divisions=n))
            np.testing.assert_allclose(c['angles'],expected)
            self.assertEqual(sorted({s[2] for s in initial_specs(c)}),expected)
        self.assertEqual(len(configuration(request(rotation_divisions=180))['angles']),180)
        for value in [0,-1,2.5,181,float('nan'),True]:
            with self.assertRaises(ValueError):configuration(request(rotation_divisions=value))
        with self.assertRaisesRegex(ValueError,'not both'):
            configuration(request(rotation_divisions=2,angles_deg=[45]))
        np.testing.assert_allclose(configuration(request(angles_deg=[0,22.5,45]))['angles'],[0,22.5,45])

    def test_divided_angles_are_actually_evaluated(self):
        a=run_survey(request(rotation_divisions=3,scale_min_ratio=.8,scale_max_ratio=.8,refinement_rounds=0))
        self.assertEqual(sorted(r['angle_deg'] for r in a['all_results']),[0,60,120])
        self.assertTrue(all(r['design_valid'] for r in a['all_results']))
        self.assertTrue(all(np.isfinite(r['c3_m2']) for r in a['all_results']))

    def test_closure_single_traversal_and_no_retraced_edges(self):
        for family in (f for f in FAMILIES if f != 'circle_cross'):
            with self.subTest(family=family):
                t=SurveyTour(family,2100,[12000,-8000],30)
                self.assertEqual(t.connector_length,0)
                np.testing.assert_allclose(t.positions([0])[0],t.positions([t.length])[0],atol=1e-7)
                xy=t.positions(np.linspace(0,t.length,8193))
                # Retracing the same sampled finite segment, in either direction,
                # is distinct from revisiting an isolated crossing point.
                keys=[]
                for a,b in zip(xy[:-1],xy[1:]):
                    ends=sorted([tuple(np.round(a,6)),tuple(np.round(b,6))]);keys.append(tuple(ends))
                self.assertEqual(len(keys),len(set(keys)),family)
                self.assertLessEqual(np.linalg.norm(xy-[12000,-8000],axis=1).max(),2100*1.0005)
        self.assertAlmostEqual(SurveyTour('circle',1000,[0,0]).length,2*np.pi*1000,delta=.001)
        self.assertAlmostEqual(SurveyTour('square',1000,[0,0]).length,4*np.sqrt(2)*1000,places=7)

    def test_unsupported_retracing_families_are_explicit(self):
        for family in ['square_cross','asterisk']:
            with self.assertRaisesRegex(ValueError,'complete survey'):configuration(request(modes=[family]))

    def test_circle_cross_complete_geometry_and_cost(self):
        t=SurveyTour('circle_cross',2100,[12000,-8000],30)
        self.assertAlmostEqual(t.length,(2*np.pi+4+np.sqrt(2))*2100,places=7)
        np.testing.assert_allclose(t.core.unit._loop_point(t.length/2100),[0,-1],atol=1e-7)
        unit=t.core.unit
        joins=[2*np.pi,2*np.pi+2,2*np.pi+2+np.sqrt(2)]
        for s in joins:
            a=unit._loop_point(s-1e-7);b=unit._loop_point(s+1e-7)
            self.assertLess(np.linalg.norm(a-b),3e-7)
        np.testing.assert_allclose(unit._loop_point(2*np.pi+1),[0,0],atol=1e-9)
        np.testing.assert_allclose(unit._loop_point(2*np.pi+3+np.sqrt(2)),[0,0],atol=1e-9)
        self.assertAlmostEqual(t.connector_length,np.sqrt(2)*2100)
        c=configuration(request(modes=['circle_cross'],rotation_divisions=4))
        specs=initial_specs(c)
        for family,ratio,angle,source in specs:
            if source=='budget_boundary':
                duration=SurveyTour(family,np.mean(-c['beacons'][:,2])*ratio,c['center'],angle).length/c['speed']/3600
                self.assertTrue(np.isclose(duration,c['budgets_h'],atol=1e-10).any())
        a=run_survey(request(modes=['circle_cross'],scale_min_ratio=1,scale_max_ratio=1,refinement_rounds=0,time_budget_max_h=4))
        self.assertTrue(all(r['design_valid'] for r in a['all_results']))
        self.assertGreater(a['n_valid'],0)
        self.assertTrue(all(r['rmse_3d_m']<.001 for r in a['all_results'] if r['valid']))

    def test_transfer_sampling_keeps_times_ids_and_shared_draws(self):
        c=configuration(request(modes=['circle_cross'],noise=.0005,gnss_noise_std=.05,temporal_sigma_mps=.3))
        tr=ResearchRayTracer(generate_munk_ssp_table(H=2200,dz=20))
        t=SurveyTour('circle_cross',2000,c['center'],45)
        off=generate_observations(t,c,tr)
        c['collect_transfer']=True
        on=generate_observations(t,c,tr)
        ix=np.searchsorted(on['times'],off['times'])
        self.assertGreater(off['skipped_transfer'],0)
        self.assertEqual(on['skipped_transfer'],0)
        for key in ['times','ids','ship','true_ship','delays','travel','field']:
            np.testing.assert_array_equal(off[key],on[key][ix])
        slot,times=schedule(t,c,tr)
        self.assertEqual(len(times)-len(off['times']),off['skipped_transfer'])
        self.assertEqual(off['attempted'],on['attempted'])

    def test_budget_boundaries_and_user_scale_bounds(self):
        for changes in [dict(),dict(scale_min_ratio=.2,scale_max_ratio=2.7),dict(ship_speed_kn=5.),
                        dict(time_budget_min_h=1.2,time_budget_max_h=4.2),dict(array_points=[[0,0,-2800],[800,-400,-2900]])]:
            c=configuration(request(**changes));depth=np.mean(-c['beacons'][:,2]);specs=initial_specs(c)
            for family,ratio,angle,source in specs:
                self.assertGreaterEqual(ratio,c['ratios'][0]);self.assertLessEqual(ratio,c['ratios'][-1])
                if source=='budget_boundary':
                    duration=SurveyTour(family,depth*ratio,c['center'],angle).length/c['speed']/3600
                    self.assertTrue(np.isclose(duration,c['budgets_h'],atol=1e-10).any())
        c=configuration(request(scale_min_ratio=11,scale_max_ratio=15))
        self.assertEqual(c['ratios'].tolist(),[11.,15.])

    def test_invalid_bounds_and_count(self):
        for x in [dict(scale_min_ratio=2,scale_max_ratio=1),dict(scale_count=1),dict(scale_count=2.5),
                  dict(scale_min_ratio=float('nan')),dict(refinement_rounds=.5)]:
            with self.assertRaises(ValueError):configuration(request(**x))

    def test_one_range_mask_does_not_remove_other_beacons(self):
        c=configuration(request(array_points=[[-2000,0,-1500],[2000,0,-1500]],scale_min_ratio=1,scale_max_ratio=1,
                                max_range_km=3,refinement_rounds=0))
        tracer=ResearchRayTracer(generate_munk_ssp_table(H=2000,dz=20))
        tour=SurveyTour('circle',2500,c['center'])
        data=generate_observations(tour,c,tracer)
        self.assertGreater(data['rejected_range'],0)
        self.assertGreaterEqual(min(data['per_beacon_counts']),8)
        ranges=np.linalg.norm(data['true_ship'][:,None,:]-c['beacons'][None,:,:],axis=2)
        self.assertTrue(np.all(ranges[np.arange(len(ranges)),data['ids']]<=3000))
        self.assertTrue(np.any(np.max(ranges,axis=1)>3000))

    def test_end_to_end_refinement_and_shared_final_pool(self):
        b=request();a=run_survey(b)
        self.assertEqual(len(a['budget_results']),10)
        self.assertGreater(a['n_total'],2)
        self.assertTrue(any(x['search_source']=='risk_refinement' for x in a['all_results']))
        self.assertTrue(a['search']['pool_frozen_before_shared_field'])
        for row in a['all_results']:
            self.assertTrue(.5<=row['ratio']<=1.5)
            self.assertEqual(row['connector_length_m'],0.)
        for budget in a['budget_results']:
            for key in ['plan_a','plan_b','plan_c']:
                if budget[key]:
                    p=a['plans_by_id'][budget[key]]
                    self.assertLessEqual(p['duration_h'],budget['budget_h']+1e-10)
                    self.assertLess(p['rmse_3d_m'],.001)
        # Changing only the objective cannot silently change the search pool.
        b['criterion']='c1';other=run_survey(b)
        self.assertEqual([r['candidate_id'] for r in a['all_results']],[r['candidate_id'] for r in other['all_results']])

    def test_empty_budget_does_not_override_scale_bounds(self):
        a=run_survey(request(scale_min_ratio=12,scale_max_ratio=15,time_budget_min_h=.1,time_budget_max_h=.2))
        self.assertIsNone(a['plan_a']);self.assertEqual(a['n_valid'],0)
        self.assertTrue(all(r['ratio']>=12 for r in a['all_results']))

if __name__=='__main__':unittest.main()
