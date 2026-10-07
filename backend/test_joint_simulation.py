"""Regression checks for geometry, units, scheduling, inference and API wiring.

Run from backend with the SeaArrayCalib Python environment: python -m unittest
test_joint_simulation -v. Tests do not modify field data or manuscript files.
"""
import os
import unittest
from pathlib import Path
import numpy as np
from joint_simulation import (configuration, run_survey, generate_observations,
                              nominal_rows, schedule, ou_product)
from research_bridge import SurveyTour, ResearchRayTracer, FAMILIES, RESEARCH_ROOT
from munk_ssp import generate_munk_ssp_table

def body(n=4, **kw):
    angles=np.arange(n)*2*np.pi/n
    b=np.c_[400*np.cos(angles),400*np.sin(angles),-1500-np.arange(n)*10]
    return dict(array_points=b.tolist(),modes=['figure_eight'],size_ratios=[1.],
                ship_speed_kn=3.,sample_freq_hz=.05,time_budget_h=8.,
                temporal_sigma_mps=0.,sv_grad_e=0.,sv_grad_n=0.,
                gnss_noise_std=0.,noise=0.,**kw)

class SurveyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracer=ResearchRayTracer(generate_munk_ssp_table(H=2200,dz=20))

    def test_all_shapes_are_complete_and_centered(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                tour=SurveyTour(family,1500,[12000,-8000],45)
                xy=tour.positions([0,tour.length])
                np.testing.assert_allclose(xy,[[12000,-8000]]*2,atol=1e-6)
                samples=tour.positions(np.linspace(0,tour.length,6001))
                # Core clover uses sampled normalization (about 0.021% tolerance).
                self.assertLessEqual(np.linalg.norm(samples-[12000,-8000],axis=1).max(),1500*1.0005)
                self.assertGreater(tour.length,3000)

    def test_joint_generalizes_and_recovers_truth_without_noise(self):
        for n in [1,3,4,6]:
            with self.subTest(n=n):
                r=run_survey(body(n))
                self.assertEqual(r['n_valid'],1,r['all_results'])
                p=r['plan_a'];self.assertIsNotNone(p)
                self.assertEqual(len(p['estimated_beacons']),n)
                self.assertEqual(p['joint_parameter_count'],4*n+2)
                self.assertLess(p['rmse_3d_m'],.001)
                self.assertEqual(p['c1_m2'],p['c3_m2'])

    def test_full_schedule_not_shared_ping_count(self):
        c=configuration(body());a=SurveyTour('circle',1000,c['center']);b=SurveyTour('circle',2000,c['center'])
        sa,ta=schedule(a,c,self.tracer);sb,tb=schedule(b,c,self.tracer)
        self.assertGreater(len(tb),len(ta))
        self.assertLessEqual(ta[-1]+sa,a.length/c['speed']+1e-7)
        self.assertLessEqual(tb[-1]+sb,b.length/c['speed']+1e-7)

    def test_noise_control_and_seed(self):
        c=configuration(body());tour=SurveyTour('figure_eight',1500,c['center'])
        clean=generate_observations(tour,c,self.tracer)
        c['sw']=.0005
        a=generate_observations(tour,c,self.tracer);b=generate_observations(tour,c,self.tracer)
        np.testing.assert_array_equal(a['delays'],b['delays'])
        self.assertGreater(np.std(a['delays']-clean['delays']),.0003)
        self.assertEqual(len(a['times']),a['attempted'])
        self.assertTrue(np.all(np.diff(a['times'])>=a['slot_s']-1e-7))

    def test_candidate_order_does_not_change_shared_field_or_solution(self):
        q=body();q.update(modes=['circle','figure_eight'],noise=.0001,temporal_sigma_mps=.3)
        a=run_survey(q);q['modes'].reverse();b=run_survey(q)
        aa={r['type']:r for r in a['all_results']};bb={r['type']:r for r in b['all_results']}
        for key in aa:
            self.assertTrue(aa[key]['valid'],aa[key])
            for metric in ['c1_m2','c3_m2','rmse_3d_m']:
                self.assertAlmostEqual(aa[key][metric],bb[key][metric],places=10)

    def test_research_geometry_is_reused_exactly(self):
        from risk_design.trajectories import CompleteTour
        for family in list(FAMILIES)[:7]:
            a=SurveyTour(family,2100,[12000,-8000],30)
            b=CompleteTour(family,2100,np.array([12000,-8000]),np.deg2rad(30),turn_radius=0.)
            self.assertEqual(a.length,b.length())
            s=np.linspace(0,a.length,300)
            np.testing.assert_array_equal(a.positions(s),b.positions(s)[:,:2])

    def test_range_and_budget_failures_stay_visible(self):
        for key,value in [('max_range_km',.1),('time_budget_h',.1)]:
            q=body();q[key]=value;r=run_survey(q)
            self.assertIsNone(r['plan_a']);self.assertEqual(r['n_valid'],0)
            self.assertTrue(r['all_results'][0]['reason'])

    def test_gdop_is_enforced_without_relabeling_failure(self):
        q=body();q['gdop_threshold']=.01;r=run_survey(q)
        self.assertEqual(r['n_valid'],1);self.assertEqual(r['n_eligible'],0)
        self.assertIsNone(r['plan_a'])

    def test_ou_operator_matches_dense(self):
        times=np.array([0.,1.,14.,50.]);x=np.arange(12).reshape(4,3)*.01
        expected=np.exp(-abs(times[:,None]-times[None,:])/20.)@x
        np.testing.assert_allclose(ou_product(times,x,20),expected,atol=1e-14)

    def test_joint_jacobian_matches_finite_difference(self):
        c=configuration(body());tour=SurveyTour('figure_eight',1500,c['center'])
        d=generate_observations(tour,c,self.tracer);n=len(c['beacons'])
        theta=np.r_[c['beacons'].ravel(),np.zeros(n+2)]
        ship=d['ship'][::15];ids=d['ids'][::15];times=d['times'][::15]
        _,jac=nominal_rows(self.tracer,ship,ids,c['beacons'],times,theta)
        for col in [0,2,3*n,3*n+1,3*n+2,3*n+3]:
            eps=1e-3 if col<=3*n else 1e-6
            a=theta.copy();b=theta.copy();a[col]+=eps;b[col]-=eps
            diff=(nominal_rows(self.tracer,ship,ids,c['beacons'],times,a)[0]-nominal_rows(self.tracer,ship,ids,c['beacons'],times,b)[0])/(2*eps)
            np.testing.assert_allclose(diff,jac[:,col],rtol=2e-5,atol=1e-8)

    def test_profile_mapping_matches_uniform_sound_perturbation_limit(self):
        ship=np.array([[200.,500.,-25.],[700.,-100.,-25.]])
        beacon=np.array([0.,0.,-1500.]);h=self.tracer.mapping(ship,beacon,[0,0],mode_depth=1e12)
        _,jac=self.tracer.varying_offset(ship,beacon,0.)
        np.testing.assert_allclose(h[:,0],jac[:,3],rtol=1e-7)

    def test_api_roundtrip_preview_and_errors(self):
        os.environ['SEAARRAY_ROOT']=str(RESEARCH_ROOT)
        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/calculate',json={'array_points':[]}).status_code,422)
            self.assertEqual(client.post('/api/calculate',json=[]).status_code,422)
            preview=client.post('/api/simulation/preview',json=body())
            self.assertEqual(preview.status_code,200)
            response=client.post('/api/calculate',json=body())
            self.assertEqual(response.status_code,200)
            result=client.get('/api/progress/'+response.json()['task_id']).json()
            self.assertEqual(result['status'],'completed',result)
            self.assertIs(result['result']['all_results'][0]['valid'],True)
            self.assertGreater(len(result['result']['plan_a']['observations']),0)
            self.assertEqual(client.get('/api/progress/nonexistent').status_code,404)

if __name__=='__main__':unittest.main()
