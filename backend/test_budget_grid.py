"""Budget endpoints, logarithmic spacing, reusable candidates and range behavior."""
import unittest
from unittest.mock import patch
import numpy as np
import joint_simulation as simulation
from test_joint_simulation import body

class BudgetTests(unittest.TestCase):
    def test_input_endpoints_and_log_spacing(self):
        c=simulation.configuration({**body(), 'time_budget_min_h':1.3,'time_budget_max_h':21.7})
        values=c['budgets_h']
        self.assertEqual(len(values),10);self.assertEqual(values[0],1.3);self.assertEqual(values[-1],21.7)
        np.testing.assert_allclose(np.diff(np.log(values)),np.repeat(np.log(21.7/1.3)/9,9),rtol=1e-12)
        self.assertEqual(c['budget'],21.7*3600)
        for bad in [dict(time_budget_min_h=2),dict(time_budget_max_h=7),dict(time_budget_min_h=7,time_budget_max_h=2),dict(time_budget_min_h=2,time_budget_max_h=2),dict(time_budget_min_h=0,time_budget_max_h=7),dict(time_budget_min_h=1,time_budget_max_h=float('nan'))]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):simulation.configuration({**body(),**bad})

    def test_each_budget_selects_from_same_complete_candidate_fits(self):
        request={**body(3),'time_budget_min_h':.2,'time_budget_max_h':8.,'size_ratios':[.5,1.0,1.5],'gdop_threshold':100}
        with patch.object(simulation,'joint_fit',wraps=simulation.joint_fit) as fit:
            r=simulation.run_survey(request)
        self.assertEqual(fit.call_count,r['n_valid'])
        self.assertEqual(len(r['budget_results']),10)
        self.assertEqual(r['budget_results'][0]['n_eligible'],0)
        for b in r['budget_results']:
            eligible=[a for a in r['all_results'] if a['valid'] and a['gdop']<=100 and a['duration_h']<=b['budget_h']]
            self.assertEqual(b['n_eligible'],len(eligible))
            if not eligible:self.assertIsNone(b['plan_a']);continue
            chosen=r['plans_by_id'][b['plan_a']]
            self.assertEqual(chosen['c3_m2'],min(a['c3_m2'] for a in eligible))
            for k in ('plan_a','plan_b','plan_c'):
                p=r['plans_by_id'][b[k]];self.assertLessEqual(p['duration_h'],b['budget_h'])
                self.assertEqual(len(p['observations']),p['observations_count'])
        self.assertEqual(r['plan_a'],r['plans_by_id'][r['budget_results'][-1]['plan_a']])
        largest=simulation.run_survey({**request,'time_budget_h':8})
        self.assertEqual(r['budget_values_h'],largest['budget_values_h'])

    def test_legacy_single_budget_and_range_setting_remain(self):
        r=simulation.run_survey({**body(3),'max_range_km':.1})
        self.assertEqual(r['budget_values_h'],[8.]);self.assertEqual(r['n_valid'],0);self.assertIsNone(r['plan_a'])
        self.assertEqual(simulation.configuration({**body(),'max_range_km':100})['max_range'],100000.)

    def test_api_returns_budget_choices_and_rejects_bad_bounds(self):
        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app) as client:
            bad=client.post('/api/calculate',json={**body(),'time_budget_min_h':5,'time_budget_max_h':2})
            self.assertEqual(bad.status_code,422)
            r=client.post('/api/calculate',json={**body(3),'time_budget_min_h':.2,'time_budget_max_h':8})
            result=client.get('/api/progress/'+r.json()['task_id']).json()
            self.assertEqual(result['status'],'completed',result)
            self.assertEqual(len(result['result']['budget_results']),10)

if __name__=='__main__':unittest.main()
