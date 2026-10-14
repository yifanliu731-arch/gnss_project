"""A validation outcome must not change a prospective design recommendation."""
import hashlib
import io
import json
import unittest
import zipfile
from unittest.mock import patch

import joint_simulation as js
from fastapi.testclient import TestClient
from main import app, _json_safe
from result_export import make_archive, select_rows
from test_joint_simulation import body


class FrozenSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = body(3)
        cls.config.update(size_ratios=[.8, 1.2], time_budget_min_h=.1,
                          time_budget_max_h=5, gdop_threshold=100,
                          refinement_rounds=1, noise=.0001, temporal_sigma_mps=.2)

    def test_failure_does_not_reselect_any_plan_budget_or_criterion(self):
        for criterion in ('c1', 'c2', 'c3'):
            with self.subTest(criterion=criterion):
                config = dict(self.config, criterion=criterion)
                baseline = js.run_survey(config)
                self.assertEqual(baseline['criterion'], criterion)
                selected_id = baseline['plan_a']['candidate_id']
                valid_ids = iter(r['candidate_id'] for r in baseline['all_results'] if r['design_valid'])
                original = js.joint_fit

                def fail_selected(data, c, tracer):
                    if next(valid_ids) == selected_id:
                        raise ValueError('Deliberate selected-route validation failure')
                    return original(data, c, tracer)

                saved = dict(kind='simulation')
                with patch.object(js, 'joint_fit', side_effect=fail_selected):
                    failed = js.run_survey(config, export_sink=lambda rows:saved.update(rows=rows))
                saved['result'] = failed
                self.assertEqual(baseline['search'], failed['search'])
                self.assertEqual(failed['search']['selection_sha256'], hashlib.sha256(
                    json.dumps(failed['search']['frozen_selection'], sort_keys=True).encode()).hexdigest())
                self.assertGreater(failed['n_validation_success'], 0)
                self.assertEqual(failed['n_validation_failed'], 1)
                for before, after in zip(baseline['budget_results'], failed['budget_results']):
                    for key in ('plan_a', 'plan_b', 'plan_c', 'n_valid', 'n_eligible'):
                        self.assertEqual(before[key], after[key])
                for before, after in zip(baseline['all_results'], failed['all_results']):
                    self.assertEqual(before['design_valid'], after['design_valid'])
                    for key in ('c1_m2', 'c2_m2', 'c3_m2'):
                        self.assertEqual(before.get(key), after.get(key))
                plan = failed['plan_a']
                self.assertEqual(plan['candidate_id'], selected_id)
                self.assertTrue(plan['valid'])
                self.assertEqual(plan['validation_status'], 'failed')
                self.assertNotIn('estimated_beacons', plan)
                self.assertNotIn('rmse_3d_m', plan)
                self.assertEqual(plan['observations'], baseline['plan_a']['observations'])
                self.assertEqual(plan['trajectory'], baseline['plan_a']['trajectory'])
                self.assertIn(selected_id, [r['candidate_id'] for r in select_rows(saved, dict(scope='top10'))])
                for scope in ('all', 'top10', 'custom'):
                    request = dict(scope=scope, ids=[selected_id], parts=['observations','residuals'])
                    z = zipfile.ZipFile(io.BytesIO(make_archive(saved, request, _json_safe)))
                    row = next(r for r in json.loads(z.read('results.json'))['results'] if r['candidate_id']==selected_id)
                    self.assertEqual(row['validation_status'], 'failed')
                    self.assertEqual(row['observations'], plan['observations'])
                    self.assertEqual(row['residuals'], [])
                self.assertIsNone(failed['budget_results'][0]['plan_a'])

    def test_all_failed_fits_are_complete_api_results_with_designs_and_exports(self):
        with patch.object(js, 'joint_fit', side_effect=ValueError('Deliberate validation failure')):
            with TestClient(app) as client:
                response = client.post('/api/calculate', json=body(3))
                task_id = response.json()['task_id']
                data = client.get('/api/progress/'+task_id).json()
                self.assertEqual(data['status'], 'completed')
                result = data['result']
                self.assertEqual(result['n_valid'], 1)
                self.assertEqual(result['n_validation_success'], 0)
                self.assertEqual(result['n_validation_failed'], 1)
                self.assertEqual(result['plan_a']['validation_status'], 'failed')
                self.assertEqual(client.post('/api/export/'+task_id, json=dict(scope='top10')).status_code, 200)


if __name__ == '__main__':
    unittest.main()
