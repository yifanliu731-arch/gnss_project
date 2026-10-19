"""Risk radius changes design decisions without changing validation evidence."""
import io
import json
import unittest
import zipfile
import numpy as np
from fastapi.testclient import TestClient
from main import app
from joint_simulation import configuration, generate_observations, run_survey
from closed_survey import SurveyTour
from profile_data import load_profiles
from profile_simulation import profile_metrics, tracer_for, truth_observations
from test_joint_simulation import body
from test_profile_simulation import settings


class RiskRadiusTests(unittest.TestCase):
    def test_default_and_invalid_radius(self):
        self.assertEqual(configuration(body())['rho'], 1.)
        for value in (0, .5, 1, 2, 100):
            self.assertEqual(configuration(body(risk_radius=value))['rho'], value)
        for value in (-1, 100.01, None, True, 'bad', float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Risk radius'):
                configuration(body(risk_radius=value))
        with TestClient(app) as client:
            response = client.post('/api/simulation/preview', json=body(risk_radius=-1))
        self.assertEqual(response.status_code, 422)
        self.assertIn('Risk radius', response.json()['detail'])

    def test_munk_radius_preserves_refined_pool_and_observations(self):
        q = body(3)
        q.update(modes=['circle', 'figure_eight'], size_ratios=[.5, 1., 1.7],
                 time_budget_h=5., gdop_threshold=100., noise=.0001,
                 temporal_sigma_mps=.3, sv_grad_n=.00004, refinement_rounds=1)
        saved = {}
        baseline = run_survey(q, export_sink=lambda rows: saved.update(base=rows))
        doubled = run_survey(dict(q, risk_radius=2), export_sink=lambda rows: saved.update(double=rows))
        zero = run_survey(dict(q, risk_radius=0), export_sink=lambda rows: saved.update(zero=rows))
        self.assertGreater(baseline['search']['round_counts'][1], 0)
        self.assertEqual(baseline['config']['risk_radius'], 1.)
        self.assertNotIn('risk_radius', q)  # Defaulting does not mutate callers.
        for result, key, rho in ((baseline, 'base', 1), (doubled, 'double', 2), (zero, 'zero', 0)):
            self.assertEqual(result['risk_radius'], rho)
            self.assertEqual(result['search']['round_counts'], baseline['search']['round_counts'])
            self.assertEqual([r['candidate_id'] for r in saved[key]], [r['candidate_id'] for r in saved['base']])
            for original, row in zip(saved['base'], saved[key]):
                self.assertEqual(original['design_valid'], row['design_valid'])
                if not row['design_valid']:
                    continue
                self.assertEqual(row['c1_m2'], original['c1_m2'])
                for metric in ('c2_m2', 'c3_m2'):
                    expected = original['c1_m2']+rho**2*(original[metric]-original['c1_m2'])
                    self.assertAlmostEqual(row[metric], expected, places=10)
                self.assertEqual(row['_observations'], original['_observations'])
                self.assertEqual(row['validation_status'], original['validation_status'])
                if row['validation_status']=='success':
                    self.assertEqual(row['estimated_beacons'], original['estimated_beacons'])
        nominal = run_survey(dict(q, risk_radius=0, criterion='c1'))
        self.assertEqual(zero['search']['frozen_selection'], nominal['search']['frozen_selection'])

    def test_uploaded_radius_scales_scores_not_truth_or_profile_amplitude(self):
        fields = load_profiles(dict(settings(), amplitude=2.))
        c = configuration(body(3))
        tracer = tracer_for(fields['depth'], fields['nominal'])
        tour = SurveyTour('figure_eight', 1500, c['center'])
        data = generate_observations(tour, c, tracer)
        unit = profile_metrics(data, c, tracer, fields)
        truth, details = truth_observations(data, c, fields, '2020-01-04', 'same-route')
        for rho in (0, .5, 1, 2):
            changed = dict(c, rho=rho)
            scores = profile_metrics(data, changed, tracer, fields)
            for metric in ('c2_m2', 'c3_m2'):
                self.assertAlmostEqual(scores[metric], unit['c1_m2']+rho**2*(unit[metric]-unit['c1_m2']), places=10)
            for metric in ('c1_m2', 'unit_c2_m2', 'unit_c3_m2'):
                self.assertEqual(scores[metric], unit[metric])
            observed, metadata = truth_observations(data, changed, fields, '2020-01-04', 'same-route')
            self.assertEqual(metadata, details)
            for key in ('delays', 'field', 'ship', 'times', 'ids'):
                np.testing.assert_array_equal(observed[key], truth[key])
        self.assertEqual(fields['alpha'], 2.)

    def test_profile_pool_and_default_equivalence(self):
        q = body(3, sound_field_mode='profiles', sound_field=settings())
        q.update(size_ratios=[.5, 1., 1.7], gdop_threshold=100., refinement_rounds=1)
        default = run_survey(q)
        explicit = run_survey(dict(q, risk_radius=1))
        self.assertEqual(default, explicit)
        zero = run_survey(dict(q, risk_radius=0))
        doubled = run_survey(dict(q, risk_radius=2))
        for result in (zero, doubled):
            self.assertEqual([r['candidate_id'] for r in default['all_results']], [r['candidate_id'] for r in result['all_results']])
            for a, b in zip(default['all_results'], result['all_results']):
                if a['design_valid']:
                    self.assertEqual(a['unit_c2_m2'], b['unit_c2_m2'])
                    self.assertEqual(a['unit_c3_m2'], b['unit_c3_m2'])
                    self.assertEqual(a['c1_m2'], b['c1_m2'])
        for comparison in zero['field_validation']['comparisons']:
            self.assertEqual(comparison['c1_id'], comparison['c2_id'])
            self.assertEqual(comparison['c1_id'], comparison['c3_id'])

    def test_api_and_export_preserve_effective_radius(self):
        for profile_mode in (False, True):
            with self.subTest(profile_mode=profile_mode), TestClient(app) as client:
                q = body(3, risk_radius=2.)
                if profile_mode:
                    q.update(sound_field_mode='profiles', sound_field=settings())
                response = client.post('/api/calculate', json=q)
                self.assertEqual(response.status_code, 200)
                task = response.json()['task_id']
                progress = client.get('/api/progress/'+task).json()
                self.assertEqual(progress['status'], 'completed', progress)
                self.assertEqual(progress['result']['risk_radius'], 2.)
                response = client.post('/api/export/'+task, json=dict(scope='all', parts=['observations']))
                self.assertEqual(response.status_code, 200)
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    metadata = json.loads(archive.read('manifest.json'))
                    self.assertEqual(metadata['config']['risk_radius'], 2.)
                    if profile_mode:
                        self.assertEqual(metadata['config']['sound_field']['amplitude'], 1.)


if __name__=='__main__':
    unittest.main()
