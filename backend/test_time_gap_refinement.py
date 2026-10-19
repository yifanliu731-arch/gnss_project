"""Local complete-route time-gap stopping, independent of validation outcomes."""
import io
import json
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from main import app
from joint_simulation import configuration, run_survey, time_gap_refinements
from test_joint_simulation import body
from test_profile_simulation import settings


def row(ratio, score, valid=True):
    return dict(type='figure_eight', angle_deg=0., ratio=ratio, duration_h=ratio,
                design_valid=valid, gdop=1., c1_m2=score, c2_m2=score, c3_m2=score,
                unit_c2_m2=score, unit_c3_m2=score)


class TimeGapTests(unittest.TestCase):
    def test_default_units_and_validation(self):
        c = configuration(body(refinement_mode='time_gap'))
        self.assertEqual(c['refinement_time_gap_min'], 5.)
        self.assertEqual(c['refinement_mode'], 'time_gap')
        for value in (0, -.1, 2881, None, True, 'bad', float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'minutes'):
                configuration(body(refinement_mode='time_gap', refinement_time_gap_min=value))
        for value in (.1, 5., 15., 2880.):
            self.assertEqual(configuration(body(refinement_mode='time_gap', refinement_time_gap_min=value))['refinement_time_gap_min'], value)
        with self.assertRaisesRegex(ValueError, 'Refinement mode'):
            configuration(body(refinement_mode='unknown'))
        # A hidden/inactive target never changes a fixed-round run.
        self.assertEqual(configuration(body(refinement_rounds=0, refinement_time_gap_min=None))['refinement_rounds'], 0)

    def test_existing_interval_width_and_inclusive_boundary(self):
        records = [row(1.,3.), row(2.,1.), row(4.,3.,valid=False)]
        proposals, largest = time_gap_refinements(records,[3.],100.,60.)
        self.assertEqual(largest, 120.)
        # 1--2 already has a 60-min gap; only 2--4 is bisected.
        self.assertEqual(len(proposals),1)
        self.assertAlmostEqual(proposals[0][1],np.sqrt(8))
        records.append(row(np.sqrt(8),2.))
        proposals, largest = time_gap_refinements(records,[3.],100.,60.)
        self.assertEqual(proposals,[])
        self.assertEqual(largest,60.)

    def test_new_winner_moves_refinement_neighbourhood(self):
        records = [row(1.,3.), row(2.,1.), row(4.,3.), row(np.sqrt(8),.5)]
        proposals, largest = time_gap_refinements(records,[5.],100.,60.)
        self.assertEqual(len(proposals),1)
        self.assertAlmostEqual(proposals[0][1],np.sqrt(4*np.sqrt(8)))
        self.assertAlmostEqual(largest,(4-np.sqrt(8))*60)
        # Full duration, not radius ratio or elapsed computation time, sets the gap.
        faster = [dict(r,duration_h=r['duration_h']/10) for r in records]
        self.assertEqual(time_gap_refinements(faster,[5.],100.,60.)[0],[])

    def test_target_reached_and_seed_candidates_retained_in_both_modes(self):
        for profiles in (False,True):
            with self.subTest(profiles=profiles):
                q=body(3)
                q.update(size_ratios=[.5,1.],gdop_threshold=100.,refinement_mode='time_gap',refinement_time_gap_min=1.)
                if profiles:
                    q.update(sound_field_mode='profiles',sound_field=settings())
                result=run_survey(q)
                search=result['search']
                self.assertTrue(search['target_met'])
                self.assertEqual(search['stop_reason'],'time_gap_reached')
                self.assertGreater(search['refinement_rounds_completed'],3)
                self.assertLessEqual(search['max_winner_gap_min'],1.+1e-9)
                self.assertEqual(result['config']['refinement_time_gap_min'],1.)
                self.assertNotIn('refinement_rounds',result['config'])
                self.assertTrue({.5,1.} <= {r['ratio'] for r in result['all_results']})
                proposals, gap=time_gap_refinements(result['all_results'],[8.],100.,1.)
                self.assertEqual(proposals,[])
                self.assertEqual(gap,search['max_winner_gap_min'])
                # Shrinking rho keeps the training-only refinement pool unchanged.
                zero=run_survey(dict(q,risk_radius=0))
                self.assertEqual([r['candidate_id'] for r in result['all_results']],
                                 [r['candidate_id'] for r in zero['all_results']])

    def test_empty_neighbourhood_is_not_claimed_as_target_reached(self):
        q=body(3,refinement_mode='time_gap')
        result=run_survey(q)
        self.assertFalse(result['search']['target_met'])
        self.assertIsNone(result['search']['max_winner_gap_min'])
        self.assertEqual(result['search']['stop_reason'],'no_refinement_intervals')
        self.assertEqual(result['search']['refinement_rounds_completed'],0)

    def test_large_target_needs_no_round_and_guard_is_explicit(self):
        q=body(3,refinement_mode='time_gap',gdop_threshold=100.)
        q['size_ratios']=[.5,1.]
        result=run_survey(dict(q,refinement_time_gap_min=1000.))
        self.assertTrue(result['search']['target_met'])
        self.assertEqual(result['search']['refinement_rounds_completed'],0)
        with patch('joint_simulation.MAX_TIME_GAP_ROUNDS',0):
            with self.assertRaisesRegex(ValueError,'Target time gap was not reached'):
                run_survey(q)

    def test_api_and_exports_preserve_default_target_and_stopping_evidence(self):
        for profiles in (False,True):
            with self.subTest(profiles=profiles), TestClient(app) as client:
                q=body(3,refinement_mode='time_gap',gdop_threshold=100.)
                q['size_ratios']=[.5,1.]
                if profiles:
                    q.update(sound_field_mode='profiles',sound_field=settings())
                response=client.post('/api/calculate',json=q)
                task=response.json()['task_id']
                completed=client.get('/api/progress/'+task).json()
                self.assertEqual(completed['status'],'completed',completed)
                self.assertEqual(completed['result']['config']['refinement_time_gap_min'],5.)
                archive=client.post('/api/export/'+task,json=dict(scope='all',parts=[]))
                with zipfile.ZipFile(io.BytesIO(archive.content)) as z:
                    manifest=json.loads(z.read('manifest.json'))
                    self.assertEqual(manifest['config']['refinement_time_gap_min'],5.)
                    self.assertTrue(manifest['run_metadata']['search']['target_met'])
                    self.assertLessEqual(manifest['run_metadata']['search']['max_winner_gap_min'],5.+1e-9)


if __name__=='__main__':
    unittest.main()
