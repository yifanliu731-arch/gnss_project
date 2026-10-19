"""Entered round counts extend the existing cumulative size search."""
import unittest
from unittest.mock import patch
from joint_simulation import configuration, run_survey
from test_joint_simulation import body
from test_profile_simulation import settings


class RefinementRoundTests(unittest.TestCase):
    def test_integer_limits_and_default(self):
        self.assertEqual(configuration(body())['refinement_rounds'], 1)
        for rounds in (0, 1, 2, 3, 7, 10):
            self.assertEqual(configuration(body(refinement_rounds=rounds))['refinement_rounds'], rounds)
        for rounds in (-1, 11, 2.5, None, True, 'bad', float('inf')):
            with self.subTest(rounds=rounds), self.assertRaisesRegex(ValueError, 'integer from 0 to 10'):
                configuration(body(refinement_rounds=rounds))

    def test_three_rounds_retain_previous_candidates_in_both_modes(self):
        for profiles in (False, True):
            with self.subTest(profiles=profiles):
                q = body(3)
                q.update(size_ratios=[.5, 1.], gdop_threshold=100, refinement_rounds=2)
                if profiles:
                    q.update(sound_field_mode='profiles', sound_field=settings())
                previous = run_survey(q)
                current = run_survey(dict(q, refinement_rounds=3))
                before = {r['candidate_id'] for r in previous['all_results']}
                after = {r['candidate_id'] for r in current['all_results']}
                self.assertLess(before, after)
                self.assertEqual(len(current['search']['round_counts']), 4)
                self.assertGreater(current['search']['round_counts'][-1], 0)
                self.assertEqual(current['config']['refinement_rounds'], 3)
                self.assertEqual(max(r['search_round'] for r in current['all_results']), 3)
                self.assertEqual(len(after), current['n_total'])
                self.assertTrue(all(.5<=r['ratio']<=1 for r in current['all_results']))
                self.assertTrue(all(r['type']=='figure_eight' and r['angle_deg']==0 for r in current['all_results']))

    def test_candidate_limit_still_fails_explicitly(self):
        too_many = [('figure_eight', .5+i/10000, 0., 'risk_refinement') for i in range(2001)]
        for profiles in (False, True):
            q = body(3, refinement_rounds=3)
            if profiles:
                q.update(sound_field_mode='profiles', sound_field=settings())
            with self.subTest(profiles=profiles), patch('joint_simulation.initial_specs', return_value=too_many):
                with self.assertRaisesRegex(ValueError, 'exceeds 2000 candidates'):
                    run_survey(q)


if __name__=='__main__':
    unittest.main()
