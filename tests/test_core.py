import unittest
from inspect import signature
from unittest.mock import patch

import numpy as np
from scipy.stats import chi2

from fdp_sensitivity.data import generate_pair_data, prepare_study
from fdp_sensitivity.multiple_testing import holm_bonferroni, naive_fdp_bound
from fdp_sensitivity.optimization import (
    DEFAULT_DECISION_TOLERANCE,
    _zeta_value,
    enumerative_elementary_rejections,
    exact_fdp_bound,
    local_test_rejects,
    miqcp_elementary_rejections,
    no_hidden_bias_pvalue,
    paired_minimum_zeta,
    paired_two_sided_worst_pvalue,
)
from fdp_sensitivity.sensitivity import compare_generalized_sensitivity_values
from fdp_sensitivity.scores import m_scores, outcome_scores
from fdp_sensitivity.simulation_settings import (
    MANUSCRIPT_SELECTION_SETTINGS,
    get_selection_setting,
)


class MultipleTestingTests(unittest.TestCase):
    def test_holm_stops_at_first_failure(self):
        self.assertEqual(holm_bonferroni([0.001, 0.02, 0.50]), {0, 1})

    def test_naive_bound_uses_requested_subset(self):
        p_values = [0.001, 0.50, 0.60]
        self.assertEqual(naive_fdp_bound(p_values, {0}), 0)
        self.assertEqual(naive_fdp_bound(p_values, {1, 2}), 2)

    def test_all_optimization_paths_share_one_decision_tolerance(self):
        functions = (
            local_test_rejects,
            exact_fdp_bound,
            enumerative_elementary_rejections,
            miqcp_elementary_rejections,
        )
        for function in functions:
            default = signature(function).parameters["decision_tolerance"].default
            self.assertEqual(default, DEFAULT_DECISION_TOLERANCE)


class ScoreTests(unittest.TestCase):
    def test_m_scores_sum_to_zero_within_set(self):
        index = np.repeat([10, 20, 30], [2, 3, 4])
        outcome = np.array([1, 2, 0, 3, 5, 2, 4, 8, 10], dtype=float)
        scores = m_scores(outcome, index)
        for label in np.unique(index):
            self.assertAlmostEqual(scores[index == label].sum(), 0.0)

    def test_automatic_scores_are_floating_point(self):
        index = np.array([0, 0, 1, 1])
        outcomes = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=int)
        scores = outcome_scores(outcomes, index)
        self.assertEqual(scores.dtype.kind, "f")
        self.assertTrue(np.any(np.abs(scores[:, 0]) > 0))


class DataTests(unittest.TestCase):
    def test_nonconsecutive_set_labels(self):
        index = np.array([10, 10, 25, 25])
        scores = np.array([[0.0], [1.0], [2.0], [4.0]])
        assignment = np.array([1, 0, 0, 1])
        study = prepare_study(index, scores, assignment)
        self.assertEqual(study.number_sets, 2)
        self.assertEqual(study.set_sizes, (2, 2))

    def test_pair_generator_is_reproducible(self):
        first = generate_pair_data(10, [0.2, 0.0], np.eye(2), 1.5, rng=np.random.default_rng(4))
        second = generate_pair_data(10, [0.2, 0.0], np.eye(2), 1.5, rng=np.random.default_rng(4))
        for left, right in zip(first, second):
            np.testing.assert_allclose(left, right)


class PairWorstPValueTests(unittest.TestCase):
    def test_reduced_pair_soc_matches_original_zeta(self):
        assignment, outcomes, index = generate_pair_data(
            12,
            [0.25, 0.0, -0.1],
            np.full((3, 3), 0.2) + 0.8 * np.eye(3),
            1.0,
            rng=np.random.default_rng(81),
        )
        study = prepare_study(index, outcome_scores(outcomes, index), assignment)
        gamma = 1.75
        max_tilt = (gamma - 1) / (gamma + 1)
        tilt = np.linspace(-max_tilt, max_tilt, study.number_sets)
        probabilities = [
            np.asarray([0.5 * (1 + x), 0.5 * (1 - x)]) for x in tilt
        ]
        critical_probability = 0.05 / 3
        quantile = float(chi2.ppf(1 - critical_probability, df=1))

        for k in range(study.number_outcomes):
            q = study.scores[:, :2, k]
            midpoint = 0.5 * (q[:, 0] + q[:, 1])
            half_difference = 0.5 * (q[:, 0] - q[:, 1])
            residual = (
                float(study.observed_statistics[k] - midpoint.sum())
                - half_difference @ tilt
            )
            reduced = (
                residual**2
                + quantile * np.sum((half_difference * tilt) ** 2)
                - quantile * np.sum(half_difference**2)
            )
            original = _zeta_value(
                study, k, critical_probability, probabilities
            )
            self.assertAlmostEqual(reduced, original, places=12)

    def test_attainable_mean_gives_one(self):
        index = np.repeat(np.arange(2), 2)
        scores = np.array([[0.0], [1.0], [0.0], [1.0]])
        assignment = np.array([1, 0, 0, 1])
        study = prepare_study(index, scores, assignment)
        value = paired_two_sided_worst_pvalue(study, 0, gamma=2)
        self.assertEqual(value, 1.0)

    def test_heterogeneous_pair_scores_use_interior_probability(self):
        # At Gamma=3, the old endpoint-mean shortcut gives a value just below
        # 0.05, although a feasible interior probability gives a value above
        # 0.05.  This is a regression test for unsafe screening.
        number_pairs = 30
        differences = np.r_[1.0, np.repeat(0.0978291806, number_pairs - 1)]
        index = np.repeat(np.arange(number_pairs), 2)
        scores = np.ravel(
            np.column_stack([np.zeros(number_pairs), differences])
        )[:, None]
        assignment = np.tile([0, 1], number_pairs)
        study = prepare_study(index, scores, assignment)

        q = study.scores[:, :2, 0]
        low = q.min(axis=1)
        high = q.max(axis=1)
        lower_probability = 1 / 4
        upper_probability = 3 / 4
        endpoint_mean = np.sum(
            lower_probability * low + upper_probability * high
        )
        endpoint_variance = np.sum(
            lower_probability * upper_probability * (high - low) ** 2
        )
        endpoint = 2 * __import__("scipy").stats.norm.sf(
            abs(study.observed_statistics[0] - endpoint_mean)
            / np.sqrt(endpoint_variance)
        )
        exact = paired_two_sided_worst_pvalue(
            study,
            0,
            gamma=3,
            tolerance=1e-10,
            decision_tolerance=0,
        )
        self.assertLess(endpoint, 0.05)
        self.assertLess(paired_minimum_zeta(study, 0, 3, 0.05), 0)
        self.assertGreater(exact, 0.05)

    def test_gamma_one_is_analytic_for_larger_sets(self):
        index = np.array([0, 0, 0, 1, 1])
        scores = np.array([[0.0], [1.0], [3.0], [-1.0], [2.0]])
        assignment = np.array([1, 0, 0, 0, 1])
        study = prepare_study(index, scores, assignment)
        observed = study.observed_statistics[0]
        mean = 0.0
        variance = 0.0
        for b, size in enumerate(study.set_sizes):
            values = study.scores[b, :size, 0]
            mean += values.mean()
            variance += np.mean(values**2) - values.mean() ** 2
        expected = 2 * __import__("scipy").stats.norm.sf(
            abs(observed - mean) / np.sqrt(variance)
        )
        self.assertAlmostEqual(no_hidden_bias_pvalue(study, 0), expected)


class SensitivityValueTests(unittest.TestCase):
    def test_exact_and_naive_bisections_share_pvalue_cache(self):
        index = np.repeat(np.arange(3), 2)
        scores = np.tile(np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 3.0, 2.0]]), (3, 1))
        assignment = np.tile([1, 0], 3)
        study = prepare_study(index, scores, assignment)

        class Result:
            def __init__(self, bound):
                self.bound = bound
                self.optimization_calls = 1

        seen = []

        def pvalues(_study, gamma, **_kwargs):
            seen.append(gamma)
            return np.full(4, min(0.99, 0.01 * gamma))

        def exact(_study, _chosen, gamma, _alpha, **_kwargs):
            return Result(3 if gamma >= 1.75 else 0)

        def naive(_pvalues, _chosen, alpha=0.05):
            del alpha
            gamma = float(_pvalues[0] / 0.01)
            return 3 if gamma >= 1.5 else 0

        with (
            patch("fdp_sensitivity.sensitivity.individual_worst_pvalues", pvalues),
            patch("fdp_sensitivity.sensitivity.exact_fdp_bound", exact),
            patch("fdp_sensitivity.sensitivity.naive_fdp_bound", naive),
        ):
            result = compare_generalized_sensitivity_values(
                study, range(4), threshold=2, precision=0.01
            )
        self.assertGreaterEqual(result.exact, 1.75)
        self.assertLess(result.exact, 1.76)
        self.assertGreaterEqual(result.naive, 1.5)
        self.assertLess(result.naive, 1.51)
        self.assertEqual(result.evaluated_gammas, len(set(seen)))


class SelectionSettingTests(unittest.TestCase):
    def test_current_manuscript_setting_indices(self):
        self.assertEqual(MANUSCRIPT_SELECTION_SETTINGS, (6, 0, 3))
        settings = [get_selection_setting(index) for index in (6, 0, 3)]
        self.assertEqual(
            [setting.effect_correlation for setting in settings], [-0.2, 0.0, 0.2]
        )
        self.assertTrue(all(setting.null_correlation == 0.2 for setting in settings))
        self.assertTrue(all(setting.gamma_bias == 1.75 for setting in settings))


if __name__ == "__main__":
    unittest.main()
