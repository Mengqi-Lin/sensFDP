import importlib.util
import unittest

import numpy as np

from fdp_sensitivity.data import generate_pair_data, prepare_study
from fdp_sensitivity.optimization import (
    DEFAULT_DECISION_TOLERANCE,
    enumerative_elementary_rejections,
    individual_worst_pvalues,
    miqcp_elementary_rejections,
)
from fdp_sensitivity.scores import outcome_scores


@unittest.skipUnless(importlib.util.find_spec("gurobipy"), "gurobipy is not installed")
class GurobiEquivalenceTests(unittest.TestCase):
    def test_elementary_decisions_match_on_small_problem(self):
        assignment, outcomes, index = generate_pair_data(
            30,
            [0.25, 0.15, 0.0],
            np.eye(3),
            1.0,
            rng=np.random.default_rng(8),
        )
        study = prepare_study(index, outcome_scores(outcomes, index), assignment)
        p_values = individual_worst_pvalues(study, gamma=1.5)
        enumerative = enumerative_elementary_rejections(
            study,
            1.5,
            worst_pvalues=p_values,
            decision_tolerance=DEFAULT_DECISION_TOLERANCE,
        )
        miqcp = miqcp_elementary_rejections(
            study,
            1.5,
            worst_pvalues=p_values,
            decision_tolerance=DEFAULT_DECISION_TOLERANCE,
        )
        self.assertEqual(enumerative, miqcp)


if __name__ == "__main__":
    unittest.main()
