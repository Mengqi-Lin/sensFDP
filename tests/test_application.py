from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from fdp_sensitivity import select_outcomes
from fdp_sensitivity.application import (
    OUTCOME_NAMES,
    covariate_sentinel_counts,
    prepare_application_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPLICATION = PROJECT_ROOT / "application"


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = pd.read_csv(APPLICATION / "data/source/wls_analysis_extract.csv")
        matches = pd.read_csv(APPLICATION / "data/matching/matched_index.csv")
        cls.prepared = prepare_application_frames(raw, matches)

    def test_application_preparation_and_nominal_screen(self):
        prepared = self.prepared
        self.assertEqual(len(prepared.complete_cases), 3469)
        self.assertEqual(len(prepared.matched_sample), 3469)
        self.assertEqual(prepared.study.number_sets, 1435)
        self.assertEqual(prepared.study.number_outcomes, 21)
        self.assertEqual(tuple(prepared.score_matrix.columns[:21]), OUTCOME_NAMES)
        selected = prepared.nominal_pvalues.loc[
            prepared.nominal_pvalues["selected_at_0.05"], "outcome_index"
        ].tolist()
        self.assertEqual(
            selected, [0, 1, 5, 6, 7, 9, 11, 14, 15, 16, 18, 19, 20]
        )

    def test_corrected_spouse_score_and_legacy_lineage(self):
        prepared = self.prepared
        legacy = pd.read_csv(
            APPLICATION / "legacy_unvalidated/scores/Whole_Qmat.csv"
        )
        corrected = prepared.score_matrix
        for name in set(OUTCOME_NAMES) - {"spouse"}:
            np.testing.assert_allclose(
                corrected[name], legacy[name], rtol=0, atol=1e-12
            )
        self.assertFalse(np.allclose(corrected["spouse"], legacy["spouse"]))
        raw_correlation = np.corrcoef(
            prepared.matched_sample["alcohol"], prepared.matched_sample["spouse"]
        )[0, 1]
        corrected_correlation = np.corrcoef(
            corrected["alcohol"], corrected["spouse"]
        )[0, 1]
        legacy_correlation = np.corrcoef(legacy["alcohol"], legacy["spouse"])[0, 1]
        self.assertLess(abs(raw_correlation), 0.1)
        self.assertLess(abs(corrected_correlation), 0.1)
        self.assertGreater(legacy_correlation, 0.95)

    def test_matching_structure_and_covariate_sentinel_audit(self):
        prepared = self.prepared
        sizes = prepared.matched_sample.groupby("index").size()
        self.assertEqual(
            sizes.value_counts().sort_index().to_dict(),
            {2: 1092, 3: 175, 4: 80, 5: 88},
        )
        treated = prepared.matched_sample.groupby("index")["Z"].sum()
        self.assertTrue(((treated == 1) | ((sizes - treated) == 1)).all())
        sentinel = covariate_sentinel_counts(prepared.complete_cases)
        observed = sentinel.set_index("covariate")["negative_count"].to_dict()
        self.assertEqual(observed["hb042re"], 615)
        self.assertEqual(observed["piearl"], 429)
        self.assertEqual(observed["sesp57"], 84)
        self.assertEqual(observed["edfa57q"], 222)
        self.assertEqual(observed["edmo57q"], 190)

    def test_select_outcomes_preserves_requested_order(self):
        prepared = self.prepared
        selected = select_outcomes(prepared.study, [5, 1, 16])
        self.assertEqual(selected.number_outcomes, 3)
        np.testing.assert_allclose(
            selected.scores[:, :, 0], prepared.study.scores[:, :, 5]
        )
        np.testing.assert_allclose(
            selected.scores[:, :, 1], prepared.study.scores[:, :, 1]
        )
        np.testing.assert_allclose(
            selected.observed_statistics,
            prepared.study.observed_statistics[[5, 1, 16]],
        )


if __name__ == "__main__":
    unittest.main()
