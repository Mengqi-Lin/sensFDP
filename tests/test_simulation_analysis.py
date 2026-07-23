from pathlib import Path
import unittest

import numpy as np

from analysis.simulation_analysis import (
    bound_summary,
    load_all_results,
    runtime_summary,
    screening_summary,
    selection_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SimulationAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = load_all_results(PROJECT_ROOT)

    def test_legacy_data_audit_and_key_manuscript_summaries(self):
        results = self.results
        self.assertEqual(results.selection_runs.shape[0], 9_000)
        self.assertEqual(results.bounds.shape[0], (1_000 + 990) * 4)
        self.assertEqual(results.runtime_malformed.shape[0], 1)

        selection, ties = selection_summary(
            results.selection_runs, results.selection_candidates
        )
        observed = {
            (row.effect_correlation, row.method): row.probability_success
            for row in selection.itertuples(index=False)
        }
        self.assertTrue(np.isclose(observed[(-0.2, "Naive selector")], 0.829))
        self.assertTrue(
            np.isclose(
                observed[(-0.2, r"$\Gamma^*(\mathcal{R},1)$ selector")],
                0.9049666667,
            )
        )
        self.assertEqual(ties["tied_replicates"].tolist(), [123, 160, 205])

        screening = screening_summary(results.screening)
        gamma_15_pairs_2000 = screening.query(
            "gamma == 1.5 and pairs == 2000"
        ).iloc[0]
        self.assertTrue(
            np.isclose(gamma_15_pairs_2000.probability_any_ip_call, 0.827)
        )
        self.assertEqual(int(gamma_15_pairs_2000.datasets), 1_000)

        bounds = bound_summary(results.bounds)
        key_bound = bounds.query("rho == 0 and gamma == 1.5").iloc[0]
        self.assertTrue(
            np.isclose(key_bound["exact_probability_at_most_0.75"], 0.571)
        )
        self.assertTrue(
            np.isclose(key_bound["naive_probability_at_most_0.75"], 0.221)
        )

        _, runtime_quality = runtime_summary(results.runtime)
        self.assertEqual(runtime_quality["records"].tolist(), [955, 606])
        self.assertEqual(
            runtime_quality["rejection_fraction_mismatches"].tolist(), [312, 1057]
        )

    def test_audit_preserves_incomplete_and_duplicate_legacy_outputs(self):
        audit = self.results.audit.set_index("file")

        self.assertEqual(audit.loc["Gamma_1.25.csv", "missing_seeds"], 1)
        self.assertEqual(
            audit.loc[
                "rho0.2_I500_Gammalist1_1.25_1.5_1.75_nsim10.csv",
                "missing_seeds",
            ],
            1,
        )
        self.assertEqual(
            audit.loc["closed_testing_equi_K10_I500.csv", "duplicate_keys"], 1
        )
        self.assertEqual(
            audit.loc["closed_testing_equi_K20_I500.csv", "malformed_records"], 1
        )

    def test_complete_files_have_the_full_seed_replicate_grid(self):
        audit = self.results.audit
        gridded = audit["experiment"].isin(("subset selection", "exact versus naive"))
        complete = audit[gridded & (audit["status"] == "complete")]
        self.assertTrue((complete["missing_record_keys"] == 0).all())
        self.assertTrue((complete["unexpected_record_keys"] == 0).all())


if __name__ == "__main__":
    unittest.main()
