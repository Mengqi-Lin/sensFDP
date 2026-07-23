# Quarantined historical application outputs

These files are retained for provenance only. They reproduce the values in the
current draft, but they must not be used as corrected manuscript results.

The legacy score matrices contain a coding error: the column labeled `spouse`
is a trim-2 transformation of alcohol rather than a score from the spouse
outcome. The old full-matching worst-p-value calculation also did not enforce
the full Rosenbaum probability-ratio constraints. In addition, the historical
full-family size-four table contains 5,976 of the required 5,985 subsets.

- `scores/`: the exact uploaded score matrices with the spouse error.
- `results/Selected_gSval.csv`: internally complete 13-outcome result table,
  but downstream of the erroneous score matrix.
- `results/CPEA_gSval.csv`: an orphaned historical fork with no surviving
  generating input; it differs materially from `Selected_gSval.csv`.
- `results/gSval_whole_size4_closed.csv` and
  `results/gSval_Whole_size4.csv`: incomplete full-family results.
- `results/runtimeComp0.csv`: the run used for the draft runtime table.
- `results/runtimeComp.csv` and `results/vR.csv`: later diagnostics, not final
  analyses.

The original scratch notebooks, checkpoints, bytecode, recovery lists, and
append-to-shared-CSV job scripts were intentionally excluded from the cleaned
application directory. The uploaded `application.zip` remains unchanged.
