# Audit of the uploaded application folder

## Workflow recovered from the notebooks

```text
wls_raw.csv
  -> outcome recoding and complete-case restriction
  -> external R full matching -> matched_index.csv
  -> derived social-support and obesity outcomes
  -> 21 matched-set score columns
  -> Gamma=1 screen -> 13 preliminary outcomes
  -> cluster sensitivity calculations
  -> selected-family and full-family summaries
  -> manuscript subsets and runtime table
```

The original work was spread across eight stateful notebooks, eight Python or
shell computation drivers, and many derived CSVs. Several required inputs were
outside the folder, and multiple array jobs appended concurrently to the same
CSV.

## Material findings

1. **Spouse score constructed from alcohol.** The score notebook assigned
   `Qmat[:,1]` from `PO[:,0]`. The corrected line uses `PO[:,1]`. Raw
   alcohol/spouse correlation is about -0.069 and the corrected score
   correlation is about -0.079; the two legacy score columns correlate 0.959.
   Spouse remains in the nominal 13-outcome set, but the joint results and
   rankings require a complete rerun.
2. **Old full-matching screening was not exact.** The alternative solver used
   coordinate bounds that do not impose every pairwise probability-ratio
   restriction for sets larger than two. The cleaned jobs use the exact convex
   minimum-zeta oracle with the full Rosenbaum constraints.
3. **Nine full-family jobs are absent.** The historical files contain 5,976
   rather than all \({21\choose4}=5,985\) size-four subsets. The new merger
   rejects incomplete coverage.
4. **Matching cannot be reproduced from the folder.** The R script, formula,
   package versions, weights, and balance code are absent. Five covariates
   contain negative WLS sentinel codes in 1,195 rows; the unseen R workflow
   must be checked to confirm how these and the stated four missingness
   indicators were handled.
5. **Historical runtime results conflict.** `runtimeComp0.csv` generated the
   draft table, while a later `runtimeComp.csv` gives markedly different
   timings. Neither records adequate solver/hardware metadata, and both depend
   on the erroneous scores.
6. **Two ordinal score exceptions are undocumented.** The code uses fixed-scale
   M-scores with outer trims 4 and 2 for alcohol and spouse, respectively,
   while the manuscript describes only binary scores and continuous M-scores
   with trim 2.5. Preserve or change this choice deliberately and document it.

## File disposition

| Original file/group | Disposition in cleaned workflow |
|---|---|
| `wls_raw.csv` | Preserved as `data/source/wls_analysis_extract.csv` |
| `matched_index.csv` | Preserved unchanged in `data/matching/` |
| `variable_info.txt` | Preserved as tentative legacy notes |
| `wls_data_cleaned.csv`, `covariate_data.csv`, `PO_data.csv` | Regenerated; old copies omitted |
| `outcome_data.csv` | Omitted; exact logical duplicate of `PO_data.csv` |
| `Whole_Qmat.csv`, `Selected_Qmat.csv` | Old copies quarantined; corrected matrix regenerated |
| `Selected_gSval.csv` and whole-size-four results | Quarantined; rerun required |
| `CPEA_gSval.csv` | Quarantined as an orphaned historical fork |
| `runtimeComp*.csv`, `vR.csv` | Quarantined as diagnostics |
| `subsets.txt` | Replaced by deterministic combination indexing |
| all old `.py`, `.sh`, and `.ipynb` files | Replaced by the cleaned scripts/jobs/notebook |
| `.ipynb_checkpoints/`, `__pycache__/`, `__MACOSX/` | Removed |

## Historical draft values

For provenance only, the old files give full-family exact/naive values 1.46 / 
1.42 for `{alcohol, spouse, anger, agreeableness}` and 1.41 / 1.32 for
`{spouse, anger, neuroticism, agreeableness}`. These are not validated values.

Among the 5,976 stored full-family subsets, the exact value is no smaller than
the naive value for every subset, strictly larger for 4,000, and equal for
1,976. Thus even apart from rerunning, the draft phrase “larger across all
subsets” should be “no smaller across all subsets.”

## Required rerun order

1. Recover and verify the R matching and balance workflow.
2. Regenerate the corrected scores and nominal p-values.
3. Compute all 715 preliminary-family size-four results.
4. Select `R1` by the stated maximum rule.
5. Compute all 5,985 full-family size-four exact and naive results.
6. Select `R2` by the stated maximum-gain rule and generate the summaries.
7. Rerun the 14 runtime configurations with paired equality checks.
8. Run the application notebook and update the manuscript values and wording.
