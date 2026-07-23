# WLS childhood-abuse application

This directory is the cleaned application workflow for the manuscript. It
preserves the supplied WLS analysis extract and R-produced matched-set index,
rebuilds the application data and score matrix deterministically, submits the
heavy sensitivity calculations as safe Slurm arrays, validates complete result
coverage, and creates manuscript-facing summaries in one notebook.

## Important status

The archived numerical sensitivity results are **not publication-final**.
During cleanup, two material problems were found:

1. The legacy notebook constructed both the columns labeled `alcohol` and
   `spouse` from the alcohol outcome. The corrected code constructs spouse from
   `gc040re`. Spouse remains among the 13 outcomes with nominal
   \(p\leq0.05\), but every joint sensitivity value and subset ranking must be
   recomputed.
2. The old full-matching worst-p-value code did not enforce all pairwise
   Rosenbaum probability-ratio constraints. The cleaned jobs use the exact
   full-matching oracle in `fdp_sensitivity.optimization`.

The previous values are retained only under `legacy_unvalidated/` so their
lineage is not lost. Corrected preprocessing outputs are already in
`data/derived/`; corrected Gurobi results are intentionally absent until the
cluster jobs are rerun.

## Layout

```text
application/
├── data/
│   ├── source/                 # pinned WLS analysis extract
│   ├── matching/               # pinned R-produced matched-set index
│   └── derived/                # corrected sample, scores, p-values, audit
├── scripts/                    # preparation, jobs, aggregation, summaries
├── cluster/
│   ├── jobs/                   # Slurm job bodies
│   └── submit/                 # user-facing launchers
├── notebooks/
│   └── application_analysis.ipynb
├── outputs/                    # notebook/manuscript summaries
├── R/                          # status of the missing matching code
├── docs/                       # detailed audit and rerun checklist
└── legacy_unvalidated/         # quarantined historical scores/results
```

## 1. Rebuild the corrected application inputs

From the repository root:

```bash
PYTHONPATH=src python application/scripts/prepare_application.py \
  --raw application/data/source/wls_analysis_extract.csv \
  --matches application/data/matching/matched_index.csv \
  --output-dir application/data/derived
```

This reproduces 3,469 complete cases, 1,435 matched sets, the 21 corrected
score columns, and the 13-outcome preliminary set. It does not rerun matching;
`matched_index.csv` is treated as a pinned external result.

## 2. Rerun generalized sensitivity values on the cluster

Set the cluster environment if its defaults differ:

```bash
export PYTHON_BIN=/path/to/venv/bin/python
export GUROBI_MODULE=gurobi/10.0.2
export MAX_CONCURRENT=100
```

Submit the 715 preliminary-family and 5,985 full-family size-four jobs:

```bash
application/cluster/submit/gsv.sh preliminary
application/cluster/submit/gsv.sh full
```

Each array task writes one CSV. No tasks append concurrently to a shared file.
After each run completes, merge it with exact coverage checks:

```bash
PYTHONPATH=src python application/scripts/merge_gsv_results.py \
  results/raw/application_preliminary_gsv/<run-id>/*.csv \
  --scope preliminary --family-size 13 --subset-size 4 \
  --output application/outputs/preliminary_size4_corrected.csv

PYTHONPATH=src python application/scripts/merge_gsv_results.py \
  results/raw/application_full_gsv/<run-id>/*.csv \
  --scope full --family-size 21 --subset-size 4 \
  --output application/outputs/full_size4_corrected.csv
```

The merge fails if even one subset is absent or duplicated. This prevents a
repeat of the legacy full-family file, which silently omitted nine subsets.

## 3. Select subsets and create the sensitivity-value summary

```bash
PYTHONPATH=src python application/scripts/summarize_application.py \
  --preliminary application/outputs/preliminary_size4_corrected.csv \
  --full application/outputs/full_size4_corrected.csv \
  --output-dir application/outputs
```

The selection rules are explicit:

- `R1` maximizes the exact half-discovery generalized sensitivity value among
  size-four subsets of the 13 preliminary outcomes.
- `R2` maximizes the exact-minus-naive gain among all 5,985 full-family
  size-four subsets.

The script records ties instead of silently selecting the first row.

## 4. Validate the runtime comparison

Read the corrected `R1` and `R2` global indices from
`outputs/headline_subsets.csv`, then submit:

```bash
application/cluster/submit/runtime.sh R1_INDICES R2_INDICES
```

For example, the old, unvalidated indices were `0,1,5,16` and
`1,5,14,16`; do not assume they remain the corrected choices. Merge the 14
jobs only after all complete:

```bash
PYTHONPATH=src python application/scripts/merge_runtime_results.py \
  results/raw/application_runtime/<run-id>/*.csv \
  --output application/outputs/runtime_corrected.csv
```

Every runtime job asserts equality of the enumerative and MIQCP rejection sets.
It records screening time separately as well as total time for each method.

## 5. Run the notebook

Open `notebooks/application_analysis.ipynb` after the corrected cluster outputs
are available. The notebook audits the sample and matches, shows the corrected
nominal p-values, validates result coverage, states the selection rules, and
creates the manuscript tables. It does not run Gurobi interactively.

## Remaining matching gap

The supplied folder did not include the R code that generated
`matched_index.csv` or the balance plot. Five matching covariates contain
negative WLS sentinel codes, while the manuscript states that missing values
and missingness indicators were handled during matching. Recover the exact R
script before final submission so that this step and the balance figure can be
reproduced. See `R/README.md`.
