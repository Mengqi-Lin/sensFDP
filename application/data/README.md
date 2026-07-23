# Application data

- `source/wls_analysis_extract.csv`: the uploaded `wls_raw.csv`, renamed to
  clarify that it is already an analysis extract with `Z` constructed. It is
  not the untouched WLS source file.
- `source/variable_info_legacy.txt`: the uploaded informal covariate notes.
  Several descriptions say “confirm”; verify them against the official WLS
  data dictionary before publication.
- `matching/matched_index.csv`: the exact uploaded R matching output. Preserve
  this file because the generating R script was not included.
- `derived/`: outputs of `scripts/prepare_application.py`. These files use the
  corrected spouse score.

The source and matching files are never modified by the Python pipeline.
`derived/audit.json` records expected row counts, matched-set sizes, treatment
counts, the nominally selected outcomes, and the matching-code limitation.
