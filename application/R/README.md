# R matching workflow: source still needed

`matched_index.csv` was produced locally in R, but the generating R script was
not present in the uploaded application folder. Consequently, this cleaned
bundle preserves the matched-set file but does not pretend to recreate it.

Before the manuscript is finalized, add the exact script and lock-file details
needed to reproduce:

- construction of the CPEA indicator `Z` from the six parental-abuse items;
- recoding of WLS negative missing-value sentinels;
- the four stated missingness indicators, including why five covariates have
  negative sentinel codes but four indicators were used;
- the logistic propensity-score formula and 0.08 caliper;
- the rank-based Mahalanobis distance;
- full-matching constraints, estimand, package versions, and any seed;
- matching weights and the before/after balance table and figure.

The provided `matched_index.csv` is structurally sound: it contains all 3,469
complete-case respondents exactly once in 1,435 sets of sizes two through five,
and each set has either one exposed or one unexposed respondent.
