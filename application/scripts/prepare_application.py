#!/usr/bin/env python3
"""Prepare the corrected WLS application data and score matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fdp_sensitivity.application import (
    COVARIATE_CODES,
    OUTCOME_NAMES,
    covariate_sentinel_counts,
    prepare_application_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.raw)
    matches = pd.read_csv(args.matches)
    prepared = prepare_application_frames(raw, matches)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared.complete_cases.to_csv(
        args.output_dir / "complete_cases.csv", index=False
    )
    prepared.matched_sample.to_csv(
        args.output_dir / "analysis_sample.csv", index=False
    )
    prepared.score_matrix.to_csv(
        args.output_dir / "application_scores.csv", index=False
    )
    prepared.nominal_pvalues.to_csv(
        args.output_dir / "nominal_pvalues.csv", index=False
    )
    sentinel = covariate_sentinel_counts(prepared.complete_cases)
    sentinel.to_csv(args.output_dir / "covariate_sentinel_audit.csv", index=False)

    set_sizes = prepared.matched_sample.groupby("index", sort=False).size()
    selected = prepared.nominal_pvalues.loc[
        prepared.nominal_pvalues["selected_at_0.05"], "outcome_index"
    ].astype(int)
    audit = {
        "source_rows": int(len(raw)),
        "complete_case_rows": int(len(prepared.complete_cases)),
        "matched_rows": int(len(prepared.matched_sample)),
        "number_matched_sets": int(set_sizes.size),
        "set_size_distribution": {
            str(int(size)): int(count)
            for size, count in set_sizes.value_counts().sort_index().items()
        },
        "treatment_counts": {
            str(int(value)): int(count)
            for value, count in prepared.matched_sample["Z"]
            .value_counts()
            .sort_index()
            .items()
        },
        "number_outcomes": len(OUTCOME_NAMES),
        "preliminary_outcome_indices": selected.tolist(),
        "preliminary_outcomes": [OUTCOME_NAMES[k] for k in selected],
        "covariates_with_negative_sentinels": sentinel.loc[
            sentinel["negative_count"] > 0, "covariate"
        ].tolist(),
        "matching_status": (
            "matched_index.csv is treated as a pinned external R result; "
            "the original R matching script was not included"
        ),
        "score_status": (
            "corrected: spouse is constructed from gc040re rather than gu034rec"
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
