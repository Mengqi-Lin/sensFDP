#!/usr/bin/env python3
"""Create manuscript-facing summaries from validated corrected results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preliminary", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_indices(value: object) -> tuple[int, ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    return tuple(int(item) for item in parsed)


def validate_results(frame: pd.DataFrame, scope: str, family_size: int) -> pd.DataFrame:
    expected = math.comb(family_size, 4)
    if len(frame) != expected:
        raise ValueError(f"{scope} results contain {len(frame)} rows; expected {expected}")
    if frame["subset_index"].nunique() != expected:
        raise ValueError(f"{scope} results contain duplicate subset indices")
    if set(frame["scope"]) != {scope}:
        raise ValueError(f"{scope} file contains an inconsistent scope")
    result = frame.copy()
    result["indices"] = result["global_indices"].map(parse_indices)
    result["gain"] = result["exact_gsv"] - result["naive_gsv"]
    if (result["gain"] < -1e-10).any():
        bad = result.loc[result["gain"] < -1e-10, "indices"].tolist()[:10]
        raise ValueError(f"exact value is smaller than naive for subsets {bad}")
    return result


def best_row(frame: pd.DataFrame, column: str) -> tuple[pd.Series, int]:
    maximum = float(frame[column].max())
    tied = frame[np.isclose(frame[column], maximum, atol=1e-12)].copy()
    tied.sort_values("global_indices", inplace=True)
    return tied.iloc[0], len(tied)


def manuscript_row(
    label: str,
    selection: pd.Series | None,
    full: pd.Series,
    selection_ties: int | None,
    rule: str,
) -> dict[str, object]:
    return {
        "label": label,
        "selection_rule": rule,
        "global_indices": full["global_indices"],
        "outcomes": full["outcomes"],
        "selection_exact_gsv": (
            np.nan if selection is None else selection["exact_gsv"]
        ),
        "selection_naive_gsv": (
            np.nan if selection is None else selection["naive_gsv"]
        ),
        "selection_ties": selection_ties,
        "full_exact_gsv": full["exact_gsv"],
        "full_naive_gsv": full["naive_gsv"],
        "full_gain": full["gain"],
    }


def main() -> None:
    args = parse_args()
    preliminary = validate_results(
        pd.read_csv(args.preliminary), "preliminary", family_size=13
    )
    full = validate_results(pd.read_csv(args.full), "full", family_size=21)

    r1_selection, r1_ties = best_row(preliminary, "exact_gsv")
    r1_indices = r1_selection["indices"]
    r1_full = full[full["indices"] == r1_indices]
    if len(r1_full) != 1:
        raise ValueError("selected R1 subset is absent from full-family results")

    r2_full, r2_ties = best_row(full, "gain")
    r2_selection = preliminary[preliminary["indices"] == r2_full["indices"]]
    r2_selection_row = None if r2_selection.empty else r2_selection.iloc[0]

    headline = pd.DataFrame(
        [
            manuscript_row(
                "R1",
                r1_selection,
                r1_full.iloc[0],
                r1_ties,
                "maximal exact half-discovery GSV among preliminary size-four subsets",
            ),
            manuscript_row(
                "R2",
                r2_selection_row,
                r2_full,
                r2_ties,
                "maximal exact-minus-naive gain among full-family size-four subsets",
            ),
        ]
    )

    tolerance = float(full["precision"].max()) / 2
    comparison = pd.DataFrame(
        [
            {
                "subsets": len(full),
                "exact_strictly_larger": int((full["gain"] > tolerance).sum()),
                "equal_within_half_precision": int(
                    (full["gain"].abs() <= tolerance).sum()
                ),
                "minimum_gain": float(full["gain"].min()),
                "median_gain": float(full["gain"].median()),
                "maximum_gain": float(full["gain"].max()),
            }
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    headline.to_csv(args.output_dir / "headline_subsets.csv", index=False)
    comparison.to_csv(args.output_dir / "exact_vs_naive_summary.csv", index=False)

    if args.runtime is not None:
        runtime = pd.read_csv(args.runtime)
        if not runtime["decisions_equal"].astype(bool).all():
            raise ValueError("runtime file contains an algorithm disagreement")
        runtime.sort_values(["subset_label", "gamma"], inplace=True)
        runtime.to_csv(args.output_dir / "runtime_table.csv", index=False)

    print(headline.to_string(index=False))
    print("\n" + comparison.to_string(index=False))


if __name__ == "__main__":
    main()
