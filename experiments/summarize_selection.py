"""Summarize the subset-selection simulation with informative criteria."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path) for path in args.inputs]
    expected_columns = list(frames[0].columns)
    if any(list(frame.columns) != expected_columns for frame in frames[1:]):
        raise ValueError("selection input files do not have identical columns")
    data = pd.concat(frames, ignore_index=True)
    key = ["seed", "setting", "replicate", "method"]
    if data.duplicated(key).any():
        duplicate = data.loc[data.duplicated(key, keep=False), key].head()
        raise ValueError(f"duplicate simulation rows detected:\n{duplicate}")
    summary = (
        data.groupby(["effect_correlation", "method"], as_index=False)
        .agg(
            probability_at_least_one=("at_least_one_true", "mean"),
            probability_exact_recovery=("exact_recovery", "mean"),
            average_number_true=("number_true_selected", "mean"),
            replicates=("replicate", "size"),
        )
        .sort_values(["effect_correlation", "method"])
    )
    # For a uniformly random size-two subset of four outcomes, these references
    # are 5/6, 1/6, and 1, respectively.  They are recorded as context rather
    # than presented as a competing scientific selector.
    summary["chance_probability_at_least_one"] = 5 / 6
    summary["chance_probability_exact_recovery"] = 1 / 6
    summary["chance_average_number_true"] = 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
