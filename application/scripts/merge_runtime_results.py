#!/usr/bin/env python3
"""Validate and merge corrected application runtime jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = [pd.read_csv(path) for path in sorted(set(args.inputs))]
    if any(len(frame) != 1 for frame in frames):
        raise ValueError("every runtime task file must contain exactly one row")
    merged = pd.concat(frames, ignore_index=True)
    expected = {(label, gamma) for label in ("R1", "R2") for gamma in (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)}
    observed = set(zip(merged["subset_label"], merged["gamma"].round(10)))
    if observed != expected or len(merged) != len(expected):
        raise ValueError(
            f"runtime coverage failure: missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    if not merged["decisions_equal"].astype(bool).all():
        raise ValueError("enumerative and MIQCP decisions disagree")
    merged.sort_values(["subset_label", "gamma"], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Validated and merged 14 runtime jobs into {args.output}")


if __name__ == "__main__":
    main()
