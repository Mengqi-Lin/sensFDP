#!/usr/bin/env python3
"""Validate and merge one-file-per-task application results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--scope", choices=("preliminary", "full"), required=True)
    parser.add_argument("--family-size", type=int, required=True)
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted({path.resolve() for path in args.inputs})
    if not paths:
        raise ValueError("no input files were supplied")
    frames = [pd.read_csv(path) for path in paths]
    if any(len(frame) != 1 for frame in frames):
        raise ValueError("every task file must contain exactly one row")
    merged = pd.concat(frames, ignore_index=True)
    if set(merged["scope"]) != {args.scope}:
        raise ValueError("input scope does not match --scope")
    if set(merged["family_size"].astype(int)) != {args.family_size}:
        raise ValueError("input family_size does not match --family-size")
    expected = math.comb(args.family_size, args.subset_size)
    identifiers = merged["subset_index"].astype(int)
    duplicates = sorted(identifiers[identifiers.duplicated()].unique().tolist())
    if duplicates:
        raise ValueError(f"duplicate subset indices: {duplicates[:20]}")
    missing = sorted(set(range(expected)) - set(identifiers.tolist()))
    unexpected = sorted(set(identifiers.tolist()) - set(range(expected)))
    if missing or unexpected:
        raise ValueError(
            f"coverage failure: expected {expected}, observed {len(merged)}, "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    merged.sort_values("subset_index", inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Validated and merged all {expected} {args.scope} subsets into {args.output}")


if __name__ == "__main__":
    main()
