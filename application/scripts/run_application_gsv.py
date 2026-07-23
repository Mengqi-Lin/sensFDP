#!/usr/bin/env python3
"""Compute one corrected size-four application sensitivity-value result."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import time
from pathlib import Path

import pandas as pd

from fdp_sensitivity import (
    compare_generalized_sensitivity_values,
    prepare_study,
    select_outcomes,
)
from fdp_sensitivity.application import OUTCOME_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--nominal-pvalues", type=Path, required=True)
    parser.add_argument("--scope", choices=("preliminary", "full"), required=True)
    parser.add_argument("--subset-index", type=int, required=True)
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lower-gamma", type=float, default=1.0)
    parser.add_argument("--upper-gamma", type=float, default=3.0)
    parser.add_argument("--precision", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def indexed_combination(population: tuple[int, ...], size: int, index: int) -> tuple[int, ...]:
    total = math.comb(len(population), size)
    if index < 0 or index >= total:
        raise ValueError(f"subset-index must lie between 0 and {total - 1}")
    return next(itertools.islice(itertools.combinations(population, size), index, None))


def main() -> None:
    args = parse_args()
    if args.subset_size <= 0:
        raise ValueError("subset-size must be positive")
    if args.threads <= 0:
        raise ValueError("threads must be positive")

    try:
        import gurobipy as gp
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "run_application_gsv.py requires gurobipy and a working license"
        ) from error
    gp.setParam("Threads", args.threads)

    score_frame = pd.read_csv(args.scores)
    nominal = pd.read_csv(args.nominal_pvalues)
    missing = sorted(set((*OUTCOME_NAMES, "Z", "index")) - set(score_frame.columns))
    if missing:
        raise ValueError(f"score file is missing columns: {missing}")
    full_study = prepare_study(
        score_frame["index"],
        score_frame[list(OUTCOME_NAMES)].to_numpy(float),
        score_frame["Z"],
    )

    if args.scope == "preliminary":
        global_family = tuple(
            nominal.loc[nominal["selected_at_0.05"].astype(bool), "outcome_index"]
            .astype(int)
            .tolist()
        )
        study = select_outcomes(full_study, global_family)
    else:
        global_family = tuple(range(full_study.number_outcomes))
        study = full_study

    local_subset = indexed_combination(
        tuple(range(len(global_family))), args.subset_size, args.subset_index
    )
    global_subset = tuple(global_family[k] for k in local_subset)
    threshold = args.subset_size // 2

    started = time.perf_counter()
    comparison = compare_generalized_sensitivity_values(
        study,
        local_subset,
        threshold,
        alpha=args.alpha,
        lower_gamma=args.lower_gamma,
        upper_gamma=args.upper_gamma,
        precision=args.precision,
    )
    elapsed = time.perf_counter() - started

    version = gp.gurobi.version()
    row = {
        "scope": args.scope,
        "family_size": len(global_family),
        "subset_index": args.subset_index,
        "subset_size": args.subset_size,
        "threshold": threshold,
        "local_indices": json.dumps(local_subset),
        "global_indices": json.dumps(global_subset),
        "outcomes": json.dumps([OUTCOME_NAMES[k] for k in global_subset]),
        "exact_gsv": comparison.exact,
        "naive_gsv": comparison.naive,
        "exact_right_censored": comparison.exact_right_censored,
        "naive_right_censored": comparison.naive_right_censored,
        "evaluated_gammas": comparison.evaluated_gammas,
        "exact_optimization_calls": comparison.exact_optimization_calls,
        "elapsed_seconds": elapsed,
        "alpha": args.alpha,
        "lower_gamma": args.lower_gamma,
        "upper_gamma": args.upper_gamma,
        "precision": args.precision,
        "gurobi_version": ".".join(str(value) for value in version),
        "gurobi_threads": args.threads,
        "python_version": platform.python_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
