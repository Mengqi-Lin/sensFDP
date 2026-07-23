"""Simulation comparing exact and naive FDP upper bounds."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from fdp_sensitivity.data import generate_pair_data, prepare_study
from fdp_sensitivity.multiple_testing import naive_fdp_bound
from fdp_sensitivity.optimization import exact_fdp_bound, individual_worst_pvalues
from fdp_sensitivity.scores import m_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--gammas", type=float, nargs="+", default=[1, 1.25, 1.5, 1.75])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import gurobipy as gp

    if args.pairs <= 0 or args.replicates <= 0:
        raise ValueError("pairs and replicates must be positive")
    if not -1 / 3 < args.rho < 1:
        raise ValueError("rho must lie in (-1/3, 1) for four outcomes")
    if any(gamma < 1 for gamma in args.gammas):
        raise ValueError("every gamma must be at least one")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    gp.setParam("Threads", args.threads)

    number_outcomes = 4
    effects = np.linspace(0.15, 0.35, number_outcomes)
    covariance = np.full((number_outcomes, number_outcomes), args.rho)
    np.fill_diagonal(covariance, 1.0)
    rows: list[dict[str, object]] = []
    for replicate in range(args.replicates):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, replicate]))
        assignment, outcomes, index = generate_pair_data(
            args.pairs, effects, covariance, 1.0, rng=rng
        )
        scores = np.column_stack(
            [m_scores(outcomes[:, k], index) for k in range(number_outcomes)]
        )
        study = prepare_study(index, scores, assignment)
        for gamma in args.gammas:
            p_values = individual_worst_pvalues(study, gamma)
            exact_result = exact_fdp_bound(
                study,
                range(number_outcomes),
                gamma,
                alpha=args.alpha,
                worst_pvalues=p_values,
            )
            exact = exact_result.bound
            naive = naive_fdp_bound(
                p_values, range(number_outcomes), alpha=args.alpha
            )
            if exact > naive:
                raise RuntimeError(
                    f"exact bound exceeds naive bound in replicate {replicate}, gamma={gamma}"
                )
            if np.isclose(gamma, 1) and exact != naive:
                raise RuntimeError(
                    f"methods disagree at Gamma=1 in replicate {replicate}"
                )
            rows.append(
                {
                    "replicate": replicate,
                    "seed": args.seed,
                    "pairs": args.pairs,
                    "rho": args.rho,
                    "gamma": gamma,
                    "exact_bound": exact,
                    "naive_bound": naive,
                    "optimization_calls": exact_result.optimization_calls,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gurobi": gp.gurobi.version(),
        "gurobi_threads": args.threads,
        "command_arguments": vars(args) | {"output": str(args.output)},
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
