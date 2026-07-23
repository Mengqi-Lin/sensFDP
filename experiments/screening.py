"""Simulation evaluating how often singleton screening is inconclusive."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from fdp_sensitivity.data import generate_pair_data, prepare_study
from fdp_sensitivity.optimization import individual_worst_pvalues
from fdp_sensitivity.scores import m_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, nargs="+", default=[500, 1000, 2000, 5000, 10000])
    parser.add_argument("--gammas", type=float, nargs="+", default=[1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--outcomes", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(number_pairs <= 0 for number_pairs in args.pairs):
        raise ValueError("every number of pairs must be positive")
    if any(gamma < 1 for gamma in args.gammas):
        raise ValueError("every gamma must be at least one")
    if args.outcomes <= 0 or args.replicates <= 0:
        raise ValueError("outcomes and replicates must be positive")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")

    effects = np.r_[
        np.repeat(0.3, (args.outcomes + 1) // 2),
        np.zeros(args.outcomes // 2),
    ]
    covariance = np.eye(args.outcomes)
    rows: list[dict[str, object]] = []
    for number_pairs in args.pairs:
        for replicate in range(args.replicates):
            rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, number_pairs, replicate])
            )
            assignment, outcomes, index = generate_pair_data(
                number_pairs, effects, covariance, 1.0, rng=rng
            )
            scores = np.column_stack(
                [m_scores(outcomes[:, k], index) for k in range(args.outcomes)]
            )
            study = prepare_study(index, scores, assignment)
            for gamma in args.gammas:
                p_values = individual_worst_pvalues(study, gamma)
                inconclusive = (args.alpha / args.outcomes < p_values) & (
                    p_values <= args.alpha
                )
                rows.append(
                    {
                        "replicate": replicate,
                        "seed": args.seed,
                        "pairs": number_pairs,
                        "outcomes": args.outcomes,
                        "gamma": gamma,
                        "alpha": args.alpha,
                        "ip_called": bool(inconclusive.any()),
                        "fraction_inconclusive": float(inconclusive.mean()),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command_arguments": vars(args) | {"output": str(args.output)},
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
