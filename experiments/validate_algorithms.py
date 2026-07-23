"""Paired correctness and runtime check for enumerative and MIQCP procedures.

By default this reproduces the 12 settings used in the manuscript runtime
experiment: two effect patterns, two covariance matrices, and three values of
Gamma.  Both methods are evaluated on the same generated dataset, and a run
fails immediately if their rejection sets differ.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fdp_sensitivity.data import generate_pair_data, prepare_study
from fdp_sensitivity.optimization import (
    enumerative_elementary_rejections,
    individual_worst_pvalues,
    miqcp_elementary_rejections,
)
from fdp_sensitivity.scores import m_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--outcomes", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=[1.25, 1.5, 1.75]
    )
    parser.add_argument(
        "--effect-patterns",
        nargs="+",
        choices=["linear", "half-null"],
        default=["linear", "half-null"],
    )
    parser.add_argument(
        "--correlations", type=float, nargs="+", default=[0.0, 0.2]
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def effect_vector(pattern: str, outcomes: int) -> np.ndarray:
    if pattern == "linear":
        return np.linspace(0.15, 0.35, outcomes)
    if pattern == "half-null":
        return np.r_[
            np.repeat(0.3, (outcomes + 1) // 2),
            np.zeros(outcomes // 2),
        ]
    raise ValueError(f"unknown effect pattern: {pattern}")


def equicorrelation_matrix(outcomes: int, correlation: float) -> np.ndarray:
    lower_limit = -1 / (outcomes - 1) if outcomes > 1 else -1.0
    if not lower_limit < correlation < 1:
        raise ValueError(
            f"correlation must lie in ({lower_limit}, 1) for {outcomes} outcomes"
        )
    covariance = np.full((outcomes, outcomes), correlation, dtype=float)
    np.fill_diagonal(covariance, 1.0)
    return covariance


def main() -> None:
    args = parse_args()
    import gurobipy as gp

    if args.pairs <= 0 or args.outcomes <= 0 or args.replicates <= 0:
        raise ValueError("pairs, outcomes, and replicates must be positive")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if any(gamma < 1 for gamma in args.gammas):
        raise ValueError("every gamma must be at least one")

    gp.setParam("Threads", args.threads)
    rows: list[dict[str, object]] = []
    setting = 0
    for effect_index, pattern in enumerate(args.effect_patterns):
        effects = effect_vector(pattern, args.outcomes)
        for correlation_index, correlation in enumerate(args.correlations):
            covariance = equicorrelation_matrix(args.outcomes, correlation)
            for gamma_index, gamma in enumerate(args.gammas):
                setting += 1
                for replicate in range(args.replicates):
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [args.seed, effect_index, correlation_index, replicate]
                        )
                    )
                    assignment, generated_outcomes, index = generate_pair_data(
                        args.pairs, effects, covariance, 1.0, rng=rng
                    )
                    scores = np.column_stack(
                        [
                            m_scores(generated_outcomes[:, k], index)
                            for k in range(args.outcomes)
                        ]
                    )

                    screening_start = time.perf_counter()
                    study = prepare_study(index, scores, assignment)
                    p_values = individual_worst_pvalues(study, gamma)
                    screening_seconds = time.perf_counter() - screening_start

                    methods = (
                        ("enumerative", enumerative_elementary_rejections),
                        ("miqcp", miqcp_elementary_rejections),
                    )
                    order_index = setting + replicate + gamma_index
                    if order_index % 2:
                        methods = methods[::-1]
                    results: dict[str, set[int]] = {}
                    optimization_seconds: dict[str, float] = {}
                    for name, function in methods:
                        start = time.perf_counter()
                        results[name] = function(
                            study,
                            gamma,
                            alpha=args.alpha,
                            worst_pvalues=p_values,
                        )
                        optimization_seconds[name] = time.perf_counter() - start

                    if results["enumerative"] != results["miqcp"]:
                        gamma_tag = str(gamma).replace(".", "p")
                        failure = args.output.with_suffix(
                            f".setting-{setting}.gamma-{gamma_tag}"
                            f".replicate-{replicate}.failure.npz"
                        )
                        failure.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            failure,
                            assignment=assignment,
                            outcomes=generated_outcomes,
                            index=index,
                            p_values=p_values,
                        )
                        raise RuntimeError(
                            "algorithm mismatch in "
                            f"setting {setting}, replicate {replicate}; "
                            f"inputs saved to {failure}"
                        )

                    enumerative_total = (
                        screening_seconds + optimization_seconds["enumerative"]
                    )
                    miqcp_total = screening_seconds + optimization_seconds["miqcp"]
                    rows.append(
                        {
                            "setting": setting,
                            "replicate": replicate,
                            "seed": args.seed,
                            "pairs": args.pairs,
                            "outcomes": args.outcomes,
                            "effect_pattern": pattern,
                            "correlation": correlation,
                            "gamma": gamma,
                            "screening_seconds": screening_seconds,
                            "enumerative_optimization_seconds": optimization_seconds[
                                "enumerative"
                            ],
                            "miqcp_optimization_seconds": optimization_seconds[
                                "miqcp"
                            ],
                            "enumerative_total_seconds": enumerative_total,
                            "miqcp_total_seconds": miqcp_total,
                            "runtime_ratio": enumerative_total / miqcp_total,
                            "relative_runtime_reduction": 1
                            - miqcp_total / enumerative_total,
                            "number_rejected": len(results["miqcp"]),
                            "rejection_set": "|".join(
                                str(k) for k in sorted(results["miqcp"])
                            ),
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
