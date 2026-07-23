"""Simulation comparing naive and generalized-sensitivity subset selection."""

from __future__ import annotations

import argparse
import json
import platform
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fdp_sensitivity.data import generate_pair_data, prepare_study
from fdp_sensitivity.optimization import individual_worst_pvalues
from fdp_sensitivity.scores import m_scores
from fdp_sensitivity.sensitivity import generalized_sensitivity_values
from fdp_sensitivity.simulation_settings import (
    MANUSCRIPT_SELECTION_SETTINGS,
    get_selection_setting,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument(
        "--settings",
        type=int,
        nargs="+",
        default=list(MANUSCRIPT_SELECTION_SETTINGS),
        help="zero-based indices from the uploaded settings.py table",
    )
    parser.add_argument("--precision", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path)
    return parser.parse_args()


def _selection_record(method: str, selected: tuple[int, ...]) -> dict[str, object]:
    true_outcomes = {0, 1}
    selected_set = set(selected)
    number_true = len(selected_set & true_outcomes)
    return {
        "method": method,
        "selected": "|".join(str(k) for k in selected),
        "number_true_selected": number_true,
        "at_least_one_true": number_true >= 1,
        "exact_recovery": selected_set == true_outcomes,
    }


def main() -> None:
    args = parse_args()
    import gurobipy as gp

    if args.pairs <= 0 or args.replicates <= 0:
        raise ValueError("pairs and replicates must be positive")
    if args.precision <= 0:
        raise ValueError("precision must be positive")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    gp.setParam("Threads", args.threads)

    candidate_subsets = list(combinations(range(4), 2))
    rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for setting_index in args.settings:
        setting = get_selection_setting(setting_index)
        effects = np.array([setting.effect, setting.effect, 0.0, 0.0])
        covariance = np.array(
            [
                [1, setting.effect_correlation, 0, 0],
                [setting.effect_correlation, 1, 0, 0],
                [0, 0, 1, setting.null_correlation],
                [0, 0, setting.null_correlation, 1],
            ],
            dtype=float,
        )
        for replicate in range(args.replicates):
            rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, setting_index, replicate])
            )
            assignment, outcomes, index = generate_pair_data(
                args.pairs, effects, covariance, setting.gamma_bias, rng=rng
            )
            scores = np.column_stack([m_scores(outcomes[:, k], index) for k in range(4)])
            study = prepare_study(index, scores, assignment)

            p_values = individual_worst_pvalues(study, gamma=1.0)
            naive = tuple(sorted(np.argsort(p_values, kind="stable")[:2].tolist()))

            sensitivity_values = np.array(
                [
                    generalized_sensitivity_values(
                        study,
                        subset,
                        thresholds=[1],
                        alpha=args.alpha,
                        precision=args.precision,
                    )[0]
                    for subset in candidate_subsets
                ]
            )
            maximizers = np.flatnonzero(
                np.isclose(sensitivity_values, sensitivity_values.max())
            )
            maximizer_set = set(maximizers.tolist())
            tie_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, setting_index, replicate, 999])
            )
            robust_index = int(tie_rng.choice(maximizers))
            robust = candidate_subsets[robust_index]

            common = {
                "setting": setting_index,
                "replicate": replicate,
                "seed": args.seed,
                "pairs": args.pairs,
                "effect_correlation": setting.effect_correlation,
                "null_correlation": setting.null_correlation,
                "effect": setting.effect,
                "gamma_bias": setting.gamma_bias,
            }
            rows.append(common | _selection_record("naive", naive))
            robust_record = common | _selection_record("sensitivity", robust)
            robust_record["selected_sensitivity_value"] = sensitivity_values[robust_index]
            robust_record["number_of_tied_maximizers"] = maximizers.size
            rows.append(robust_record)

            if args.candidate_output is not None:
                naive_set = set(naive)
                for subset_index, subset in enumerate(candidate_subsets):
                    candidate_rows.append(
                        common
                        | {
                            "subset": "|".join(str(k) for k in subset),
                            "sensitivity_value": sensitivity_values[subset_index],
                            "is_maximizer": subset_index in maximizer_set,
                            "selected_by_sensitivity": subset_index == robust_index,
                            "selected_by_naive": set(subset) == naive_set,
                            "pvalue_0": p_values[0],
                            "pvalue_1": p_values[1],
                            "pvalue_2": p_values[2],
                            "pvalue_3": p_values[3],
                        }
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    if args.candidate_output is not None:
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(candidate_rows).to_csv(args.candidate_output, index=False)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gurobi": gp.gurobi.version(),
        "gurobi_threads": args.threads,
        "command_arguments": vars(args)
        | {
            "output": str(args.output),
            "candidate_output": (
                str(args.candidate_output)
                if args.candidate_output is not None
                else None
            ),
        },
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
