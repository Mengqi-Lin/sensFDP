#!/usr/bin/env python3
"""Validate and time enumerative versus MIQCP application decisions."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import pandas as pd

from fdp_sensitivity import prepare_study
from fdp_sensitivity.application import OUTCOME_NAMES
from fdp_sensitivity.optimization import (
    enumerative_elementary_rejections,
    individual_worst_pvalues,
    miqcp_elementary_rejections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--subset", required=True, help="Comma-separated global indices")
    parser.add_argument("--subset-label", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--order", choices=("miqcp-first", "enumerative-first"), default="miqcp-first"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = tuple(sorted({int(value) for value in args.subset.split(",")}))
    if not subset or not set(subset).issubset(range(len(OUTCOME_NAMES))):
        raise ValueError("subset contains an invalid outcome index")
    if args.threads <= 0:
        raise ValueError("threads must be positive")

    try:
        import gurobipy as gp
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "run_application_runtime.py requires gurobipy and a working license"
        ) from error
    gp.setParam("Threads", args.threads)

    frame = pd.read_csv(args.scores)
    study = prepare_study(
        frame["index"],
        frame[list(OUTCOME_NAMES)].to_numpy(float),
        frame["Z"],
    )

    start = time.perf_counter()
    worst = individual_worst_pvalues(study, args.gamma)
    screening_seconds = time.perf_counter() - start

    methods = {
        "miqcp": miqcp_elementary_rejections,
        "enumerative": enumerative_elementary_rejections,
    }
    sequence = (
        ("miqcp", "enumerative")
        if args.order == "miqcp-first"
        else ("enumerative", "miqcp")
    )
    decisions: dict[str, set[int]] = {}
    runtimes: dict[str, float] = {}
    for name in sequence:
        started = time.perf_counter()
        decisions[name] = methods[name](
            study,
            args.gamma,
            alpha=args.alpha,
            targets=subset,
            worst_pvalues=worst,
        )
        runtimes[name] = time.perf_counter() - started

    equal = decisions["miqcp"] == decisions["enumerative"]
    if not equal:
        raise RuntimeError(
            "algorithm disagreement: "
            f"MIQCP={sorted(decisions['miqcp'])}, "
            f"enumerative={sorted(decisions['enumerative'])}"
        )
    version = gp.gurobi.version()
    row = {
        "subset_label": args.subset_label,
        "global_indices": json.dumps(subset),
        "outcomes": json.dumps([OUTCOME_NAMES[k] for k in subset]),
        "gamma": args.gamma,
        "screening_seconds": screening_seconds,
        "miqcp_seconds": runtimes["miqcp"],
        "enumerative_seconds": runtimes["enumerative"],
        "miqcp_total_seconds": screening_seconds + runtimes["miqcp"],
        "enumerative_total_seconds": screening_seconds + runtimes["enumerative"],
        "miqcp_rejections": json.dumps(sorted(decisions["miqcp"])),
        "enumerative_rejections": json.dumps(sorted(decisions["enumerative"])),
        "decisions_equal": equal,
        "execution_order": args.order,
        "gurobi_version": ".".join(str(value) for value in version),
        "gurobi_threads": args.threads,
        "python_version": platform.python_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
