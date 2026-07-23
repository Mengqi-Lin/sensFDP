"""Parse legacy simulation CSVs and generate manuscript-facing outputs.

The raw files are intentionally left unchanged.  Their array-valued cells are
expanded into tidy in-memory tables before any summaries or figures are made.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from fdp_sensitivity.simulation_settings import get_selection_setting


PAIR_COUNTS = (500, 1000, 2000, 5000, 10000)
BOUND_GAMMAS = (1.0, 1.25, 1.5, 1.75)
RUNTIME_GAMMAS = (1.25, 1.5, 1.75)
CANDIDATE_SUBSETS = tuple(itertools.combinations(range(4), 2))
TRUE_OUTCOMES = frozenset((0, 1))


@dataclass(frozen=True)
class SimulationResults:
    selection_runs: pd.DataFrame
    selection_candidates: pd.DataFrame
    screening: pd.DataFrame
    bounds: pd.DataFrame
    runtime: pd.DataFrame
    audit: pd.DataFrame
    runtime_malformed: pd.DataFrame


def _numpy_array(text: str, expected_length: int) -> np.ndarray:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError("expected a bracketed NumPy-style array")
    values = np.fromstring(stripped[1:-1], sep=" ")
    if values.size != expected_length or np.any(~np.isfinite(values)):
        raise ValueError(f"expected {expected_length} finite values")
    return values


def _audit_row(
    experiment: str,
    path: Path,
    *,
    records: int,
    valid_records: int,
    seeds: set[int],
    expected_seeds: set[int],
    duplicate_keys: int,
    malformed_records: int,
    missing_record_keys: int = 0,
    unexpected_record_keys: int = 0,
    note: str = "",
) -> dict[str, object]:
    missing = sorted(expected_seeds - seeds)
    extra = sorted(seeds - expected_seeds)

    def describe_seed_issue(label: str, values: list[int]) -> str:
        if len(values) <= 12:
            return f"{label}: {','.join(map(str, values))}"
        preview = ",".join(map(str, values[:12]))
        return f"{label}: {len(values)} ({preview},...)"

    issues = []
    if missing:
        issues.append(describe_seed_issue("missing seeds", missing))
    if extra:
        issues.append(describe_seed_issue("extra seeds", extra))
    if duplicate_keys:
        issues.append(f"duplicate keys: {duplicate_keys}")
    if missing_record_keys:
        issues.append(f"missing record keys: {missing_record_keys}")
    if unexpected_record_keys:
        issues.append(f"unexpected record keys: {unexpected_record_keys}")
    if malformed_records:
        issues.append(f"malformed records: {malformed_records}")
    if note:
        issues.append(note)
    return {
        "experiment": experiment,
        "file": path.name,
        "records": records,
        "valid_records": valid_records,
        "distinct_seeds": len(seeds),
        "expected_seeds": len(expected_seeds),
        "missing_seeds": len(missing),
        "extra_seeds": len(extra),
        "duplicate_keys": duplicate_keys,
        "missing_record_keys": missing_record_keys,
        "unexpected_record_keys": unexpected_record_keys,
        "malformed_records": malformed_records,
        "status": "complete" if not issues else "warning",
        "note": "; ".join(issues),
    }


def load_selection(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    runs: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    pattern = re.compile(r"I(?P<pairs>\d+)_setting(?P<setting>\d+)\.csv$")
    for path in sorted(directory.glob("I*_setting*.csv")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        pairs = int(match.group("pairs"))
        setting_index = int(match.group("setting"))
        setting = get_selection_setting(setting_index)
        keys: list[tuple[int, int]] = []
        records = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for records, row in enumerate(csv.reader(handle), start=1):
                if len(row) != 5:
                    raise ValueError(f"{path.name}, record {records}: expected 5 fields")
                seed, replicate = int(row[0]), int(row[1])
                p_values = _numpy_array(row[2], 4)
                gsv_r0 = _numpy_array(row[3], 6)
                gsv_r1 = _numpy_array(row[4], 6)
                if np.any((p_values < 0) | (p_values > 1)):
                    raise ValueError(f"{path.name}, record {records}: invalid p-value")
                if np.any(gsv_r0 > gsv_r1 + 1e-12):
                    raise ValueError(f"{path.name}, record {records}: nonmonotone GSV")
                keys.append((seed, replicate))
                common = {
                    "pairs": pairs,
                    "setting": setting_index,
                    "seed": seed,
                    "replicate": replicate,
                    "effect": setting.effect,
                    "effect_correlation": setting.effect_correlation,
                    "null_correlation": setting.null_correlation,
                    "gamma_bias": setting.gamma_bias,
                }
                runs.append(
                    common
                    | {f"pvalue_{k}": float(p_values[k]) for k in range(4)}
                )
                for subset_index, subset in enumerate(CANDIDATE_SUBSETS):
                    candidates.append(
                        common
                        | {
                            "subset_index": subset_index,
                            "subset": "|".join(map(str, subset)),
                            "gsv_r0": float(gsv_r0[subset_index]),
                            "gsv_r1": float(gsv_r1[subset_index]),
                        }
                    )
        seeds = {key[0] for key in keys}
        expected_keys = set(itertools.product(range(100), range(10)))
        observed_keys = set(keys)
        audit.append(
            _audit_row(
                "subset selection",
                path,
                records=records,
                valid_records=records,
                seeds=seeds,
                expected_seeds=set(range(100)),
                duplicate_keys=len(keys) - len(set(keys)),
                malformed_records=0,
                missing_record_keys=len(expected_keys - observed_keys),
                unexpected_record_keys=len(observed_keys - expected_keys),
            )
        )
    run_frame = pd.DataFrame(runs).sort_values(
        ["setting", "seed", "replicate"], ignore_index=True
    )
    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["setting", "seed", "replicate", "subset_index"], ignore_index=True
    )
    return run_frame, candidate_frame, audit


def load_screening(directory: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    pattern = re.compile(r"Gamma_(?P<gamma>[0-9.]+)\.csv$")
    for path in sorted(directory.glob("Gamma_*.csv")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        file_gamma = float(match.group("gamma"))
        keys: list[int] = []
        records = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for records, row in enumerate(csv.reader(handle), start=1):
                if len(row) != 3:
                    raise ValueError(f"{path.name}, record {records}: expected 3 fields")
                gamma, seed = float(row[0]), int(row[1])
                if not np.isclose(gamma, file_gamma):
                    raise ValueError(f"{path.name}, record {records}: Gamma mismatch")
                average_any, average_fraction = ast.literal_eval(row[2])
                if len(average_any) != 5 or len(average_fraction) != 5:
                    raise ValueError(f"{path.name}, record {records}: expected length 5")
                keys.append(seed)
                for position, pairs in enumerate(PAIR_COUNTS):
                    any_value = float(average_any[position])
                    fraction_value = float(average_fraction[position])
                    if not (0 <= any_value <= 1 and 0 <= fraction_value <= 1):
                        raise ValueError(f"{path.name}, record {records}: invalid rate")
                    rows.append(
                        {
                            "gamma": gamma,
                            "seed": seed,
                            "pairs": pairs,
                            "replicates_in_seed": 10,
                            "average_any_ip_call": any_value,
                            "average_fraction_called": fraction_value,
                        }
                    )
        audit.append(
            _audit_row(
                "screening",
                path,
                records=records,
                valid_records=records,
                seeds=set(keys),
                expected_seeds=set(range(100)),
                duplicate_keys=len(keys) - len(set(keys)),
                malformed_records=0,
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["gamma", "pairs", "seed"], ignore_index=True
    ), audit


def load_bounds(directory: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    pattern = re.compile(r"rho(?P<rho>-?[0-9.]+)_I(?P<pairs>\d+)_.*\.csv$")
    for path in sorted(directory.glob("rho*.csv")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        rho = float(match.group("rho"))
        pairs = int(match.group("pairs"))
        keys: list[tuple[int, int]] = []
        records = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for records, row in enumerate(csv.reader(handle), start=1):
                if len(row) != 4:
                    raise ValueError(f"{path.name}, record {records}: expected 4 fields")
                seed, replicate = int(row[0]), int(row[1])
                exact = np.asarray(ast.literal_eval(row[2]), dtype=int)
                naive = np.asarray(ast.literal_eval(row[3]), dtype=int)
                if exact.shape != (4,) or naive.shape != (4,):
                    raise ValueError(f"{path.name}, record {records}: expected length 4")
                if np.any((exact < 0) | (exact > 4) | (naive < 0) | (naive > 4)):
                    raise ValueError(f"{path.name}, record {records}: invalid bound")
                if np.any(exact > naive):
                    raise ValueError(f"{path.name}, record {records}: exact exceeds naive")
                if np.any(np.diff(exact) < 0) or np.any(np.diff(naive) < 0):
                    raise ValueError(f"{path.name}, record {records}: nonmonotone Gamma")
                if exact[0] != naive[0]:
                    raise ValueError(f"{path.name}, record {records}: disagreement at Gamma=1")
                keys.append((seed, replicate))
                for position, gamma in enumerate(BOUND_GAMMAS):
                    rows.append(
                        {
                            "rho": rho,
                            "pairs": pairs,
                            "seed": seed,
                            "replicate": replicate,
                            "gamma": gamma,
                            "exact_bound": int(exact[position]),
                            "naive_bound": int(naive[position]),
                        }
                    )
        seeds = {key[0] for key in keys}
        expected_keys = set(itertools.product(range(100), range(10)))
        observed_keys = set(keys)
        audit.append(
            _audit_row(
                "exact versus naive",
                path,
                records=records,
                valid_records=records,
                seeds=seeds,
                expected_seeds=set(range(100)),
                duplicate_keys=len(keys) - len(set(keys)),
                malformed_records=0,
                missing_record_keys=len(expected_keys - observed_keys),
                unexpected_record_keys=len(observed_keys - expected_keys),
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["rho", "seed", "replicate", "gamma"], ignore_index=True
    ), audit


def _runtime_setting(setting: int) -> dict[str, object]:
    zero_based = setting - 1
    effect_pattern = "linear" if zero_based < 6 else "half-null"
    within_pattern = zero_based % 6
    correlation = 0.0 if within_pattern < 3 else 0.2
    gamma = RUNTIME_GAMMAS[within_pattern % 3]
    return {
        "effect_pattern": effect_pattern,
        "correlation": correlation,
        "gamma": gamma,
    }


def load_runtime(
    directory: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    malformed: list[dict[str, object]] = []
    pattern = re.compile(r"closed_testing_equi_K(?P<K>\d+)_I(?P<pairs>\d+)\.csv$")
    names = (
        "enumerative_rejection_fraction",
        "enumerative_seconds",
        "proposed_rejection_fraction",
        "proposed_seconds",
        "enumerative_conditional_seconds",
        "proposed_conditional_seconds",
        "ip_called",
        "fraction_called",
    )
    for path in sorted(directory.glob("closed_testing_equi*.csv")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        expected_k = int(match.group("K"))
        pairs = int(match.group("pairs"))
        keys: list[tuple[int, int]] = []
        records = 0
        valid_records = 0
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for records, row in enumerate(reader, start=1):
                seed_text = row[1] if len(row) > 1 else ""
                try:
                    if len(row) != 3:
                        raise ValueError("expected 3 fields")
                    k, seed = int(row[0]), int(row[1])
                    result = ast.literal_eval(row[2])
                    if k != expected_k:
                        raise ValueError("K does not match filename")
                    if len(result) != 8 or any(len(values) != 12 for values in result):
                        raise ValueError("expected eight lists of length 12")
                except Exception as error:
                    malformed.append(
                        {
                            "file": path.name,
                            "record": records,
                            "seed": seed_text,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                valid_records += 1
                keys.append((k, seed))
                for setting in range(1, 13):
                    values = {
                        name: result[position][setting - 1]
                        for position, name in enumerate(names)
                    }
                    rows.append(
                        {
                            "K": k,
                            "pairs": pairs,
                            "seed": seed,
                            "record": records,
                            "setting": setting,
                            **_runtime_setting(setting),
                            **values,
                        }
                    )
        seeds = {key[1] for key in keys}
        note = "legacy runtime outputs; algorithm decisions must be revalidated"
        audit.append(
            _audit_row(
                "runtime",
                path,
                records=records,
                valid_records=valid_records,
                seeds=seeds,
                expected_seeds=set(range(1000)),
                duplicate_keys=len(keys) - len(set(keys)),
                malformed_records=records - valid_records,
                note=note,
            )
        )
    frame = pd.DataFrame(rows).sort_values(
        ["K", "setting", "seed", "record"], ignore_index=True
    )
    numeric = [
        "enumerative_rejection_fraction",
        "enumerative_seconds",
        "proposed_rejection_fraction",
        "proposed_seconds",
        "ip_called",
        "fraction_called",
    ]
    if not frame.empty and (
        frame[numeric].isna().any().any()
        or (frame[["enumerative_seconds", "proposed_seconds"]] < 0).any().any()
    ):
        raise ValueError("runtime file contains invalid numeric values")
    return frame, audit, pd.DataFrame(malformed)


def load_all_results(project_root: Path) -> SimulationResults:
    data_root = Path(project_root) / "data"
    selection_runs, selection_candidates, selection_audit = load_selection(
        data_root / "subsets_compete"
    )
    screening, screening_audit = load_screening(data_root / "optcall_expr")
    bounds, bounds_audit = load_bounds(data_root / "compare_vR_naive_vs_exact")
    runtime, runtime_audit, malformed = load_runtime(
        data_root / "closed_testing_equi"
    )
    audit = pd.DataFrame(
        selection_audit + screening_audit + bounds_audit + runtime_audit
    )
    return SimulationResults(
        selection_runs=selection_runs,
        selection_candidates=selection_candidates,
        screening=screening,
        bounds=bounds,
        runtime=runtime,
        audit=audit,
        runtime_malformed=malformed,
    )


def selection_summary(
    runs: pd.DataFrame,
    candidates: pd.DataFrame,
    settings: tuple[int, ...] = (6, 0, 3),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_runs = runs[runs["setting"].isin(settings)].copy()
    selected_candidates = candidates[candidates["setting"].isin(settings)].copy()
    records: list[dict[str, object]] = []
    tie_records: list[dict[str, object]] = []
    key_columns = ["setting", "seed", "replicate"]
    candidate_groups = selected_candidates.groupby(key_columns, sort=False)
    for run in selected_runs.itertuples(index=False):
        key = (run.setting, run.seed, run.replicate)
        p_values = np.array([getattr(run, f"pvalue_{k}") for k in range(4)])
        naive_indices = tuple(np.argsort(p_values, kind="stable")[:2].tolist())
        naive_set = set(naive_indices)
        records.append(
            {
                "setting": run.setting,
                "effect_correlation": run.effect_correlation,
                "seed": run.seed,
                "replicate": run.replicate,
                "method": "Naive selector",
                "success": float(bool(naive_set & TRUE_OUTCOMES)),
                "exact_recovery": float(naive_set == TRUE_OUTCOMES),
                "number_true_selected": float(len(naive_set & TRUE_OUTCOMES)),
            }
        )
        group = candidate_groups.get_group(key)
        values = group["gsv_r1"].to_numpy()
        maximizers = np.flatnonzero(np.isclose(values, values.max()))
        subsets = [set(CANDIDATE_SUBSETS[index]) for index in maximizers]
        records.append(
            {
                "setting": run.setting,
                "effect_correlation": run.effect_correlation,
                "seed": run.seed,
                "replicate": run.replicate,
                "method": r"$\Gamma^*(\mathcal{R},1)$ selector",
                "success": float(np.mean([bool(subset & TRUE_OUTCOMES) for subset in subsets])),
                "exact_recovery": float(np.mean([subset == TRUE_OUTCOMES for subset in subsets])),
                "number_true_selected": float(
                    np.mean([len(subset & TRUE_OUTCOMES) for subset in subsets])
                ),
            }
        )
        tie_records.append(
            {
                "setting": run.setting,
                "effect_correlation": run.effect_correlation,
                "seed": run.seed,
                "replicate": run.replicate,
                "number_of_maximizers": len(maximizers),
                "tied": len(maximizers) > 1,
            }
        )
    record_frame = pd.DataFrame(records)
    summary = (
        record_frame.groupby(["setting", "effect_correlation", "method"], as_index=False)
        .agg(
            probability_success=("success", "mean"),
            probability_exact_recovery=("exact_recovery", "mean"),
            average_number_true=("number_true_selected", "mean"),
            monte_carlo_se=("success", lambda x: x.std(ddof=1) / np.sqrt(x.size)),
            replicates=("success", "size"),
        )
        .sort_values(["effect_correlation", "method"], ignore_index=True)
    )
    tie_frame = pd.DataFrame(tie_records)
    tie_summary = (
        tie_frame.groupby(["setting", "effect_correlation"], as_index=False)
        .agg(
            tied_replicates=("tied", "sum"),
            tie_probability=("tied", "mean"),
            maximum_number_of_maximizers=("number_of_maximizers", "max"),
            replicates=("tied", "size"),
        )
        .sort_values("effect_correlation", ignore_index=True)
    )
    return summary, tie_summary


def screening_summary(screening: pd.DataFrame) -> pd.DataFrame:
    summary = (
        screening.groupby(["gamma", "pairs"], as_index=False)
        .agg(
            probability_any_ip_call=("average_any_ip_call", "mean"),
            average_fraction_called=("average_fraction_called", "mean"),
            seed_level_se_any=(
                "average_any_ip_call",
                lambda x: x.std(ddof=1) / np.sqrt(x.size),
            ),
            seed_level_se_fraction=(
                "average_fraction_called",
                lambda x: x.std(ddof=1) / np.sqrt(x.size),
            ),
            seeds=("seed", "nunique"),
            datasets=("replicates_in_seed", "sum"),
        )
        .sort_values(["gamma", "pairs"], ignore_index=True)
    )
    return summary


def bound_cdf(bounds: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (rho, gamma), group in bounds.groupby(["rho", "gamma"]):
        for method, column in (("Exact method", "exact_bound"), ("Naive method", "naive_bound")):
            values = group[column].to_numpy() / 4
            for upper_bound in np.arange(5) / 4:
                records.append(
                    {
                        "rho": rho,
                        "gamma": gamma,
                        "method": method,
                        "fdp_upper_bound": upper_bound,
                        "cdf": float(np.mean(values <= upper_bound)),
                        "datasets": values.size,
                    }
                )
    return pd.DataFrame(records)


def bound_summary(bounds: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (rho, gamma), group in bounds.groupby(["rho", "gamma"]):
        exact = group["exact_bound"].to_numpy() / 4
        naive = group["naive_bound"].to_numpy() / 4
        records.append(
            {
                "rho": rho,
                "gamma": gamma,
                "datasets": group.shape[0],
                "exact_mean_bound": exact.mean(),
                "naive_mean_bound": naive.mean(),
                "exact_probability_at_most_0.75": np.mean(exact <= 0.75),
                "naive_probability_at_most_0.75": np.mean(naive <= 0.75),
                "exact_probability_vacuous": np.mean(exact == 1),
                "naive_probability_vacuous": np.mean(naive == 1),
                "probability_strict_improvement": np.mean(exact < naive),
            }
        )
    return pd.DataFrame(records).sort_values(["rho", "gamma"], ignore_index=True)


def runtime_summary(runtime: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = runtime.copy()
    frame["same_rejection_fraction"] = np.isclose(
        frame["enumerative_rejection_fraction"],
        frame["proposed_rejection_fraction"],
    )
    summary = (
        frame.groupby(
            ["K", "setting", "effect_pattern", "correlation", "gamma"],
            as_index=False,
        )
        .agg(
            records=("record", "size"),
            distinct_seeds=("seed", "nunique"),
            enumerative_seconds=("enumerative_seconds", "mean"),
            proposed_seconds=("proposed_seconds", "mean"),
            rejection_fraction_mismatches=(
                "same_rejection_fraction",
                lambda x: int((~x).sum()),
            ),
            mismatch_probability=("same_rejection_fraction", lambda x: 1 - x.mean()),
        )
        .sort_values(["K", "setting"], ignore_index=True)
    )
    summary["enumerative_minutes"] = summary["enumerative_seconds"] / 60
    summary["proposed_minutes"] = summary["proposed_seconds"] / 60
    summary["speedup"] = summary["enumerative_seconds"] / summary["proposed_seconds"]
    summary["relative_runtime_reduction"] = (
        1 - summary["proposed_seconds"] / summary["enumerative_seconds"]
    )
    quality = (
        frame.groupby("K", as_index=False)
        .agg(
            records=("record", "nunique"),
            distinct_seeds=("seed", "nunique"),
            setting_record_cells=("setting", "size"),
            rejection_fraction_mismatches=(
                "same_rejection_fraction",
                lambda x: int((~x).sum()),
            ),
        )
        .sort_values("K", ignore_index=True)
    )
    quality["mismatch_probability"] = (
        quality["rejection_fraction_mismatches"] / quality["setting_record_cells"]
    )
    return summary, quality


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,       # Sharper display inside Jupyter
            "savefig.dpi": 600,      # High-quality saved raster figures
            "pdf.fonttype": 42,
            "ps.fonttype": 42,

        }
    )



def _save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(directory / f"{stem}.png", bbox_inches="tight")


def plot_bound_distributions(
    cdf: pd.DataFrame,
    directory: Path,
) -> plt.Figure:
    _plot_style()

    gamma_styles = {
        1.25: {"color": "#08306b", "marker": "o", "linestyle": "-"},
        1.50: {"color": "#2171b5", "marker": "s", "linestyle": "--"},
        1.75: {"color": "#6baed6", "marker": "^", "linestyle": "-."},
    }

    method_styles = {
        "Exact method": {
            "filled": True,
            "linewidth": 1.9,
        },
        "Naive method": {
            "filled": False,
            "linewidth": 1.4,
        },
    }

    figure, axes = plt.subplots(
        1, 2,
        figsize=(7.15, 3.35),
        sharey=True,
    )

    for axis, rho in zip(axes, (0.0, 0.2)):
        panel = cdf.loc[
            cdf["rho"].eq(rho) & cdf["gamma"].gt(1)
        ]

        for gamma, gamma_style in gamma_styles.items():
            for method, method_style in method_styles.items():
                curve = (
                    panel.loc[
                        panel["gamma"].eq(gamma)
                        & panel["method"].eq(method)
                    ]
                    .sort_values("fdp_upper_bound")
                )

                axis.step(
                    curve["fdp_upper_bound"],
                    curve["cdf"],
                    where="post",
                    color=gamma_style["color"],
                    marker=gamma_style["marker"],
                    linestyle=gamma_style["linestyle"],
                    markerfacecolor=(
                        gamma_style["color"]
                        if method_style["filled"]
                        else "white"
                    ),
                    markeredgecolor=gamma_style["color"],
                    markeredgewidth=1.0,
                    linewidth=method_style["linewidth"],
                    markersize=4.8,
                )

        axis.set_title(
            r"Independent outcomes, $\Sigma^{(1)}$"
            if rho == 0
            else r"Equicorrelated outcomes, $\Sigma^{(2)}$"
        )
        
        axis.set_xlabel("FDP upper bound")
        axis.set_xticks(np.arange(5) / 4)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(
            axis="y",
            color="#D9D9D9",
            linewidth=0.6,
        )

    axes[0].set_ylabel("Empirical cumulative probability")

    gamma_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markerfacecolor=style["color"],
            linewidth=1.7,
            markersize=4.8,
            label=rf"$\Gamma={gamma:.2f}$",
        )
        for gamma, style in gamma_styles.items()
    ]

    method_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="o",
            linestyle="-",
            markerfacecolor=(
                "#333333"
                if style["filled"]
                else "white"
            ),
            markeredgecolor="#333333",
            linewidth=style["linewidth"],
            markersize=4.8,
            label=method,
        )
        for method, style in method_styles.items()
    ]

    figure.legend(
        handles=gamma_handles + method_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    figure.subplots_adjust(
        bottom=0.25,
        wspace=0.12,
    )

    _save_figure(
        figure,
        directory,
        "bound_distributions",
    )
    return figure


def plot_screening(summary: pd.DataFrame, directory: Path) -> plt.Figure:
    _plot_style()

    styles = {
        1.25: {"color": "#08306b", "marker": "o", "linestyle": "-"},
        1.50: {"color": "#2171b5", "marker": "s", "linestyle": "--"},
        1.75: {"color": "#6baed6", "marker": "^", "linestyle": "-."},
        2.00: {"color": "#9ecae1", "marker": "D", "linestyle": ":"},
    }

    figure, axes = plt.subplots(
        1, 2, figsize=(7.15, 3.35), sharex=True
    )

    panels = (
        ("probability_any_ip_call", "At least one IP call"),
        ("average_fraction_called", "Fraction of outcomes requiring IP"),
    )

    for axis, (column, title) in zip(axes, panels):
        for gamma, style in styles.items():
            curve = (
                summary.loc[summary["gamma"].eq(gamma)]
                .sort_values("pairs")
            )

            axis.plot(
                curve["pairs"],
                curve[column],
                linewidth=1.7,
                markersize=4.8,
                label=rf"$\Gamma={gamma:.2f}$",
                **style,
            )

        axis.set_title(title)
        axis.set_xlabel("Number of matched pairs B")
        axis.set_xscale("log")
        axis.set_xticks(PAIR_COUNTS)
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{int(value):,}")
        )
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axis.set_ylim(
            -0.02,
            1.02 if column == "probability_any_ip_call" else 0.22,
        )
        axis.grid(
            axis="y",
            color="#D9D9D9",
            linewidth=0.6,
        )

    axes[0].set_ylabel("Monte Carlo proportion")
    axes[1].legend(frameon=False, loc="upper right")

    figure.tight_layout()
    _save_figure(figure, directory, "screening_performance")
    return figure


def plot_runtime_diagnostic(summary: pd.DataFrame, directory: Path) -> plt.Figure:
    _plot_style()
    colors = {10: "#0072B2", 20: "#D55E00"}
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 5.0), sharey=True)
    for axis, k in zip(axes, (10, 20)):
        panel = summary[summary["K"] == k].sort_values("setting")
        axis.scatter(
            panel["relative_runtime_reduction"],
            panel["setting"],
            color=colors[k],
            s=34,
            zorder=3,
        )
        for row in panel.itertuples(index=False):
            axis.text(
                row.relative_runtime_reduction + 0.004,
                row.setting,
                f"{row.speedup:.1f}x",
                va="center",
                fontsize=8,
            )
        axis.set_title(f"K = {k}")
        axis.set_xlabel("Relative runtime reduction")
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_xlim(0.78, 1.04)
        axis.set_xticks((0.8, 0.85, 0.9, 0.95, 1.0))
        axis.set_yticks(range(1, 13))
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    axes[0].set_ylabel("Simulation setting")
    figure.suptitle("Legacy runtime results — diagnostic only", color="#A33A2B", y=1.01)
    figure.text(
        0.5,
        -0.01,
        "Incomplete files and disagreement between algorithms prevent publication use.",
        ha="center",
        color="#A33A2B",
        fontsize=9,
    )
    figure.tight_layout()
    _save_figure(figure, directory, "runtime_reduction_legacy_diagnostic")
    return figure


def write_tables(
    output_directory: Path,
    *,
    audit: pd.DataFrame,
    selection: pd.DataFrame,
    selection_ties: pd.DataFrame,
    screening: pd.DataFrame,
    bounds: pd.DataFrame,
    runtime: pd.DataFrame,
    runtime_quality: pd.DataFrame,
    runtime_malformed: pd.DataFrame,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_directory / "data_audit.csv", index=False)
    selection.to_csv(output_directory / "subset_selection_summary.csv", index=False)
    selection_ties.to_csv(output_directory / "subset_selection_ties.csv", index=False)
    screening.to_csv(output_directory / "screening_summary.csv", index=False)
    bounds.to_csv(output_directory / "bound_summary.csv", index=False)
    runtime.to_csv(output_directory / "runtime_summary_legacy.csv", index=False)
    runtime_quality.to_csv(output_directory / "runtime_quality_legacy.csv", index=False)
    runtime_malformed.to_csv(output_directory / "runtime_malformed_records.csv", index=False)

    selection_table = selection.pivot(
        index="effect_correlation", columns="method", values="probability_success"
    ).reset_index()
    sensitivity_column = r"$\Gamma^*(\mathcal{R},1)$ selector"
    selection_table = selection_table[
        ["effect_correlation", "Naive selector", sensitivity_column]
    ].sort_values("effect_correlation")
    selection_table.to_csv(
        output_directory / "subset_selection_manuscript_table.csv", index=False
    )
    latex_rows = [
        r"\begin{tabular}{rcc}",
        r"\toprule",
        r"$\eta_{12}$ & Naive selector & $\Gamma^*(\mathcal{R},1)$ selector \\",
        r"\midrule",
    ]
    latex_rows.extend(
        f"{rho:g} & {naive:.3f} & {sensitivity:.3f} \\\\"
        for rho, naive, sensitivity in selection_table.itertuples(
            index=False, name=None
        )
    )
    latex_rows.extend(
        (
            r"\midrule",
            r"\multicolumn{3}{l}{\footnotesize Sensitivity-selector values average uniformly over tied maximizers.} \\",
            r"\bottomrule",
            r"\end{tabular}",
        )
    )
    (output_directory / "subset_selection_table.tex").write_text(
        "\n".join(latex_rows) + "\n", encoding="utf-8"
    )


def run_analysis(project_root: Path) -> dict[str, object]:
    project_root = Path(project_root).resolve()
    results = load_all_results(project_root)
    selection, selection_ties = selection_summary(
        results.selection_runs, results.selection_candidates
    )
    screening = screening_summary(results.screening)
    cdf = bound_cdf(results.bounds)
    bounds = bound_summary(results.bounds)
    runtime, runtime_quality = runtime_summary(results.runtime)

    output_root = project_root / "outputs"
    figure_directory = output_root / "figures"
    table_directory = output_root / "tables"
    write_tables(
        table_directory,
        audit=results.audit,
        selection=selection,
        selection_ties=selection_ties,
        screening=screening,
        bounds=bounds,
        runtime=runtime,
        runtime_quality=runtime_quality,
        runtime_malformed=results.runtime_malformed,
    )
    figures = {
        "bounds": plot_bound_distributions(cdf, figure_directory),
        "screening": plot_screening(screening, figure_directory),
        "runtime_diagnostic": plot_runtime_diagnostic(runtime, figure_directory),
    }
    return {
        "raw": results,
        "selection": selection,
        "selection_ties": selection_ties,
        "screening": screening,
        "bound_cdf": cdf,
        "bounds": bounds,
        "runtime": runtime,
        "runtime_quality": runtime_quality,
        "figures": figures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


if __name__ == "__main__":
    analysis = run_analysis(parse_args().project_root)
    print("Data audit")
    print(analysis["raw"].audit.to_string(index=False))
    print("\nSubset-selection summary")
    print(analysis["selection"].to_string(index=False))
    print("\nRuntime quality")
    print(analysis["runtime_quality"].to_string(index=False))
