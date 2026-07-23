"""Gurobi implementations of the sensitivity-analysis optimizations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, log2
from typing import Iterable

import numpy as np
from scipy.stats import chi2, norm

from .data import PreparedStudy
from .multiple_testing import holm_bonferroni

try:  # Keep non-optimization utilities importable without Gurobi.
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:  # pragma: no cover - depends on local installation
    gp = None
    GRB = None


def _require_gurobi() -> None:
    if gp is None:
        raise ModuleNotFoundError(
            "gurobipy is required for this operation; install it in a licensed environment"
        )


def _new_model(output_flag: int = 0):
    _require_gurobi()
    model = gp.Model()
    model.Params.OutputFlag = output_flag
    return model


def _assignment_variables(model, set_sizes: tuple[int, ...], gamma: float):
    if gamma < 1:
        raise ValueError("gamma must be at least one")
    keys = [(b, j) for b, size in enumerate(set_sizes) for j in range(size)]
    rho = model.addVars(keys, lb=0.0, ub=1.0, name="rho")
    lower = model.addVars(len(set_sizes), lb=0.0, ub=1.0, name="rho_min")
    for b, size in enumerate(set_sizes):
        model.addConstr(gp.quicksum(rho[b, j] for j in range(size)) == 1)
        for j in range(size):
            model.addConstr(rho[b, j] >= lower[b])
            model.addConstr(rho[b, j] <= gamma * lower[b])
    return rho


def _zeta_expression(
    model,
    rho,
    study: PreparedStudy,
    outcome: int,
    critical_probability: float,
    prefix: str,
):
    if not 0 < critical_probability < 1:
        raise ValueError("critical_probability must lie in (0, 1)")
    quantile = float(chi2.ppf(1 - critical_probability, df=1))
    set_means = model.addVars(
        study.number_sets, lb=-GRB.INFINITY, name=f"{prefix}_set_mean"
    )
    residual = model.addVar(lb=-GRB.INFINITY, name=f"{prefix}_residual")
    second_moment_terms = []
    for b, size in enumerate(study.set_sizes):
        q = study.scores[b, :size, outcome]
        model.addConstr(
            set_means[b] == gp.quicksum(rho[b, j] * float(q[j]) for j in range(size))
        )
        second_moment_terms.append(
            gp.quicksum(rho[b, j] * float(q[j] ** 2) for j in range(size))
        )
    model.addConstr(
        residual
        == float(study.observed_statistics[outcome])
        - gp.quicksum(set_means[b] for b in range(study.number_sets))
    )
    return residual * residual + quantile * gp.quicksum(
        set_means[b] * set_means[b] - second_moment_terms[b]
        for b in range(study.number_sets)
    )


def _optimize_with_numeric_retry(model) -> None:
    model.optimize()
    if model.Status == GRB.NUMERIC:
        model.Params.NumericFocus = 3
        model.Params.ScaleFlag = 2
        model.Params.BarHomogeneous = 1
        model.optimize()


def minimum_zeta(
    study: PreparedStudy,
    outcome: int,
    gamma: float,
    critical_probability: float,
    *,
    output_flag: int = 0,
) -> float:
    """Compute the minimum of ``zeta_k(rho; critical_probability)``."""
    if outcome not in range(study.number_outcomes):
        raise ValueError("invalid outcome index")
    model = _new_model(output_flag)
    rho = _assignment_variables(model, study.set_sizes, gamma)
    expression = _zeta_expression(
        model, rho, study, outcome, critical_probability, prefix=f"k{outcome}"
    )
    zeta = model.addVar(lb=-GRB.INFINITY, name="zeta")
    model.addConstr(zeta >= expression)
    model.setObjective(zeta, GRB.MINIMIZE)
    _optimize_with_numeric_retry(model)
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"minimum-zeta optimization ended with status {model.Status}")
    return float(model.ObjVal)


def exact_two_sided_worst_pvalue(
    study: PreparedStudy,
    outcome: int,
    gamma: float,
    *,
    tolerance: float = 1e-6,
    decision_tolerance: float = 1e-9,
    output_flag: int = 0,
) -> float:
    """Compute the worst normally approximated two-sided p-value by bisection.

    For a candidate value ``c``, the worst p-value exceeds ``c`` exactly when
    the minimum of ``zeta_k(rho; c)`` is negative.  The feasible assignment set
    uses the full pairwise Rosenbaum ratio restriction, including for sets with
    more than two units.
    """
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must lie in (0, 1)")
    lower, upper = 0.0, 1.0
    iterations = ceil(log2(1 / tolerance))
    for _ in range(iterations):
        midpoint = (lower + upper) / 2
        value = minimum_zeta(
            study,
            outcome,
            gamma,
            midpoint,
            output_flag=output_flag,
        )
        if value < -decision_tolerance:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def paired_two_sided_worst_pvalue(
    study: PreparedStudy, outcome: int, gamma: float
) -> float:
    """Fast analytic worst p-value for matched pairs."""
    if any(size != 2 for size in study.set_sizes):
        raise ValueError("paired_two_sided_worst_pvalue requires matched pairs")
    if gamma < 1:
        raise ValueError("gamma must be at least one")
    q = study.scores[:, :2, outcome]
    low_score = q.min(axis=1)
    high_score = q.max(axis=1)
    low_probability = 1 / (1 + gamma)
    high_probability = gamma / (1 + gamma)
    lower_mean = np.sum(high_probability * low_score + low_probability * high_score)
    upper_mean = np.sum(low_probability * low_score + high_probability * high_score)
    observed = float(study.observed_statistics[outcome])
    if lower_mean <= observed <= upper_mean:
        return 1.0
    variance = np.sum(
        low_probability * high_probability * (high_score - low_score) ** 2
    )
    if variance <= 0:
        return float(observed == lower_mean)
    target_mean = upper_mean if observed > upper_mean else lower_mean
    return float(2 * norm.sf(abs(observed - target_mean) / np.sqrt(variance)))


def no_hidden_bias_pvalue(study: PreparedStudy, outcome: int) -> float:
    """Normally approximated two-sided p-value when ``Gamma=1``.

    At ``Gamma=1`` the assignment probability is uniform within every matched
    set, so this calculation is analytic for pairs and larger full-matching
    sets and does not require Gurobi.
    """
    if outcome not in range(study.number_outcomes):
        raise ValueError("invalid outcome index")
    mean = 0.0
    variance = 0.0
    for b, size in enumerate(study.set_sizes):
        values = study.scores[b, :size, outcome]
        set_mean = float(np.mean(values))
        mean += set_mean
        variance += float(np.mean(values**2) - set_mean**2)
    observed = float(study.observed_statistics[outcome])
    if variance <= 0:
        return float(observed == mean)
    return float(2 * norm.sf(abs(observed - mean) / np.sqrt(variance)))


def individual_worst_pvalues(
    study: PreparedStudy,
    gamma: float,
    *,
    tolerance: float = 1e-6,
    decision_tolerance: float = 1e-9,
    output_flag: int = 0,
) -> np.ndarray:
    """Compute all individual worst-case two-sided p-values."""
    if abs(gamma - 1.0) <= 1e-12:
        return np.asarray(
            [no_hidden_bias_pvalue(study, k) for k in range(study.number_outcomes)]
        )
    paired = all(size == 2 for size in study.set_sizes)
    values = np.empty(study.number_outcomes)
    for k in range(study.number_outcomes):
        if paired:
            values[k] = paired_two_sided_worst_pvalue(study, k, gamma)
        else:
            values[k] = exact_two_sided_worst_pvalue(
                study,
                k,
                gamma,
                tolerance=tolerance,
                decision_tolerance=decision_tolerance,
                output_flag=output_flag,
            )
    return values


def local_test_rejects(
    study: PreparedStudy,
    outcomes: Iterable[int],
    gamma: float,
    alpha: float,
    *,
    decision_tolerance: float = 1e-9,
    output_flag: int = 0,
) -> bool:
    """Return the Bonferroni local sensitivity-test decision for one subset."""
    selected = tuple(sorted(set(int(k) for k in outcomes)))
    if not selected:
        raise ValueError("outcomes must be nonempty")
    if not set(selected).issubset(range(study.number_outcomes)):
        raise ValueError("invalid outcome index")
    model = _new_model(output_flag)
    rho = _assignment_variables(model, study.set_sizes, gamma)
    y = model.addVar(lb=-GRB.INFINITY, name="maximum_zeta")
    for k in selected:
        expression = _zeta_expression(
            model, rho, study, k, alpha / len(selected), prefix=f"k{k}"
        )
        model.addConstr(y >= expression)
    model.setObjective(y, GRB.MINIMIZE)
    _optimize_with_numeric_retry(model)
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"local-test optimization ended with status {model.Status}")
    # Treat values within ``decision_tolerance`` of zero as rejections.  This
    # matches the MIQCP convention that a nonrejection requires a value below
    # ``-decision_tolerance``.
    return bool(model.ObjVal > -decision_tolerance)


@dataclass(frozen=True)
class FDPBoundResult:
    bound: int
    optimization_calls: int
    selected_superset: tuple[int, ...] | None


def exact_fdp_bound(
    study: PreparedStudy,
    subset: Iterable[int],
    gamma: float,
    alpha: float = 0.05,
    *,
    worst_pvalues: np.ndarray | None = None,
    decision_tolerance: float = 1e-8,
    output_flag: int = 0,
) -> FDPBoundResult:
    """Compute the exact common-configuration FDP numerator bound by MIQCP."""
    chosen = set(int(k) for k in subset)
    if not chosen:
        return FDPBoundResult(0, 0, None)
    if not chosen.issubset(range(study.number_outcomes)):
        raise ValueError("subset contains an invalid outcome index")
    if worst_pvalues is None:
        worst_pvalues = individual_worst_pvalues(
            study, gamma, decision_tolerance=decision_tolerance, output_flag=output_flag
        )
    worst_pvalues = np.asarray(worst_pvalues, dtype=float)
    if worst_pvalues.shape != (study.number_outcomes,):
        raise ValueError("worst_pvalues has the wrong shape")

    calls = 0
    all_outcomes = set(range(study.number_outcomes))
    for r in range(len(chosen), 0, -1):
        for size in range(r, r + study.number_outcomes - len(chosen) + 1):
            threshold = alpha / size
            eligible = {k for k in all_outcomes if worst_pvalues[k] > threshold}
            eligible_chosen = eligible & chosen
            if len(eligible) < size or len(eligible_chosen) < r:
                continue

            calls += 1
            model = _new_model(output_flag)
            model.Params.SolutionLimit = 1
            rho = _assignment_variables(model, study.set_sizes, gamma)
            theta = model.addVars(sorted(eligible), vtype=GRB.BINARY, name="theta")
            y = model.addVar(
                lb=-GRB.INFINITY,
                ub=-decision_tolerance,
                name="maximum_zeta",
            )
            for k in sorted(eligible):
                expression = _zeta_expression(
                    model, rho, study, k, threshold, prefix=f"k{k}"
                )
                zeta = model.addVar(lb=-GRB.INFINITY, name=f"zeta_{k}")
                model.addConstr(zeta >= expression)
                model.addGenConstrIndicator(theta[k], True, y >= zeta)
            model.addConstr(gp.quicksum(theta[k] for k in eligible) == size)
            model.addConstr(gp.quicksum(theta[k] for k in eligible_chosen) == r)
            model.setObjective(0.0)
            _optimize_with_numeric_retry(model)

            feasible_status = model.Status in (
                GRB.OPTIMAL,
                GRB.SOLUTION_LIMIT,
                GRB.SUBOPTIMAL,
            )
            if model.Status == GRB.TIME_LIMIT and model.SolCount:
                feasible_status = True
            if feasible_status:
                selected = tuple(k for k in sorted(eligible) if theta[k].X > 0.5)
                return FDPBoundResult(r, calls, selected)
            if model.Status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                raise RuntimeError(f"MIQCP ended with unresolved status {model.Status}")
    return FDPBoundResult(0, calls, None)


def enumerative_elementary_rejections(
    study: PreparedStudy,
    gamma: float,
    alpha: float = 0.05,
    *,
    targets: Iterable[int] | None = None,
    worst_pvalues: np.ndarray | None = None,
    decision_tolerance: float = 1e-9,
    output_flag: int = 0,
) -> set[int]:
    """Reference closed-testing implementation for elementary decisions."""
    if worst_pvalues is None:
        worst_pvalues = individual_worst_pvalues(
            study, gamma, decision_tolerance=decision_tolerance, output_flag=output_flag
        )
    worst_pvalues = np.asarray(worst_pvalues)
    number_outcomes = study.number_outcomes
    if targets is None:
        requested = set(range(number_outcomes))
    else:
        requested = set(int(k) for k in targets)
        if not requested.issubset(range(number_outcomes)):
            raise ValueError("targets contains an invalid outcome index")
    rejected: set[int] = set()
    cache: dict[frozenset[int], bool] = {}
    for k in sorted(requested):
        if worst_pvalues[k] > alpha:
            continue
        if worst_pvalues[k] <= alpha / number_outcomes:
            rejected.add(k)
            continue
        blocked = False
        others = [j for j in range(number_outcomes) if j != k]
        for size in range(1, number_outcomes + 1):
            for remainder in combinations(others, size - 1):
                candidate = frozenset((k, *remainder))
                if any(worst_pvalues[j] <= alpha / size for j in candidate):
                    continue
                if candidate not in cache:
                    cache[candidate] = local_test_rejects(
                        study,
                        candidate,
                        gamma,
                        alpha,
                        decision_tolerance=decision_tolerance,
                        output_flag=output_flag,
                    )
                if not cache[candidate]:
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            rejected.add(k)
    return rejected


def miqcp_elementary_rejections(
    study: PreparedStudy,
    gamma: float,
    alpha: float = 0.05,
    *,
    targets: Iterable[int] | None = None,
    worst_pvalues: np.ndarray | None = None,
    decision_tolerance: float = 1e-8,
    output_flag: int = 0,
) -> set[int]:
    """Elementary decisions obtained from the MIQCP FDP bound."""
    if worst_pvalues is None:
        worst_pvalues = individual_worst_pvalues(
            study, gamma, decision_tolerance=decision_tolerance, output_flag=output_flag
        )
    if targets is None:
        requested = set(range(study.number_outcomes))
    else:
        requested = set(int(k) for k in targets)
        if not requested.issubset(range(study.number_outcomes)):
            raise ValueError("targets contains an invalid outcome index")
    naive_rejections = holm_bonferroni(worst_pvalues, alpha=alpha)
    decisions: set[int] = set()
    for k in sorted(requested):
        if k in naive_rejections:
            decisions.add(k)
        elif worst_pvalues[k] <= alpha:
            result = exact_fdp_bound(
                study,
                {k},
                gamma,
                alpha,
                worst_pvalues=worst_pvalues,
                decision_tolerance=decision_tolerance,
                output_flag=output_flag,
            )
            if result.bound == 0:
                decisions.add(k)
    return decisions
