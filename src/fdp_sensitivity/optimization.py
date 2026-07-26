"""Gurobi implementations of the sensitivity-analysis optimizations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, log2
from typing import Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, norm

from .data import PreparedStudy
from .multiple_testing import holm_bonferroni

try:  # Keep non-optimization utilities importable without Gurobi.
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:  # pragma: no cover - depends on local installation
    gp = None
    GRB = None


# The statistical boundary is zero.  We use one small *positive* numerical
# slack everywhere, so a solver-near-boundary intersection is treated as
# nonrejected.  This is conservative and, unlike the old negative margins,
# cannot create extra rejections merely because of numerical uncertainty.
DEFAULT_DECISION_TOLERANCE = 1e-7
SOLVER_FEASIBILITY_TOLERANCE = 1e-9
SOLVER_OPTIMALITY_TOLERANCE = 1e-9
SOLVER_BAR_QCP_TOLERANCE = 1e-10


class NumericalDecisionError(RuntimeError):
    """Raised when an optimization does not certify the requested decision."""


def _require_gurobi() -> None:
    if gp is None:
        raise ModuleNotFoundError(
            "gurobipy is required for this operation; install it in a licensed environment"
        )


def _new_model(output_flag: int = 0):
    _require_gurobi()
    model = gp.Model()
    model.Params.OutputFlag = output_flag
    model.Params.FeasibilityTol = SOLVER_FEASIBILITY_TOLERANCE
    model.Params.IntFeasTol = SOLVER_FEASIBILITY_TOLERANCE
    model.Params.OptimalityTol = SOLVER_OPTIMALITY_TOLERANCE
    model.Params.BarQCPConvTol = SOLVER_BAR_QCP_TOLERANCE
    model.Params.NumericFocus = 2
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
    if model.Status in (GRB.NUMERIC, GRB.SUBOPTIMAL):
        model.Params.NumericFocus = 3
        model.Params.ScaleFlag = 2
        model.Params.BarHomogeneous = 1
        model.optimize()


def _retry_inf_or_unbd(model) -> None:
    """Disambiguate ``INF_OR_UNBD`` before it is used as a decision."""
    if model.Status == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.reset()
        _optimize_with_numeric_retry(model)


def _zeta_value(
    study: PreparedStudy,
    outcome: int,
    critical_probability: float,
    probabilities: list[np.ndarray],
) -> float:
    """Recompute zeta outside Gurobi for incumbent validation."""
    quantile = float(chi2.ppf(1 - critical_probability, df=1))
    mean = 0.0
    variance = 0.0
    for b, size in enumerate(study.set_sizes):
        p = probabilities[b]
        q = study.scores[b, :size, outcome]
        set_mean = float(p @ q)
        mean += set_mean
        variance += float(p @ (q**2) - set_mean**2)
    residual = float(study.observed_statistics[outcome]) - mean
    return residual**2 - quantile * max(variance, 0.0)


def paired_minimum_zeta(
    study: PreparedStudy,
    outcome: int,
    gamma: float,
    critical_probability: float,
) -> float:
    """Exact minimum-zeta oracle for matched pairs.

    The pairwise problem is a box-constrained convex quadratic.  Its KKT
    equations reduce to one monotone scalar root, so this calculation is both
    exact (up to floating point) and much faster than a Gurobi solve.
    """
    if any(size != 2 for size in study.set_sizes):
        raise ValueError("paired_minimum_zeta requires matched pairs")
    if outcome not in range(study.number_outcomes):
        raise ValueError("invalid outcome index")
    if gamma < 1:
        raise ValueError("gamma must be at least one")
    if not 0 < critical_probability < 1:
        raise ValueError("critical_probability must lie in (0, 1)")

    q = study.scores[:, :2, outcome]
    low_score = q.min(axis=1)
    differences = q.max(axis=1) - low_score
    positive = differences > 0
    differences = differences[positive]
    centered_observed = float(study.observed_statistics[outcome] - low_score.sum())
    if differences.size == 0:
        return centered_observed**2

    lower_probability = 1 / (1 + gamma)
    upper_probability = gamma / (1 + gamma)
    quantile = float(chi2.ppf(1 - critical_probability, df=1))
    residual_lower = centered_observed - upper_probability * differences.sum()
    residual_upper = centered_observed - lower_probability * differences.sum()

    def equation(residual: float) -> float:
        probabilities = np.clip(
            0.5 + residual / (quantile * differences),
            lower_probability,
            upper_probability,
        )
        return float(
            residual - centered_observed + probabilities @ differences
        )

    lower_value = equation(residual_lower)
    upper_value = equation(residual_upper)
    if lower_value >= 0:
        residual = residual_lower
    elif upper_value <= 0:
        residual = residual_upper
    else:
        residual = float(
            brentq(
                equation,
                residual_lower,
                residual_upper,
                xtol=1e-14,
                rtol=1e-14,
            )
        )
    probabilities = np.clip(
        0.5 + residual / (quantile * differences),
        lower_probability,
        upper_probability,
    )
    variance = np.sum(
        probabilities * (1 - probabilities) * differences**2
    )
    return float(residual**2 - quantile * variance)


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
    if all(size == 2 for size in study.set_sizes):
        return paired_minimum_zeta(
            study, outcome, gamma, critical_probability
        )
    model = _new_model(output_flag)
    rho = _assignment_variables(model, study.set_sizes, gamma)
    expression = _zeta_expression(
        model, rho, study, outcome, critical_probability, prefix=f"k{outcome}"
    )
    model.setObjective(expression, GRB.MINIMIZE)
    _optimize_with_numeric_retry(model)
    if model.Status != GRB.OPTIMAL:
        raise NumericalDecisionError(
            f"minimum-zeta optimization did not certify optimality; "
            f"status={model.Status}, solutions={model.SolCount}"
        )
    return float(model.ObjVal)


def exact_two_sided_worst_pvalue(
    study: PreparedStudy,
    outcome: int,
    gamma: float,
    *,
    tolerance: float = 1e-6,
    decision_tolerance: float = DEFAULT_DECISION_TOLERANCE,
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
        # Ambiguity is resolved upward: retaining a possibly eligible outcome
        # is conservative for every downstream multiple-testing screen.
        if value <= decision_tolerance:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def paired_two_sided_worst_pvalue(
    study: PreparedStudy,
    outcome: int,
    gamma: float,
    *,
    tolerance: float = 1e-6,
    decision_tolerance: float = DEFAULT_DECISION_TOLERANCE,
) -> float:
    """Exact worst normally approximated two-sided p-value for matched pairs."""
    if any(size != 2 for size in study.set_sizes):
        raise ValueError("paired_two_sided_worst_pvalue requires matched pairs")
    if gamma < 1:
        raise ValueError("gamma must be at least one")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must lie in (0, 1)")
    lower, upper = 0.0, 1.0
    iterations = ceil(log2(1 / tolerance))
    for _ in range(iterations):
        midpoint = (lower + upper) / 2
        value = paired_minimum_zeta(
            study, outcome, gamma, midpoint
        )
        if value <= decision_tolerance:
            lower = midpoint
        else:
            upper = midpoint
    return upper


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
    decision_tolerance: float = DEFAULT_DECISION_TOLERANCE,
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
            values[k] = paired_two_sided_worst_pvalue(
                study,
                k,
                gamma,
                tolerance=tolerance,
                decision_tolerance=decision_tolerance,
            )
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
    decision_tolerance: float = DEFAULT_DECISION_TOLERANCE,
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
    if model.SolCount and model.ObjVal <= decision_tolerance:
        # A feasible point at or below the positive numerical slack is a
        # conservative certificate of nonrejection.
        return False
    if model.Status == GRB.OPTIMAL and model.ObjBound > decision_tolerance:
        return True
    raise NumericalDecisionError(
        "local-test sign was not certified: "
        f"status={model.Status}, solutions={model.SolCount}, "
        f"objective={getattr(model, 'ObjVal', np.nan)}, "
        f"bound={getattr(model, 'ObjBound', np.nan)}"
    )


@dataclass(frozen=True)
class FDPBoundResult:
    bound: int
    optimization_calls: int
    selected_superset: tuple[int, ...] | None


def _validated_miqcp_witness(
    model,
    rho,
    theta,
    study: PreparedStudy,
    selected_candidates: set[int],
    required_size: int,
    chosen_candidates: set[int],
    required_chosen: int,
    gamma: float,
    critical_probability: float,
    decision_tolerance: float,
) -> tuple[int, ...] | None:
    """Return a selected subset only after validating the incumbent in NumPy."""
    if model.SolCount <= 0:
        return None
    selected = tuple(
        k for k in sorted(selected_candidates) if float(theta[k].X) > 0.5
    )
    if len(selected) != required_size:
        return None
    if len(set(selected) & chosen_candidates) != required_chosen:
        return None

    probabilities: list[np.ndarray] = []
    validation_tolerance = 10 * SOLVER_FEASIBILITY_TOLERANCE
    for b, size in enumerate(study.set_sizes):
        p = np.asarray([float(rho[b, j].X) for j in range(size)])
        if np.any(~np.isfinite(p)) or np.any(p < -validation_tolerance):
            return None
        if abs(float(p.sum()) - 1.0) > validation_tolerance:
            return None
        smallest = float(p.min())
        largest = float(p.max())
        if smallest <= 0 or largest > gamma * smallest + validation_tolerance:
            return None
        probabilities.append(p)

    max_zeta = max(
        _zeta_value(
            study, k, critical_probability, probabilities
        )
        for k in selected
    )
    if max_zeta > decision_tolerance + validation_tolerance:
        return None
    return selected


def exact_fdp_bound(
    study: PreparedStudy,
    subset: Iterable[int],
    gamma: float,
    alpha: float = 0.05,
    *,
    worst_pvalues: np.ndarray | None = None,
    decision_tolerance: float = DEFAULT_DECISION_TOLERANCE,
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
    if np.any(~np.isfinite(worst_pvalues)) or np.any(
        (worst_pvalues < 0) | (worst_pvalues > 1)
    ):
        raise ValueError("worst_pvalues must be finite and lie in [0, 1]")

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
                ub=decision_tolerance,
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
            _retry_inf_or_unbd(model)

            selected = _validated_miqcp_witness(
                model,
                rho,
                theta,
                study,
                eligible,
                size,
                eligible_chosen,
                r,
                gamma,
                threshold,
                decision_tolerance,
            )
            if selected is not None:
                return FDPBoundResult(r, calls, selected)
            if model.Status != GRB.INFEASIBLE:
                raise NumericalDecisionError(
                    "MIQCP did not provide a validated witness or an "
                    f"infeasibility certificate; status={model.Status}, "
                    f"solutions={model.SolCount}, max_violation="
                    f"{getattr(model, 'MaxVio', np.nan)}"
                )
    return FDPBoundResult(0, calls, None)


def enumerative_elementary_rejections(
    study: PreparedStudy,
    gamma: float,
    alpha: float = 0.05,
    *,
    targets: Iterable[int] | None = None,
    worst_pvalues: np.ndarray | None = None,
    decision_tolerance: float = DEFAULT_DECISION_TOLERANCE,
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
    decision_tolerance: float = DEFAULT_DECISION_TOLERANCE,
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
