"""High-level FDP sensitivity analyses."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .data import PreparedStudy
from .multiple_testing import naive_fdp_bound
from .optimization import exact_fdp_bound, individual_worst_pvalues


@dataclass(frozen=True)
class SensitivityValueComparison:
    """Exact and naive generalized sensitivity values for one threshold."""

    exact: float
    naive: float
    exact_right_censored: bool
    naive_right_censored: bool
    evaluated_gammas: int
    exact_optimization_calls: int


def compare_generalized_sensitivity_values(
    study: PreparedStudy,
    subset: Iterable[int],
    threshold: int,
    *,
    alpha: float = 0.05,
    lower_gamma: float = 1.0,
    upper_gamma: float = 3.0,
    precision: float = 0.01,
    decision_tolerance: float = 1e-8,
    output_flag: int = 0,
) -> SensitivityValueComparison:
    """Compute exact and naive values while sharing worst-p-value work.

    The returned right-censoring flags indicate that the corresponding FDP
    bound had not crossed ``threshold`` by ``upper_gamma``.
    """
    chosen = tuple(sorted(set(int(k) for k in subset)))
    if not chosen:
        raise ValueError("subset must be nonempty")
    if not set(chosen).issubset(range(study.number_outcomes)):
        raise ValueError("subset contains an invalid outcome index")
    if threshold not in range(len(chosen)):
        raise ValueError("threshold must lie between 0 and len(subset)-1")
    if lower_gamma < 1 or upper_gamma < lower_gamma:
        raise ValueError("require 1 <= lower_gamma <= upper_gamma")
    if precision <= 0:
        raise ValueError("precision must be positive")

    pvalue_cache: dict[float, np.ndarray] = {}
    exact_cache: dict[float, int] = {}
    naive_cache: dict[float, int] = {}
    exact_calls = 0

    def key(gamma: float) -> float:
        return round(float(gamma), 12)

    def pvalues(gamma: float) -> np.ndarray:
        gamma_key = key(gamma)
        if gamma_key not in pvalue_cache:
            pvalue_cache[gamma_key] = individual_worst_pvalues(
                study,
                gamma_key,
                decision_tolerance=decision_tolerance,
                output_flag=output_flag,
            )
        return pvalue_cache[gamma_key]

    def exact_bound(gamma: float) -> int:
        nonlocal exact_calls
        gamma_key = key(gamma)
        if gamma_key not in exact_cache:
            result = exact_fdp_bound(
                study,
                chosen,
                gamma_key,
                alpha,
                worst_pvalues=pvalues(gamma_key),
                decision_tolerance=decision_tolerance,
                output_flag=output_flag,
            )
            exact_cache[gamma_key] = result.bound
            exact_calls += result.optimization_calls
        return exact_cache[gamma_key]

    def naive_bound(gamma: float) -> int:
        gamma_key = key(gamma)
        if gamma_key not in naive_cache:
            naive_cache[gamma_key] = naive_fdp_bound(
                pvalues(gamma_key), chosen, alpha=alpha
            )
        return naive_cache[gamma_key]

    def locate(bound_function) -> tuple[float, bool]:
        if bound_function(lower_gamma) > threshold:
            return float(lower_gamma), False
        if bound_function(upper_gamma) <= threshold:
            return float(upper_gamma), True
        left, right = float(lower_gamma), float(upper_gamma)
        while right - left > precision:
            midpoint = (left + right) / 2
            if bound_function(midpoint) > threshold:
                right = midpoint
            else:
                left = midpoint
        return right, False

    exact_value, exact_censored = locate(exact_bound)
    naive_value, naive_censored = locate(naive_bound)
    return SensitivityValueComparison(
        exact=exact_value,
        naive=naive_value,
        exact_right_censored=exact_censored,
        naive_right_censored=naive_censored,
        evaluated_gammas=len(pvalue_cache),
        exact_optimization_calls=exact_calls,
    )


def generalized_sensitivity_values(
    study: PreparedStudy,
    subset: Iterable[int],
    thresholds: Iterable[int] | None = None,
    *,
    alpha: float = 0.05,
    lower_gamma: float = 1.0,
    upper_gamma: float = 3.0,
    precision: float = 0.01,
    decision_tolerance: float = 1e-8,
    output_flag: int = 0,
) -> np.ndarray:
    """Compute generalized sensitivity values by monotone bisection in Gamma.

    Values equal to ``upper_gamma`` should be interpreted as right-censored if
    the FDP bound has not crossed the requested threshold there.
    """
    chosen = tuple(sorted(set(int(k) for k in subset)))
    if not chosen:
        raise ValueError("subset must be nonempty")
    if thresholds is None:
        requested = np.arange(len(chosen), dtype=int)
    else:
        requested = np.asarray(list(thresholds), dtype=int)
    if np.any((requested < 0) | (requested >= len(chosen))):
        raise ValueError("thresholds must lie between 0 and len(subset)-1")
    if lower_gamma < 1 or upper_gamma < lower_gamma:
        raise ValueError("require 1 <= lower_gamma <= upper_gamma")
    if precision <= 0:
        raise ValueError("precision must be positive")

    cache: dict[float, int] = {}

    def bound(gamma: float) -> int:
        key = round(float(gamma), 12)
        if key not in cache:
            p_values = individual_worst_pvalues(
                study,
                key,
                decision_tolerance=decision_tolerance,
                output_flag=output_flag,
            )
            cache[key] = exact_fdp_bound(
                study,
                chosen,
                key,
                alpha,
                worst_pvalues=p_values,
                decision_tolerance=decision_tolerance,
                output_flag=output_flag,
            ).bound
        return cache[key]

    lower_bound = bound(lower_gamma)
    upper_bound = bound(upper_gamma)
    answers = np.empty(requested.size, dtype=float)
    for position, threshold in enumerate(requested):
        if lower_bound > threshold:
            answers[position] = lower_gamma
            continue
        if upper_bound <= threshold:
            answers[position] = upper_gamma
            continue
        left, right = lower_gamma, upper_gamma
        while right - left > precision:
            midpoint = (left + right) / 2
            if bound(midpoint) > threshold:
                right = midpoint
            else:
                left = midpoint
        answers[position] = right
    return answers
