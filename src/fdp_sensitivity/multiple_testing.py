"""Multiple-testing utilities that do not depend on Gurobi."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def holm_bonferroni(p_values: Iterable[float], alpha: float = 0.05) -> set[int]:
    """Return the indices rejected by Holm's step-down procedure."""
    p = np.asarray(list(p_values), dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a nonempty one-dimensional sequence")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")

    order = np.argsort(p, kind="stable")
    ordered = p[order]
    critical = alpha / (p.size - np.arange(p.size))
    failures = np.flatnonzero(ordered > critical)
    number_rejected = failures[0] if failures.size else p.size
    return set(order[:number_rejected].tolist())


def naive_fdp_bound(
    p_values: Iterable[float], subset: Iterable[int], alpha: float = 0.05
) -> int:
    """FDP numerator bound obtained from individual worst-case p-values."""
    p = np.asarray(list(p_values), dtype=float)
    chosen = set(int(k) for k in subset)
    if not chosen.issubset(range(p.size)):
        raise ValueError("subset contains an outcome index outside p_values")
    rejected = holm_bonferroni(p, alpha=alpha)
    return len(chosen - rejected)

