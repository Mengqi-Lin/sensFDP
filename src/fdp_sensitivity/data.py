"""Simulation and preprocessing for matched observational studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PreparedStudy:
    """Arrays used repeatedly by the sensitivity-analysis optimizations."""

    number_sets: int
    number_outcomes: int
    set_sizes: tuple[int, ...]
    scores: np.ndarray
    observed_statistics: np.ndarray
    set_labels: np.ndarray


def select_outcomes(
    study: PreparedStudy, outcomes: Sequence[int]
) -> PreparedStudy:
    """Return a study restricted to an ordered collection of outcomes."""
    chosen = tuple(int(k) for k in outcomes)
    if not chosen:
        raise ValueError("outcomes must be nonempty")
    if len(set(chosen)) != len(chosen):
        raise ValueError("outcomes must not contain duplicates")
    if not set(chosen).issubset(range(study.number_outcomes)):
        raise ValueError("outcomes contains an invalid index")
    return PreparedStudy(
        number_sets=study.number_sets,
        number_outcomes=len(chosen),
        set_sizes=study.set_sizes,
        scores=study.scores[:, :, chosen].copy(),
        observed_statistics=study.observed_statistics[list(chosen)].copy(),
        set_labels=study.set_labels.copy(),
    )


def generate_pair_data(
    number_pairs: int,
    effects: Sequence[float] | float,
    covariance: np.ndarray | None,
    gamma_bias: float = 1.0,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the paired Gaussian design used in the manuscript."""
    if number_pairs <= 0:
        raise ValueError("number_pairs must be positive")
    if gamma_bias < 1:
        raise ValueError("gamma_bias must be at least one")
    rng = np.random.default_rng() if rng is None else rng
    tau = np.atleast_1d(np.asarray(effects, dtype=float))
    number_outcomes = tau.size
    if covariance is None:
        covariance = np.eye(number_outcomes)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (number_outcomes, number_outcomes):
        raise ValueError("covariance has the wrong shape")

    controls = rng.multivariate_normal(
        np.zeros(number_outcomes), covariance, size=(number_pairs, 2)
    )
    null_outcomes = np.flatnonzero(tau == 0)
    if null_outcomes.size:
        contrast = controls[:, 0, null_outcomes].sum(axis=1) - controls[
            :, 1, null_outcomes
        ].sum(axis=1)
    else:
        contrast = np.zeros(number_pairs)
    odds = np.where(contrast > 0, gamma_bias, np.where(contrast < 0, 1 / gamma_bias, 1))
    probability_first = odds / (1 + odds)
    first_treated = rng.binomial(1, probability_first)
    assignment = np.column_stack([first_treated, 1 - first_treated])
    observed = controls + assignment[:, :, None] * tau[None, None, :]
    index = np.repeat(np.arange(number_pairs), 2)
    return assignment.reshape(-1), observed.reshape(-1, number_outcomes), index


def prepare_study(
    index: Sequence[object], scores: np.ndarray, assignment: Sequence[int]
) -> PreparedStudy:
    """Validate and reshape scores for pairs or full-matching sets.

    Each set must contain exactly one treated unit or exactly one control unit.
    Sets of the latter type are converted to an equivalent one-selected-unit
    representation.
    """
    labels, inverse = np.unique(np.asarray(index), return_inverse=True)
    z = np.asarray(assignment, dtype=int).copy()
    q = np.asarray(scores, dtype=float).copy()
    if q.ndim == 1:
        q = q[:, None]
    if inverse.size != z.size or q.shape[0] != z.size:
        raise ValueError("index, scores, and assignment must have equal lengths")
    if np.any((z != 0) & (z != 1)):
        raise ValueError("assignment must be binary")
    if np.any(~np.isfinite(q)):
        raise ValueError("scores must be finite; remove endpoint-specific missing sets first")

    groups = [np.flatnonzero(inverse == b) for b in range(labels.size)]
    set_sizes = tuple(int(g.size) for g in groups)
    if any(size < 2 for size in set_sizes):
        raise ValueError("every matched set must contain at least two units")

    for positions in groups:
        treated = int(z[positions].sum())
        controls = positions.size - treated
        if treated != 1 and controls != 1:
            raise ValueError("each set must contain one treated unit or one control unit")
        if treated > 1:
            z[positions] = 1 - z[positions]
            totals = q[positions].sum(axis=0)
            q[positions] = totals - q[positions]

    if any(int(z[g].sum()) != 1 for g in groups):
        raise RuntimeError("failed to convert matched sets to one selected unit")

    scale = q.std(axis=0) * np.sqrt(q.shape[0])
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        bad = np.flatnonzero((~np.isfinite(scale)) | (scale <= 0)).tolist()
        raise ValueError(f"score columns have zero or non-finite scale: {bad}")
    q /= scale

    observed = q[z == 1].sum(axis=0)
    padded = np.zeros((labels.size, max(set_sizes), q.shape[1]), dtype=float)
    for b, positions in enumerate(groups):
        padded[b, : positions.size, :] = q[positions]
    return PreparedStudy(
        number_sets=labels.size,
        number_outcomes=q.shape[1],
        set_sizes=set_sizes,
        scores=padded,
        observed_statistics=observed,
        set_labels=labels,
    )


def data_process(index, Qmat, Z):
    """Compatibility wrapper using the legacy argument names ``Qmat`` and ``Z``."""
    study = prepare_study(index, Qmat, Z)
    return (
        study.number_sets,
        study.number_outcomes,
        list(study.set_sizes),
        study.scores,
        study.observed_statistics,
    )
