"""Construction of matched-set score statistics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _groups(index: Sequence[object]) -> tuple[np.ndarray, list[np.ndarray]]:
    labels, inverse = np.unique(np.asarray(index), return_inverse=True)
    groups = [np.flatnonzero(inverse == b) for b in range(labels.size)]
    return labels, groups


def binary_scores(outcome: Sequence[float]) -> np.ndarray:
    """Return the observed binary outcomes as floating-point scores."""
    y = np.asarray(outcome, dtype=float)
    observed = y[~np.isnan(y)]
    if observed.size == 0 or not np.all(np.isin(observed, [0.0, 1.0])):
        raise ValueError("binary outcome must contain only 0, 1, and optional NaN")
    return y.copy()


def difference_in_means_scores(
    outcome: Sequence[float], index: Sequence[object]
) -> np.ndarray:
    """Construct the within-set difference-in-means score."""
    y = np.asarray(outcome, dtype=float)
    if y.ndim != 1 or y.size != len(index):
        raise ValueError("outcome and index must be one-dimensional and equally long")
    _, groups = _groups(index)
    number_sets = len(groups)
    scores = np.full(y.shape, np.nan, dtype=float)
    for positions in groups:
        valid = positions[~np.isnan(y[positions])]
        n = valid.size
        if n < 2:
            continue
        total = y[valid].sum()
        scores[valid] = (n * y[valid] - total) / (number_sets * (n - 1))
    return scores


def m_scores(
    outcome: Sequence[float],
    index: Sequence[object],
    *,
    inner: float = 0.0,
    trim: float = 2.5,
    scale_quantile: float | None = 0.5,
) -> np.ndarray:
    """Construct Huber-type matched-set M-scores.

    Set ``scale_quantile=None`` to apply the score transformation without a
    data-dependent scale, as used for ordinal outcomes.
    """
    y = np.asarray(outcome, dtype=float)
    if y.ndim != 1 or y.size != len(index):
        raise ValueError("outcome and index must be one-dimensional and equally long")
    if inner < 0 or trim <= 0 or inner > trim:
        raise ValueError("require 0 <= inner <= trim and trim > 0")
    if scale_quantile is not None and not 0 <= scale_quantile <= 1:
        raise ValueError("scale_quantile must lie in [0, 1]")

    _, groups = _groups(index)
    valid_groups: list[tuple[np.ndarray, np.ndarray]] = []
    absolute_differences: list[np.ndarray] = []
    for positions in groups:
        valid = positions[~np.isnan(y[positions])]
        if valid.size < 2:
            continue
        values = y[valid]
        differences = values[:, None] - values[None, :]
        off_diagonal = differences[~np.eye(valid.size, dtype=bool)]
        absolute_differences.append(np.abs(off_diagonal))
        valid_groups.append((valid, differences))

    if not valid_groups:
        raise ValueError("no matched set contains two observed outcomes")

    if scale_quantile is None:
        scale = 1.0
    else:
        scale = float(np.quantile(np.concatenate(absolute_differences), scale_quantile))
        if scale <= 0:
            return difference_in_means_scores(y, index)

    scores = np.full(y.shape, np.nan, dtype=float)
    for valid, differences in valid_groups:
        standardized = differences / scale
        if inner < trim:
            magnitude = np.clip((np.abs(standardized) - inner) / (trim - inner), 0, 1)
        else:
            magnitude = (np.abs(standardized) > inner).astype(float)
        transformed = np.sign(standardized) * magnitude
        scores[valid] = transformed.sum(axis=1) / valid.size
    return scores


def classify_outcomes(
    outcomes: np.ndarray, ordinal_threshold: int = 10
) -> tuple[list[int], list[int], list[int]]:
    """Classify columns as binary, ordinal, or continuous."""
    matrix = np.asarray(outcomes, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("outcomes must be a two-dimensional matrix")
    binary: list[int] = []
    ordinal: list[int] = []
    continuous: list[int] = []
    for k in range(matrix.shape[1]):
        observed = matrix[:, k][~np.isnan(matrix[:, k])]
        if observed.size == 0:
            raise ValueError(f"outcome column {k} is entirely missing")
        unique = np.unique(observed)
        if np.all(np.isin(unique, [0.0, 1.0])):
            binary.append(k)
        elif np.all(np.isclose(unique, np.round(unique))) and unique.size <= ordinal_threshold:
            ordinal.append(k)
        else:
            continuous.append(k)
    return binary, ordinal, continuous


def outcome_scores(
    outcomes: np.ndarray,
    index: Sequence[object],
    *,
    ordinal_threshold: int = 10,
    inner: float = 0.0,
    trim: float = 2.5,
    scale_quantile: float = 0.5,
) -> np.ndarray:
    """Construct one score column for every outcome."""
    matrix = np.asarray(outcomes, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[0] != len(index):
        raise ValueError("outcomes and index must contain the same number of units")

    result = np.empty(matrix.shape, dtype=float)
    binary, ordinal, continuous = classify_outcomes(matrix, ordinal_threshold)
    for k in binary:
        result[:, k] = binary_scores(matrix[:, k])
    for k in ordinal:
        result[:, k] = m_scores(
            matrix[:, k], index, inner=inner, trim=trim, scale_quantile=None
        )
    for k in continuous:
        result[:, k] = m_scores(
            matrix[:, k],
            index,
            inner=inner,
            trim=trim,
            scale_quantile=scale_quantile,
        )
    return result

