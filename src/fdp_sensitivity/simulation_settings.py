"""Simulation settings used for the subset-selection experiment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionSetting:
    effect: float
    effect_correlation: float
    null_correlation: float
    gamma_bias: float


# These use the zero-based order in the uploaded ``settings.py`` file.
SELECTION_SETTINGS = (
    SelectionSetting(0.25, 0.0, 0.2, 1.75),
    SelectionSetting(0.25, 0.0, 0.0, 1.86),
    SelectionSetting(0.25, 0.0, -0.2, 2.02),
    SelectionSetting(0.25, 0.2, 0.2, 1.75),
    SelectionSetting(0.25, 0.2, 0.0, 1.86),
    SelectionSetting(0.25, 0.2, -0.2, 2.02),
    SelectionSetting(0.25, -0.2, 0.2, 1.75),
    SelectionSetting(0.25, -0.2, 0.0, 1.85),
    SelectionSetting(0.25, -0.2, -0.2, 2.02),
)

# The current manuscript fixes the null-outcome correlation at 0.2 and orders
# the effect-outcome correlations as -0.2, 0, and 0.2.
MANUSCRIPT_SELECTION_SETTINGS = (6, 0, 3)


def get_selection_setting(index: int) -> SelectionSetting:
    if index not in range(len(SELECTION_SETTINGS)):
        raise ValueError(
            f"setting index must lie between 0 and {len(SELECTION_SETTINGS) - 1}"
        )
    return SELECTION_SETTINGS[index]
