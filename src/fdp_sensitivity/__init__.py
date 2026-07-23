"""Simultaneous FDP sensitivity analysis."""

from .data import PreparedStudy, generate_pair_data, prepare_study, select_outcomes
from .multiple_testing import holm_bonferroni, naive_fdp_bound
from .scores import outcome_scores
from .sensitivity import (
    SensitivityValueComparison,
    compare_generalized_sensitivity_values,
    generalized_sensitivity_values,
)
from .simulation_settings import (
    MANUSCRIPT_SELECTION_SETTINGS,
    SELECTION_SETTINGS,
    SelectionSetting,
    get_selection_setting,
)

__all__ = [
    "PreparedStudy",
    "SelectionSetting",
    "SensitivityValueComparison",
    "SELECTION_SETTINGS",
    "MANUSCRIPT_SELECTION_SETTINGS",
    "generate_pair_data",
    "compare_generalized_sensitivity_values",
    "generalized_sensitivity_values",
    "get_selection_setting",
    "holm_bonferroni",
    "naive_fdp_bound",
    "outcome_scores",
    "prepare_study",
    "select_outcomes",
]
