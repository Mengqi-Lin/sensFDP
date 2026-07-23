"""Data preparation for the WLS childhood-abuse application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from .data import PreparedStudy, prepare_study
from .optimization import no_hidden_bias_pvalue
from .scores import binary_scores, m_scores


RAW_OUTCOME_CODES: Final[tuple[str, ...]] = (
    "gu034rec",
    "gc040re",
    "gb001re",
    "ix013rec",
    "gp250rec",
    "ix011rec",
    "iuc34rec",
    "iua33rec",
    "iv201rer",
    "iv203rer",
    "in046rec",
    "in037rec",
    "in028rec",
    "in019rec",
    "in010rec",
    "in001rec",
    "ih032rec",
    "ih025rec",
    "ih017rec",
    "ih009rec",
    "ih001rec",
    "in070rec",
)

COVARIATE_CODES: Final[tuple[str, ...]] = (
    "sexrsp",
    "sibcount",
    "hb042re",
    "piearl",
    "ocf157",
    "ocm157",
    "ses57",
    "sesp57",
    "edfa57q",
    "edmo57q",
    "res57",
    "pop57",
    "rlur57",
)

# Ordered exactly as the 21 hypotheses in the manuscript analysis.
OUTCOME_CODE_TO_NAME: Final[dict[str, str]] = {
    "gu034rec": "alcohol",
    "gc040re": "spouse",
    "gb001re": "college",
    "ix013rec": "smoke",
    "gp250rec": "income",
    "iuc34rec": "anger",
    "iua33rec": "anxiety",
    "in046rec": "self_acceptance",
    "in037rec": "purpose_in_life",
    "in028rec": "positive_relations",
    "in019rec": "personal_growth",
    "in010rec": "environmental_mastery",
    "in001rec": "autonomy",
    "ih032rec": "openness",
    "ih025rec": "neuroticism",
    "ih017rec": "conscientiousness",
    "ih009rec": "agreeableness",
    "ih001rec": "extraversion",
    "in070rec": "optimism",
    "social_support_avg": "social_support",
    "obesity": "obesity",
}
OUTCOME_NAMES: Final[tuple[str, ...]] = tuple(OUTCOME_CODE_TO_NAME.values())
BINARY_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"college", "smoke", "obesity"}
)
ORDINAL_TRIMS: Final[dict[str, float]] = {"alcohol": 4.0, "spouse": 2.0}


@dataclass(frozen=True)
class ApplicationFrames:
    """Clean application data and diagnostics before sensitivity analysis."""

    complete_cases: pd.DataFrame
    matched_sample: pd.DataFrame
    score_matrix: pd.DataFrame
    nominal_pvalues: pd.DataFrame
    study: PreparedStudy


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def clean_complete_cases(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the legacy outcome recoding and complete-case restriction."""
    _require_columns(
        raw,
        ("idpub", "Z", *COVARIATE_CODES, *RAW_OUTCOME_CODES),
        "raw WLS extract",
    )
    result = raw.copy()
    if result["idpub"].duplicated().any():
        raise ValueError("idpub must uniquely identify rows in the WLS extract")
    for code in RAW_OUTCOME_CODES:
        result[code] = pd.to_numeric(result[code], errors="coerce")
        result.loc[result[code] < 0, code] = np.nan
        unique = set(result[code].dropna().unique().tolist())
        if unique == {1.0, 2.0} or unique == {1, 2}:
            result[code] = result[code].replace(2, 0)
    return result.dropna(subset=["Z", *RAW_OUTCOME_CODES]).copy()


def build_matched_sample(
    complete_cases: pd.DataFrame, matched_index: pd.DataFrame
) -> pd.DataFrame:
    """Join the supplied R matching output and construct the 21 outcomes."""
    _require_columns(matched_index, ("idpub", "index"), "matched index")
    if matched_index["idpub"].duplicated().any():
        raise ValueError("matched_index.csv contains duplicate idpub values")
    matched = matched_index.merge(
        complete_cases, on="idpub", how="left", sort=False, validate="one_to_one"
    )
    if matched["Z"].isna().any():
        missing = matched.loc[matched["Z"].isna(), "idpub"].tolist()[:10]
        raise ValueError(f"matched IDs absent from complete cases: {missing}")

    matched["social_support_avg"] = matched[["iv201rer", "iv203rer"]].mean(
        axis=1
    )
    matched["obesity"] = (matched["ix011rec"] >= 30).astype(int)

    final_codes = tuple(OUTCOME_CODE_TO_NAME)
    selected = matched[["idpub", "index", "Z", *final_codes]].copy()
    selected.rename(columns=OUTCOME_CODE_TO_NAME, inplace=True)
    selected["Z"] = selected["Z"].astype(int)

    sizes = selected.groupby("index", sort=False).size()
    treated = selected.groupby("index", sort=False)["Z"].sum()
    valid = (treated == 1) | ((sizes - treated) == 1)
    if not bool(valid.all()):
        bad = valid.index[~valid].tolist()[:10]
        raise ValueError(f"matched sets without one treated or one control: {bad}")
    if (sizes < 2).any():
        raise ValueError("every matched set must have at least two units")

    # The analysis code uses compact zero-based labels; the original labels are
    # preserved in matched_index.csv.
    selected["index"] = pd.factorize(selected["index"], sort=False)[0]
    return selected[["idpub", "index", "Z", *OUTCOME_NAMES]]


def build_score_matrix(matched_sample: pd.DataFrame) -> pd.DataFrame:
    """Construct the 21 score columns used by the matched-set statistics."""
    _require_columns(
        matched_sample,
        ("index", "Z", *OUTCOME_NAMES),
        "matched application sample",
    )
    index = matched_sample["index"].to_numpy()
    scores = np.empty((len(matched_sample), len(OUTCOME_NAMES)), dtype=float)
    for k, name in enumerate(OUTCOME_NAMES):
        outcome = matched_sample[name].to_numpy(dtype=float)
        if name in BINARY_OUTCOMES:
            scores[:, k] = binary_scores(outcome)
        elif name in ORDINAL_TRIMS:
            scores[:, k] = m_scores(
                outcome,
                index,
                inner=0.0,
                trim=ORDINAL_TRIMS[name],
                scale_quantile=None,
            )
        else:
            scores[:, k] = m_scores(
                outcome,
                index,
                inner=0.0,
                trim=2.5,
                scale_quantile=0.5,
            )
    result = pd.DataFrame(scores, columns=OUTCOME_NAMES)
    result["Z"] = matched_sample["Z"].to_numpy(dtype=int)
    result["index"] = matched_sample["index"].to_numpy(dtype=int)
    return result


def nominal_pvalue_table(study: PreparedStudy, alpha: float = 0.05) -> pd.DataFrame:
    """Return normally approximated two-sided p-values at ``Gamma=1``."""
    if study.number_outcomes != len(OUTCOME_NAMES):
        raise ValueError("application study must contain all 21 outcomes")
    values = [no_hidden_bias_pvalue(study, k) for k in range(study.number_outcomes)]
    return pd.DataFrame(
        {
            "outcome_index": np.arange(study.number_outcomes, dtype=int),
            "outcome": OUTCOME_NAMES,
            "p_value": values,
            "selected_at_0.05": np.asarray(values) <= alpha,
        }
    )


def prepare_application_frames(
    raw: pd.DataFrame, matched_index: pd.DataFrame, alpha: float = 0.05
) -> ApplicationFrames:
    """Run deterministic preparation from the supplied extract and matches."""
    complete = clean_complete_cases(raw)
    sample = build_matched_sample(complete, matched_index)
    scores = build_score_matrix(sample)
    study = prepare_study(
        scores["index"],
        scores[list(OUTCOME_NAMES)].to_numpy(dtype=float),
        scores["Z"],
    )
    nominal = nominal_pvalue_table(study, alpha=alpha)
    return ApplicationFrames(complete, sample, scores, nominal, study)


def covariate_sentinel_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count negative WLS sentinel codes in the matching covariates."""
    _require_columns(frame, COVARIATE_CODES, "WLS extract")
    rows = []
    for code in COVARIATE_CODES:
        values = pd.to_numeric(frame[code], errors="coerce")
        negative = values[values < 0]
        rows.append(
            {
                "covariate": code,
                "negative_count": int(negative.size),
                "negative_codes": ",".join(
                    str(value) for value in sorted(negative.dropna().unique())
                ),
            }
        )
    return pd.DataFrame(rows)
