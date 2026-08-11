"""Loading, validation, and feature derivation for the HAIID dataset.

The accepted paper evaluates confidence-display policies on four HAIID tasks
(``art``, ``sarcasm``, ``cities``, and ``census``) in the common
``perceived_accuracy == 80`` condition.  This module makes that analysis
cohort the default and derives the exact behavioral variables used by the
paper's reliance model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

MAIN_TASKS: tuple[str, ...] = ("art", "sarcasm", "cities", "census")
"""The four HAIID tasks used in the paper's main analysis."""

MAIN_PERCEIVED_ACCURACY = 80
"""The common perceived-accuracy condition used for policy evaluation."""

RAW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "task_name",
    "task_instance_id",
    "advice_source",
    "advice",
    "participant_id",
    "response_1",
    "response_2",
    "perceived_accuracy",
)
"""Raw columns required by the public analysis pipeline."""

DERIVED_COLUMNS: tuple[str, ...] = (
    "prob_correct_1",
    "prob_correct_2",
    "advice_prob",
    "correct_pre",
    "correct_post",
    "ai_correct",
    "is_ai",
    "shifted_toward",
    "abs_gap_pre_advice",
    "disagreement",
    "pre_conf_tercile",
    "woa",
    "mse_post",
)
"""Columns added by :func:`derive_haiid_features`."""


def _require_dataframe(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("HAIID data must be provided as a pandas DataFrame.")


def _finite_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = frame[column].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column '{column}' must be numeric.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Column '{column}' must not contain missing or infinite values.")
    return values


def validate_haiid(frame: pd.DataFrame) -> None:
    """Validate the raw HAIID fields required by the analysis.

    Parameters
    ----------
    frame:
        A raw or feature-enriched HAIID data frame.

    Raises
    ------
    TypeError
        If ``frame`` is not a pandas data frame.
    ValueError
        If required columns are absent, values are missing/non-finite, or the
        signed confidence variables fall outside ``[-1, 1]``.

    Notes
    -----
    HAIID stores responses and advice as signed correctness confidences:
    positive values indicate the correct answer and negative values indicate
    the incorrect answer.  Consequently, the ground-truth target on this
    response scale is always ``+1``.
    """

    _require_dataframe(frame)
    missing = sorted(set(RAW_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"HAIID data are missing required columns: {missing}")
    if frame.empty:
        raise ValueError("HAIID data must contain at least one row.")

    for column in ("response_1", "response_2", "advice"):
        values = _finite_numeric(frame, column)
        if np.any((values < -1.0) | (values > 1.0)):
            raise ValueError(f"Column '{column}' must lie within [-1, 1].")

    _finite_numeric(frame, "perceived_accuracy")
    for column in ("task_name", "task_instance_id", "advice_source", "participant_id"):
        if frame[column].isna().any():
            raise ValueError(f"Column '{column}' must not contain missing values.")


def self_confidence_terciles(values: Iterable[float]) -> pd.Categorical:
    """Assign normalized self-confidence values to low/mid/high terciles.

    Tercile cut points are estimated from the supplied values.  Ties remain in
    the same group, so a highly discrete input can yield an empty group; the
    subgroup policy handles that case through its global fallback.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Self-confidence must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Self-confidence must not contain missing or infinite values.")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("Self-confidence must lie within [0, 1].")

    lower, upper = np.quantile(array, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.where(array <= lower, "low", np.where(array <= upper, "mid", "high"))
    return pd.Categorical(labels, categories=["low", "mid", "high"], ordered=True)


def derive_haiid_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of HAIID data with the paper's derived variables.

    The derivation follows ``scripts/revision_diagnostics.py``.  In
    particular, ``shifted_toward`` is one when the observed post-advice change
    has a positive projection onto the direction from the initial response to
    the advice, and ``pre_conf_tercile`` depends only on pre-advice
    self-confidence.
    """

    validate_haiid(frame)
    result = frame.copy()
    response_1 = result["response_1"].to_numpy(dtype=float)
    response_2 = result["response_2"].to_numpy(dtype=float)
    advice = result["advice"].to_numpy(dtype=float)

    result["prob_correct_1"] = (response_1 + 1.0) / 2.0
    result["prob_correct_2"] = (response_2 + 1.0) / 2.0
    result["advice_prob"] = (advice + 1.0) / 2.0
    result["correct_pre"] = (response_1 > 0.0).astype(int)
    result["correct_post"] = (response_2 > 0.0).astype(int)
    result["ai_correct"] = (advice > 0.0).astype(int)
    result["is_ai"] = (result["advice_source"].astype(str) == "ai").astype(int)
    result["shifted_toward"] = ((response_2 - response_1) * (advice - response_1) > 0.0).astype(int)
    result["abs_gap_pre_advice"] = np.abs(advice - response_1)
    result["disagreement"] = (np.sign(response_1) != np.sign(advice)).astype(int)
    result["pre_conf_tercile"] = self_confidence_terciles(result["prob_correct_1"])

    denominator = advice - response_1
    weight_of_advice = np.zeros_like(denominator, dtype=float)
    np.divide(
        response_2 - response_1,
        denominator,
        out=weight_of_advice,
        where=np.abs(denominator) > 0.01,
    )
    result["woa"] = np.clip(weight_of_advice, 0.0, 1.0)
    result["mse_post"] = np.square(response_2 - 1.0)
    return result


def prepare_haiid(
    frame: pd.DataFrame,
    *,
    tasks: Iterable[str] = MAIN_TASKS,
    perceived_accuracy: int | float = MAIN_PERCEIVED_ACCURACY,
) -> pd.DataFrame:
    """Validate, filter, and enrich the paper's main HAIID cohort.

    Parameters
    ----------
    frame:
        Raw HAIID records.
    tasks:
        Task names to retain.  Defaults to the four main-paper tasks.
    perceived_accuracy:
        Experimental perceived-accuracy condition to retain.  Defaults to 80.
    """

    validate_haiid(frame)
    selected_tasks = tuple(str(task).strip().lower() for task in tasks)
    if not selected_tasks:
        raise ValueError("At least one task must be selected.")
    unknown = sorted(set(selected_tasks).difference(MAIN_TASKS))
    if unknown:
        raise ValueError(f"The main analysis supports only {list(MAIN_TASKS)}; received {unknown}.")

    task_names = frame["task_name"].astype(str).str.lower()
    accuracy = frame["perceived_accuracy"].to_numpy(dtype=float)
    mask = task_names.isin(selected_tasks) & np.isclose(accuracy, float(perceived_accuracy))
    cohort = frame.loc[mask].copy()
    if cohort.empty:
        raise ValueError(
            "No HAIID rows remain after applying the task and perceived-accuracy filters."
        )
    cohort["task_name"] = task_names.loc[mask]
    return derive_haiid_features(cohort)


def load_haiid(
    path: str | Path,
    *,
    tasks: Iterable[str] = MAIN_TASKS,
    perceived_accuracy: int | float = MAIN_PERCEIVED_ACCURACY,
) -> pd.DataFrame:
    """Load a HAIID CSV and return the prepared main-analysis cohort.

    ``path`` may name either ``haiid_dataset.csv`` itself or the directory that
    contains it.  The source file is never modified.
    """

    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "haiid_dataset.csv"
    if not source.is_file():
        raise FileNotFoundError(f"HAIID dataset not found: {source}")
    frame = pd.read_csv(source, dtype={"job_title": str})
    return prepare_haiid(
        frame,
        tasks=tasks,
        perceived_accuracy=perceived_accuracy,
    )
