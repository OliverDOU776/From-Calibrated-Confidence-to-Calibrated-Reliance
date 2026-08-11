"""Baseline reliance model and off-policy counterfactual evaluation.

The feature specification mirrors the accepted paper's
``scripts/revision_diagnostics.py`` baseline: displayed confidence,
pre-advice self-confidence, AI-source indicator, pre-advice advice gap, their
display-by-self-confidence interaction, task indicators, and advice source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metrics import reliance_calibration_error

NUMERIC_FEATURES: tuple[str, ...] = (
    "displayed",
    "prob_correct_1",
    "is_ai",
    "abs_gap_pre_advice",
    "display_x_self",
)
"""Numeric predictors in the accepted-paper baseline reliance model."""

CATEGORICAL_FEATURES: tuple[str, ...] = ("task_name", "advice_source")
"""Categorical predictors in the accepted-paper baseline reliance model."""

RELIANCE_TARGET = "shifted_toward"


@dataclass(frozen=True)
class CounterfactualEvaluation:
    """Outputs of model-based evaluation under one display policy."""

    response: np.ndarray
    reliance_probability: np.ndarray
    mse: float
    rce: float
    mean_reliance: float


def _require_frame(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Reliance-model data must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Reliance-model data must contain at least one row.")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Reliance-model data are missing required columns: {missing}")


def _displayed(displayed: Iterable[float], expected_length: int) -> np.ndarray:
    try:
        values = np.asarray(displayed, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Displayed confidence must be numeric.") from exc
    if values.ndim != 1 or values.size != expected_length:
        raise ValueError(
            f"Displayed confidence must be one-dimensional with length {expected_length}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Displayed confidence must be finite.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Displayed confidence must lie within [0, 1].")
    return values


def _finite_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = frame[column].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column '{column}' must be numeric.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Column '{column}' must be finite.")
    return values


def build_feature_frame(frame: pd.DataFrame, displayed: Iterable[float]) -> pd.DataFrame:
    """Construct the exact baseline feature frame for a proposed display.

    ``frame`` must have been enriched with :func:`derive_haiid_features` or
    :func:`prepare_haiid` before this function is called.
    """

    required = (
        "prob_correct_1",
        "is_ai",
        "abs_gap_pre_advice",
        "task_name",
        "advice_source",
    )
    _require_frame(frame, required)
    display = _displayed(displayed, len(frame))
    self_confidence = _finite_column(frame, "prob_correct_1")
    is_ai = _finite_column(frame, "is_ai")
    advice_gap = _finite_column(frame, "abs_gap_pre_advice")

    if np.any((self_confidence < 0.0) | (self_confidence > 1.0)):
        raise ValueError("Column 'prob_correct_1' must lie within [0, 1].")
    if np.any((is_ai < 0.0) | (is_ai > 1.0)):
        raise ValueError("Column 'is_ai' must lie within [0, 1].")

    features = pd.DataFrame(
        {
            "displayed": display,
            "prob_correct_1": self_confidence,
            "is_ai": is_ai,
            "abs_gap_pre_advice": advice_gap,
            "display_x_self": display * self_confidence,
            "task_name": frame["task_name"].astype(str).to_numpy(),
            "advice_source": frame["advice_source"].astype(str).to_numpy(),
        },
        index=frame.index,
    )
    return features


def make_reliance_pipeline(
    *,
    random_state: int = 42,
    max_iter: int = 1500,
) -> Pipeline:
    """Create the accepted-paper baseline logistic reliance pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    estimator = LogisticRegression(max_iter=max_iter, random_state=random_state)
    return Pipeline([("pre", preprocessor), ("model", estimator)])


def fit_reliance_model(
    train_frame: pd.DataFrame,
    displayed: Iterable[float] | None = None,
    *,
    random_state: int = 42,
    max_iter: int = 1500,
) -> Pipeline:
    """Fit the baseline shift-toward-advice model on logged interactions.

    By default, training uses the observed ``advice_prob`` display.  Supplying
    another display vector is supported for controlled diagnostics, but policy
    evaluation should ordinarily fit on observed displays and intervene only
    at prediction time.
    """

    _require_frame(train_frame, ("advice_prob", RELIANCE_TARGET))
    observed_display = (
        _finite_column(train_frame, "advice_prob") if displayed is None else displayed
    )
    features = build_feature_frame(train_frame, observed_display)
    target = _finite_column(train_frame, RELIANCE_TARGET)
    if np.any((target != 0.0) & (target != 1.0)):
        raise ValueError("Column 'shifted_toward' must be binary.")
    if np.unique(target).size < 2:
        raise ValueError("Reliance-model training requires both target classes.")

    pipeline = make_reliance_pipeline(random_state=random_state, max_iter=max_iter)
    pipeline.fit(features, target.astype(int))
    return pipeline


def predict_reliance(
    model: Pipeline,
    frame: pd.DataFrame,
    displayed: Iterable[float],
) -> np.ndarray:
    """Predict the probability of shifting toward advice under a display."""

    if not hasattr(model, "predict_proba"):
        raise TypeError("model must expose predict_proba, as the baseline pipeline does.")
    features = build_feature_frame(frame, displayed)
    probability = np.asarray(model.predict_proba(features), dtype=float)
    if probability.ndim != 2 or probability.shape != (len(frame), 2):
        raise ValueError("The reliance model returned an unexpected probability shape.")
    result = probability[:, 1]
    if not np.all(np.isfinite(result)):
        raise ValueError("The reliance model returned non-finite probabilities.")
    return np.clip(result, 0.0, 1.0)


def counterfactual_response(
    model: Pipeline,
    frame: pd.DataFrame,
    displayed: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate expected post-advice response under a display policy.

    The paper's response equation is
    ``r2 = r1 + pi(display, x, u) * (advice - r1)``.
    """

    _require_frame(frame, ("response_1", "advice"))
    response_1 = _finite_column(frame, "response_1")
    advice = _finite_column(frame, "advice")
    if np.any((response_1 < -1.0) | (response_1 > 1.0)):
        raise ValueError("Column 'response_1' must lie within [-1, 1].")
    if np.any((advice < -1.0) | (advice > 1.0)):
        raise ValueError("Column 'advice' must lie within [-1, 1].")

    probability = predict_reliance(model, frame, displayed)
    response = response_1 + probability * (advice - response_1)
    return response, probability


def team_mse(
    response: Iterable[float],
    target: float | Iterable[float] = 1.0,
) -> float:
    """Compute mean squared error on the signed HAIID response scale."""

    try:
        prediction = np.asarray(response, dtype=float)
        truth = np.asarray(target, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("response and target must be numeric.") from exc
    if prediction.ndim != 1 or prediction.size == 0:
        raise ValueError("response must be a non-empty one-dimensional array.")
    if truth.ndim > 1 or (truth.ndim == 1 and truth.shape != prediction.shape):
        raise ValueError("target must be scalar or have the same shape as response.")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(truth)):
        raise ValueError("response and target must be finite.")
    return float(np.mean(np.square(prediction - truth)))


def counterfactual_rce(
    displayed: Iterable[float],
    response: Iterable[float],
    *,
    n_bins: int = 10,
    scheme: str = "equal_width",
) -> float:
    """Compute model-predicted counterfactual RCE from signed responses."""

    response_array = np.asarray(response, dtype=float)
    if response_array.ndim != 1 or response_array.size == 0:
        raise ValueError("response must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(response_array)):
        raise ValueError("response must be finite.")
    if np.any((response_array < -1.0) | (response_array > 1.0)):
        raise ValueError("response must lie within [-1, 1].")
    team_score = np.clip((response_array + 1.0) / 2.0, 0.0, 1.0)
    return reliance_calibration_error(
        displayed,
        team_score,
        n_bins=n_bins,
        scheme=scheme,
    )


def team_mse_and_rce(
    model: Pipeline,
    frame: pd.DataFrame,
    displayed: Iterable[float],
    *,
    n_bins: int = 10,
    scheme: str = "equal_width",
) -> tuple[float, float, float]:
    """Return HAIID MSE, counterfactual RCE, and mean predicted reliance."""

    display = _displayed(displayed, len(frame))
    response, probability = counterfactual_response(model, frame, display)
    mse = team_mse(response, 1.0)
    calibration = counterfactual_rce(
        display,
        response,
        n_bins=n_bins,
        scheme=scheme,
    )
    return mse, calibration, float(np.mean(probability))


def evaluate_policy(
    model: Pipeline,
    frame: pd.DataFrame,
    displayed: Iterable[float],
    *,
    n_bins: int = 10,
    scheme: str = "equal_width",
) -> CounterfactualEvaluation:
    """Return the full model-based evaluation for one display vector."""

    display = _displayed(displayed, len(frame))
    response, probability = counterfactual_response(model, frame, display)
    return CounterfactualEvaluation(
        response=response,
        reliance_probability=probability,
        mse=team_mse(response, 1.0),
        rce=counterfactual_rce(
            display,
            response,
            n_bins=n_bins,
            scheme=scheme,
        ),
        mean_reliance=float(np.mean(probability)),
    )
