"""Confidence-display policies from the accepted paper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

TERCILE_LABELS: tuple[str, ...] = ("low", "mid", "high")


def _confidence(values: float | Iterable[float]) -> tuple[np.ndarray, bool]:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Confidence values must be numeric.") from exc
    scalar = array.ndim == 0
    if scalar:
        array = array.reshape(1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Confidence values must be a scalar or non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Confidence values must be finite.")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("Confidence values must lie within [0, 1].")
    return array, scalar


def _restore(array: np.ndarray, scalar: bool) -> float | np.ndarray:
    return float(array[0]) if scalar else array


def g0(confidence: float | Iterable[float]) -> float | np.ndarray:
    """Direct display: ``g0(c) = c``."""

    array, scalar = _confidence(confidence)
    return _restore(array.copy(), scalar)


def g1(confidence: float | Iterable[float]) -> float | np.ndarray:
    """Paper-fitted global mapping: ``clip(2.4 * c - 0.5, 0, 1)``."""

    array, scalar = _confidence(confidence)
    mapped = np.clip(2.4 * array - 0.5, 0.0, 1.0)
    return _restore(mapped, scalar)


def g3(confidence: float | Iterable[float]) -> float | np.ndarray:
    """Default robustness guard: ``clip(g1(c), 0.15, 0.85)``."""

    array, scalar = _confidence(confidence)
    mapped = np.clip(2.4 * array - 0.5, 0.15, 0.85)
    return _restore(mapped, scalar)


def bounded_shift(
    confidence: float | Iterable[float],
    proposed_display: float | Iterable[float],
    *,
    delta: float,
    display_low: float = 0.05,
    display_high: float = 0.95,
) -> float | np.ndarray:
    """Constrain a proposed display by shift and practical display bounds.

    This implements the bounded-policy diagnostic in
    ``scripts/revision_diagnostics.py``: first enforce
    ``|display - confidence| <= delta``, then clip to the practical display
    range.  The default range is ``[0.05, 0.95]``.
    """

    raw, raw_scalar = _confidence(confidence)
    proposed, proposed_scalar = _confidence(proposed_display)
    if raw.shape != proposed.shape:
        raise ValueError("confidence and proposed_display must have the same shape.")
    if raw_scalar != proposed_scalar:
        raise ValueError("confidence and proposed_display must have the same shape.")
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("delta must be a finite non-negative value.")
    if not (0.0 <= display_low < display_high <= 1.0):
        raise ValueError("Display bounds must satisfy 0 <= low < high <= 1.")

    shifted = np.clip(proposed, raw - delta, raw + delta)
    bounded = np.clip(shifted, display_low, display_high)
    return _restore(bounded, raw_scalar)


def bounded_g1(
    confidence: float | Iterable[float],
    delta: float,
    *,
    display_low: float = 0.05,
    display_high: float = 0.95,
) -> float | np.ndarray:
    """Apply the paper's global map subject to a maximum display shift."""

    return bounded_shift(
        confidence,
        g1(confidence),
        delta=delta,
        display_low=display_low,
        display_high=display_high,
    )


def _vector(values: Iterable[float], name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    return array


def _self_confidence(values: Iterable[float]) -> np.ndarray:
    array = _vector(values, "self_confidence")
    if np.any(np.isinf(array)):
        raise ValueError("self_confidence must not contain infinite values.")
    observed = array[~np.isnan(array)]
    if np.any((observed < 0.0) | (observed > 1.0)):
        raise ValueError("self_confidence must lie within [0, 1] when observed.")
    return array


def _group_array(groups: Iterable[object], *, allow_unknown: bool) -> np.ndarray:
    array = np.asarray(groups, dtype=object)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Tercile groups must be a non-empty one-dimensional array.")
    if not allow_unknown:
        observed = {str(value) for value in array if not pd.isna(value)}
        unknown = sorted(observed.difference(TERCILE_LABELS))
        if unknown:
            raise ValueError(
                "g2 accepts only self-confidence terciles "
                f"{list(TERCILE_LABELS)}; received {unknown}."
            )
    return array


@dataclass
class SelfConfidenceIsotonicPolicy:
    """Self-confidence-tercile isotonic display policy (``g2``).

    A global isotonic model is always fitted.  Each low/mid/high
    self-confidence group receives its own isotonic model only when it has at
    least ``min_group_size`` rows and ``min_unique_confidences`` distinct
    confidence values.  Sparse, missing, or unknown groups use the global
    model, matching the accepted paper's revision diagnostics.

    This class deliberately does not use task, advice source, gender, or any
    other legacy fairness grouping.
    """

    min_group_size: int = 100
    min_unique_confidences: int = 3
    models_: dict[str, IsotonicRegression] = field(default_factory=dict, init=False)
    global_model_: IsotonicRegression | None = field(default=None, init=False)
    tercile_cutpoints_: tuple[float, float] | None = field(default=None, init=False)
    group_counts_: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        valid_group_size = isinstance(self.min_group_size, (int, np.integer)) and not isinstance(
            self.min_group_size, bool
        )
        if not valid_group_size or self.min_group_size <= 0:
            raise ValueError("min_group_size must be a positive integer.")
        valid_unique_count = isinstance(
            self.min_unique_confidences, (int, np.integer)
        ) and not isinstance(self.min_unique_confidences, bool)
        if not valid_unique_count or self.min_unique_confidences <= 0:
            raise ValueError("min_unique_confidences must be a positive integer.")

    @staticmethod
    def _new_model() -> IsotonicRegression:
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def _labels_from_self_confidence(self, values: np.ndarray) -> np.ndarray:
        labels = np.full(values.shape, None, dtype=object)
        observed = ~np.isnan(values)
        if self.tercile_cutpoints_ is None:
            return labels
        lower, upper = self.tercile_cutpoints_
        labels[observed & (values <= lower)] = "low"
        labels[observed & (values > lower) & (values <= upper)] = "mid"
        labels[observed & (values > upper)] = "high"
        return labels

    def fit(
        self,
        confidence: Iterable[float],
        self_confidence: Iterable[float],
        team_correctness: Iterable[float],
        *,
        groups: Iterable[object] | None = None,
    ) -> "SelfConfidenceIsotonicPolicy":
        """Fit global and self-confidence-tercile isotonic regressions.

        ``groups`` is optional.  It supports the exact revision workflow in
        which terciles were derived once on the prepared HAIID cohort before
        the train/validation split.  When omitted, cut points are learned from
        the supplied training self-confidence values.
        """

        c, _ = _confidence(confidence)
        self_c = _self_confidence(self_confidence)
        outcome = _vector(team_correctness, "team_correctness")
        if not (c.shape == self_c.shape == outcome.shape):
            raise ValueError("confidence, self_confidence, and team_correctness must align.")
        if not np.all(np.isfinite(outcome)):
            raise ValueError("team_correctness must be finite.")
        if np.any((outcome < 0.0) | (outcome > 1.0)):
            raise ValueError("team_correctness must lie within [0, 1].")

        observed_self = self_c[~np.isnan(self_c)]
        self.tercile_cutpoints_ = (
            tuple(float(value) for value in np.quantile(observed_self, [1 / 3, 2 / 3]))
            if observed_self.size
            else None
        )
        if groups is None:
            labels = self._labels_from_self_confidence(self_c)
        else:
            labels = _group_array(groups, allow_unknown=False)
            if labels.shape != c.shape:
                raise ValueError("groups must align with confidence.")

        self.global_model_ = self._new_model().fit(c, outcome)
        self.models_ = {}
        self.group_counts_ = {}
        for label in TERCILE_LABELS:
            mask = np.array(
                [not pd.isna(value) and str(value) == label for value in labels],
                dtype=bool,
            )
            count = int(np.sum(mask))
            self.group_counts_[label] = count
            enough_rows = count >= self.min_group_size
            enough_confidences = np.unique(c[mask]).size >= self.min_unique_confidences
            if enough_rows and enough_confidences:
                self.models_[label] = self._new_model().fit(c[mask], outcome[mask])
            else:
                self.models_[label] = self.global_model_
        return self

    def _require_fitted(self) -> IsotonicRegression:
        if self.global_model_ is None:
            raise RuntimeError("The g2 policy must be fitted before prediction.")
        return self.global_model_

    def predict(
        self,
        confidence: Iterable[float],
        self_confidence: Iterable[float],
    ) -> np.ndarray:
        """Map confidence using terciles derived from numeric self-confidence."""

        c, _ = _confidence(confidence)
        self_c = _self_confidence(self_confidence)
        if c.shape != self_c.shape:
            raise ValueError("confidence and self_confidence must align.")
        labels = self._labels_from_self_confidence(self_c)
        return self.predict_groups(c, labels)

    def predict_groups(
        self,
        confidence: Iterable[float],
        groups: Iterable[object],
    ) -> np.ndarray:
        """Map confidence using explicit low/mid/high tercile labels.

        Missing and unrecognized labels are intentionally handled by the
        global isotonic fallback.
        """

        c, _ = _confidence(confidence)
        labels = _group_array(groups, allow_unknown=True)
        if c.shape != labels.shape:
            raise ValueError("confidence and groups must align.")
        global_model = self._require_fitted()
        result = np.asarray(global_model.predict(c), dtype=float)
        for label in TERCILE_LABELS:
            mask = np.array(
                [not pd.isna(value) and str(value) == label for value in labels],
                dtype=bool,
            )
            if np.any(mask):
                model = self.models_.get(label, global_model)
                result[mask] = model.predict(c[mask])
        return np.clip(result, 0.0, 1.0)


def fit_g2(
    data: pd.DataFrame | Iterable[float],
    self_confidence: Iterable[float] | None = None,
    team_correctness: Iterable[float] | None = None,
    *,
    min_group_size: int = 100,
    min_unique_confidences: int = 3,
    confidence_col: str = "advice_prob",
    self_confidence_col: str = "prob_correct_1",
    outcome_col: str = "correct_post",
) -> SelfConfidenceIsotonicPolicy:
    """Fit ``g2`` from a prepared HAIID frame or aligned arrays."""

    policy = SelfConfidenceIsotonicPolicy(
        min_group_size=min_group_size,
        min_unique_confidences=min_unique_confidences,
    )
    if isinstance(data, pd.DataFrame):
        required = [confidence_col, self_confidence_col, outcome_col]
        missing = [column for column in required if column not in data]
        if missing:
            raise ValueError(f"Prepared HAIID data are missing columns: {missing}")
        groups = data["pre_conf_tercile"] if "pre_conf_tercile" in data else None
        return policy.fit(
            data[confidence_col],
            data[self_confidence_col],
            data[outcome_col],
            groups=groups,
        )
    if self_confidence is None or team_correctness is None:
        raise ValueError("Array input requires both self_confidence and team_correctness.")
    return policy.fit(data, self_confidence, team_correctness)


def apply_g2(
    policy: SelfConfidenceIsotonicPolicy,
    data: pd.DataFrame | Iterable[float],
    self_confidence: Iterable[float] | None = None,
    *,
    confidence_col: str = "advice_prob",
    self_confidence_col: str = "prob_correct_1",
) -> np.ndarray:
    """Apply a fitted ``g2`` policy to a prepared frame or aligned arrays."""

    if not isinstance(policy, SelfConfidenceIsotonicPolicy):
        raise TypeError("policy must be a SelfConfidenceIsotonicPolicy.")
    if isinstance(data, pd.DataFrame):
        if confidence_col not in data:
            raise ValueError(f"Prepared HAIID data are missing column '{confidence_col}'.")
        if "pre_conf_tercile" in data:
            return policy.predict_groups(data[confidence_col], data["pre_conf_tercile"])
        if self_confidence_col not in data:
            raise ValueError(f"Prepared HAIID data are missing column '{self_confidence_col}'.")
        return policy.predict(data[confidence_col], data[self_confidence_col])
    if self_confidence is None:
        raise ValueError("Array input requires self_confidence.")
    return policy.predict(data, self_confidence)
