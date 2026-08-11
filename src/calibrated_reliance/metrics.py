"""Calibration metrics for displayed confidence and team outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    """Summary of one non-empty reliance-reliability bin."""

    index: int
    count: int
    mean_display: float
    mean_team_score: float
    absolute_gap: float


def _validated_pair(
    displayed: Iterable[float], team_score: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    try:
        confidence = np.asarray(displayed, dtype=float)
        outcome = np.asarray(team_score, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Displayed confidence and team scores must be numeric.") from exc

    if confidence.ndim != 1 or outcome.ndim != 1:
        raise ValueError("Displayed confidence and team scores must be one-dimensional.")
    if confidence.size == 0:
        raise ValueError("Displayed confidence and team scores must not be empty.")
    if confidence.shape != outcome.shape:
        raise ValueError("Displayed confidence and team scores must have the same length.")
    if not np.all(np.isfinite(confidence)) or not np.all(np.isfinite(outcome)):
        raise ValueError("Displayed confidence and team scores must be finite.")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("Displayed confidence must lie within [0, 1].")
    if np.any((outcome < 0.0) | (outcome > 1.0)):
        raise ValueError("Team scores must lie within [0, 1].")
    return confidence, outcome


def _validate_bins(n_bins: int) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
        raise TypeError("n_bins must be a positive integer.")
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    return int(n_bins)


def _bin_indices(confidence: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    normalized_scheme = str(scheme).strip().lower().replace("-", "_")
    if normalized_scheme == "equal_width":
        return np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    if normalized_scheme != "equal_mass":
        raise ValueError("scheme must be either 'equal_width' or 'equal_mass'.")

    # Quantile edges preserve ties rather than arbitrarily splitting identical
    # displays across bins.  Degenerate confidence vectors form one valid bin.
    edges = np.unique(np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size == 1:
        return np.zeros(confidence.size, dtype=int)
    # pandas.qcut, used in the paper diagnostics, assigns an observation that
    # equals a quantile edge to the lower (right-closed) interval.
    return np.searchsorted(edges[1:-1], confidence, side="left").astype(int)


def reliability_bins(
    displayed: Iterable[float],
    team_score: Iterable[float],
    *,
    n_bins: int = 10,
    scheme: str = "equal_width",
) -> tuple[ReliabilityBin, ...]:
    """Summarize non-empty bins for a reliance reliability diagram.

    Parameters
    ----------
    displayed:
        Communicated confidence values in ``[0, 1]``.
    team_score:
        Binary team correctness or a model-predicted correctness score in
        ``[0, 1]``.
    n_bins:
        Requested number of bins.
    scheme:
        ``"equal_width"`` for fixed-width bins or ``"equal_mass"`` for
        empirical-quantile bins.
    """

    confidence, outcome = _validated_pair(displayed, team_score)
    count = _validate_bins(n_bins)
    indices = _bin_indices(confidence, count, scheme)

    summaries: list[ReliabilityBin] = []
    for index in np.unique(indices):
        mask = indices == index
        mean_display = float(np.mean(confidence[mask]))
        mean_outcome = float(np.mean(outcome[mask]))
        summaries.append(
            ReliabilityBin(
                index=int(index),
                count=int(np.sum(mask)),
                mean_display=mean_display,
                mean_team_score=mean_outcome,
                absolute_gap=abs(mean_outcome - mean_display),
            )
        )
    return tuple(summaries)


def reliance_calibration_error(
    displayed: Iterable[float],
    team_score: Iterable[float],
    *,
    n_bins: int = 10,
    scheme: str = "equal_width",
) -> float:
    """Compute weighted binned Reliance Calibration Error (RCE).

    RCE replaces model correctness in the standard Expected Calibration Error
    template with final-team correctness.  Model-predicted counterfactual team
    scores are also accepted, matching the evaluation in the accepted paper.
    Bin gaps use each bin's mean display rather than its geometric midpoint.
    """

    confidence, outcome = _validated_pair(displayed, team_score)
    summaries = reliability_bins(
        confidence,
        outcome,
        n_bins=n_bins,
        scheme=scheme,
    )
    total = confidence.size
    return float(sum(item.count * item.absolute_gap for item in summaries) / total)


def rce(
    displayed: Iterable[float],
    team_score: Iterable[float],
    *,
    n_bins: int = 10,
    scheme: str = "equal_width",
) -> float:
    """Alias for :func:`reliance_calibration_error`."""

    return reliance_calibration_error(
        displayed,
        team_score,
        n_bins=n_bins,
        scheme=scheme,
    )
