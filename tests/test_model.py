"""Unit tests for the accepted-paper baseline reliance model."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from calibrated_reliance.data import prepare_haiid
from calibrated_reliance.model import (
    build_feature_frame,
    counterfactual_response,
    evaluate_policy,
    fit_reliance_model,
    team_mse_and_rce,
)
from calibrated_reliance.policies import g0, g3


def synthetic_haiid(n_rows: int = 240, seed: int = 17) -> pd.DataFrame:
    """Generate fully synthetic interactions with both reliance classes."""

    rng = np.random.default_rng(seed)
    tasks = np.resize(np.array(["art", "sarcasm", "cities", "census"]), n_rows)
    sources = np.resize(np.array(["ai", "human"]), n_rows)
    response_1 = rng.uniform(-0.95, 0.95, n_rows)
    advice = rng.uniform(-0.95, 0.95, n_rows)
    advice_prob = (advice + 1.0) / 2.0
    self_confidence = (response_1 + 1.0) / 2.0
    logit = -0.7 + 1.8 * advice_prob - 1.2 * self_confidence + 0.4 * (sources == "ai")
    probability = 1.0 / (1.0 + np.exp(-logit))
    shifted = rng.random(n_rows) < probability
    fraction = rng.uniform(0.2, 0.9, n_rows)
    response_2 = np.where(
        shifted,
        response_1 + fraction * (advice - response_1),
        response_1,
    )
    return pd.DataFrame(
        {
            "task_name": tasks,
            "task_instance_id": [f"item-{index % 40}" for index in range(n_rows)],
            "advice_source": sources,
            "advice": advice,
            "participant_id": [f"participant-{index % 30}" for index in range(n_rows)],
            "response_1": response_1,
            "response_2": response_2,
            "perceived_accuracy": np.full(n_rows, 80),
        }
    )


class ModelTests(unittest.TestCase):
    """Fit and evaluate the complete pipeline without external data."""

    @classmethod
    def setUpClass(cls) -> None:
        prepared = prepare_haiid(synthetic_haiid())
        cls.train = prepared.iloc[:180].copy()
        cls.validation = prepared.iloc[180:].copy()
        cls.model = fit_reliance_model(cls.train)

    def test_feature_frame_contains_accepted_baseline_predictors(self) -> None:
        displayed = np.asarray(g0(self.validation["advice_prob"]))
        features = build_feature_frame(self.validation, displayed)
        self.assertEqual(
            list(features.columns),
            [
                "displayed",
                "prob_correct_1",
                "is_ai",
                "abs_gap_pre_advice",
                "display_x_self",
                "task_name",
                "advice_source",
            ],
        )
        np.testing.assert_allclose(
            features["display_x_self"],
            displayed * self.validation["prob_correct_1"].to_numpy(),
        )

    def test_counterfactual_response_matches_paper_equation(self) -> None:
        displayed = np.asarray(g3(self.validation["advice_prob"]))
        response, probability = counterfactual_response(
            self.model,
            self.validation,
            displayed,
        )
        expected = self.validation["response_1"].to_numpy() + probability * (
            self.validation["advice"].to_numpy() - self.validation["response_1"].to_numpy()
        )
        np.testing.assert_allclose(response, expected)
        self.assertTrue(np.all((probability >= 0.0) & (probability <= 1.0)))

    def test_mse_and_rce_are_consistent_across_apis(self) -> None:
        displayed = np.asarray(g0(self.validation["advice_prob"]))
        mse, calibration, mean_reliance = team_mse_and_rce(
            self.model,
            self.validation,
            displayed,
        )
        evaluation = evaluate_policy(self.model, self.validation, displayed)
        self.assertAlmostEqual(mse, evaluation.mse)
        self.assertAlmostEqual(calibration, evaluation.rce)
        self.assertAlmostEqual(mean_reliance, evaluation.mean_reliance)
        self.assertTrue(np.isfinite(mse))
        self.assertGreaterEqual(calibration, 0.0)
        self.assertLessEqual(calibration, 1.0)

    def test_invalid_counterfactual_display_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
            counterfactual_response(
                self.model,
                self.validation,
                np.full(len(self.validation), 1.1),
            )


if __name__ == "__main__":
    unittest.main()
