"""Unit tests for the four confidence-display policies."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from calibrated_reliance.policies import (
    SelfConfidenceIsotonicPolicy,
    apply_g2,
    bounded_g1,
    fit_g2,
    g0,
    g1,
    g3,
)


class PolicyTests(unittest.TestCase):
    """Verify paper-exact mappings and subgroup fallback behavior."""

    def setUp(self) -> None:
        self.confidence = np.tile(np.linspace(0.0, 1.0, 5), 3)
        self.self_confidence = np.repeat([0.1, 0.5, 0.9], 5)
        self.correctness = np.concatenate(
            [
                [0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1],
            ]
        )

    def test_g0_g1_and_g3_are_paper_exact(self) -> None:
        confidence = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(g0(confidence), confidence)
        np.testing.assert_allclose(g1(confidence), [0.0, 0.1, 0.7, 1.0, 1.0])
        np.testing.assert_allclose(g3(confidence), [0.15, 0.15, 0.7, 0.85, 0.85])
        self.assertAlmostEqual(g1(0.5), 0.7)

    def test_bounded_g1_enforces_requested_shift(self) -> None:
        confidence = np.array([0.25, 0.5, 0.75])
        displayed = bounded_g1(confidence, 0.1)
        np.testing.assert_allclose(displayed, [0.15, 0.6, 0.85])
        self.assertTrue(np.all(np.abs(displayed - confidence) <= 0.1 + 1e-12))

    def test_g2_uses_self_confidence_terciles(self) -> None:
        policy = fit_g2(
            self.confidence,
            self.self_confidence,
            self.correctness,
            min_group_size=5,
        )
        self.assertEqual(set(policy.models_), {"low", "mid", "high"})
        low_high = policy.predict([0.5, 0.5], [0.1, 0.9])
        self.assertLess(low_high[0], low_high[1])

    def test_sparse_missing_and_unknown_groups_use_global_fallback(self) -> None:
        policy = fit_g2(
            self.confidence,
            self.self_confidence,
            self.correctness,
            min_group_size=6,
        )
        self.assertIsNotNone(policy.global_model_)
        for model in policy.models_.values():
            self.assertIs(model, policy.global_model_)

        confidence = np.array([0.25, 0.75])
        expected = policy.global_model_.predict(confidence)
        actual = policy.predict_groups(confidence, [None, "unseen"])
        np.testing.assert_allclose(actual, expected)

    def test_dataframe_api_accepts_only_tercile_grouping(self) -> None:
        frame = pd.DataFrame(
            {
                "advice_prob": self.confidence,
                "prob_correct_1": self.self_confidence,
                "correct_post": self.correctness,
                "pre_conf_tercile": np.repeat(["low", "mid", "high"], 5),
                "task_name": np.repeat(["art", "cities", "sarcasm"], 5),
                "advice_source": np.tile(["ai", "human", "ai", "human", "ai"], 3),
            }
        )
        policy = fit_g2(frame, min_group_size=5)
        displayed = apply_g2(policy, frame)
        self.assertEqual(displayed.shape, (len(frame),))
        self.assertTrue(np.all((displayed >= 0.0) & (displayed <= 1.0)))

        invalid = frame.copy()
        invalid["pre_conf_tercile"] = invalid["task_name"] + "_" + invalid["advice_source"]
        with self.assertRaisesRegex(ValueError, "self-confidence terciles"):
            fit_g2(invalid, min_group_size=5)

    def test_predict_before_fit_fails(self) -> None:
        policy = SelfConfidenceIsotonicPolicy(min_group_size=1)
        with self.assertRaisesRegex(RuntimeError, "fitted"):
            policy.predict([0.5], [0.5])


if __name__ == "__main__":
    unittest.main()
