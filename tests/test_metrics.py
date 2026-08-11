"""Unit tests for Reliance Calibration Error."""

from __future__ import annotations

import unittest

import numpy as np

from calibrated_reliance.metrics import rce, reliability_bins


class MetricsTests(unittest.TestCase):
    """Check both binning schemes and fail-closed validation."""

    def test_perfect_calibration_is_zero_for_both_schemes(self) -> None:
        displayed = np.array([0.0, 0.0, 1.0, 1.0])
        team_correctness = displayed.copy()
        self.assertAlmostEqual(rce(displayed, team_correctness, n_bins=2), 0.0)
        self.assertAlmostEqual(
            rce(displayed, team_correctness, n_bins=2, scheme="equal_mass"),
            0.0,
        )

    def test_known_equal_width_and_equal_mass_value(self) -> None:
        displayed = np.array([0.1, 0.2, 0.8, 0.9])
        team_correctness = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(rce(displayed, team_correctness, n_bins=2), 0.15)
        self.assertAlmostEqual(
            rce(displayed, team_correctness, n_bins=2, scheme="equal_mass"),
            0.15,
        )

    def test_equal_mass_handles_constant_confidence(self) -> None:
        bins = reliability_bins(
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 1.0, 1.0],
            n_bins=10,
            scheme="equal_mass",
        )
        self.assertEqual(len(bins), 1)
        self.assertEqual(bins[0].count, 4)
        self.assertAlmostEqual(bins[0].absolute_gap, 0.25)

    def test_input_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            rce([0.2], [0.0, 1.0])
        with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
            rce([1.2], [1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            rce([np.nan], [1.0])
        with self.assertRaisesRegex(ValueError, "scheme"):
            rce([0.5], [1.0], scheme="unknown")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            rce([0.5], [1.0], n_bins=0)


if __name__ == "__main__":
    unittest.main()
