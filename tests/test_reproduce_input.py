"""Negative input-validation tests for the public reproduction entry point."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "calibrated-reliance-test-matplotlib"),
)
SPEC = importlib.util.spec_from_file_location(
    "calibrated_reliance_reproduce",
    ROOT / "scripts" / "reproduce.py",
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError("Could not load scripts/reproduce.py for validation tests.")
REPRODUCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPRODUCE
SPEC.loader.exec_module(REPRODUCE)


def valid_input() -> pd.DataFrame:
    """Return the smallest structurally valid four-task HAIID-like frame."""

    tasks = ("art", "sarcasm", "cities", "census")
    return pd.DataFrame(
        {
            "task_name": tasks,
            "perceived_accuracy": [80] * 4,
            "response_1": [-0.4, 0.2, 0.5, -0.1],
            "response_2": [-0.2, 0.4, 0.7, 0.1],
            "advice": [0.8, -0.6, 0.7, 0.5],
            "advice_source": ["ai", "human", "ai", "human"],
            "participant_id": ["p1", "p2", "p3", "p4"],
            "task_instance_id": ["i1", "i2", "i3", "i4"],
        }
    )


class ReproductionInputTests(unittest.TestCase):
    """The CLI must fail closed before fitting models on malformed data."""

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            frame.to_csv(path, index=False)
            return REPRODUCE.prepare_data(path)

    def test_rejects_out_of_range_signed_response(self) -> None:
        frame = valid_input()
        frame.loc[0, "response_1"] = 1.2
        with self.assertRaisesRegex(ValueError, r"response_1.*\[-1, 1\]"):
            self.prepare(frame)

    def test_rejects_missing_main_task(self) -> None:
        frame = valid_input().loc[lambda value: value["task_name"] != "census"]
        with self.assertRaisesRegex(ValueError, "missing required main task.*census"):
            self.prepare(frame)

    def test_rejects_blank_critical_identifier(self) -> None:
        frame = valid_input()
        frame.loc[0, "participant_id"] = " "
        with self.assertRaisesRegex(ValueError, "participant_id.*missing or blank"):
            self.prepare(frame)

    def test_relative_reduction_sign_convention(self) -> None:
        self.assertAlmostEqual(REPRODUCE.relative_reduction_pct(2.0, 1.0), 50.0)
        self.assertAlmostEqual(REPRODUCE.relative_reduction_pct(1.0, 1.25), -25.0)
        with self.assertRaisesRegex(ValueError, "baseline must be positive"):
            REPRODUCE.relative_reduction_pct(0.0, 0.0)

    def test_task_user_state_sensitivity_has_canonical_long_schema(self) -> None:
        frame = pd.DataFrame(
            {
                "task_name": [
                    "art",
                    "art",
                    "sarcasm",
                    "sarcasm",
                    "cities",
                    "cities",
                    "census",
                    "census",
                ],
                "pre_conf_tercile": ["low", "mid", "high", "low"] * 2,
                "advice_prob": [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.1, 0.9],
                "correct_post": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        def mean_display(_pipe, _frame, displayed):
            return float(np.mean(displayed)), 0.0, 0.0

        with mock.patch.object(
            REPRODUCE, "team_mse_and_rce", side_effect=mean_display
        ), mock.patch.object(
            REPRODUCE,
            "rce_from_values",
            side_effect=lambda displayed, *_args, **_kwargs: float(np.mean(displayed)),
        ):
            table = REPRODUCE.task_user_state_sensitivity(object(), frame)

        self.assertEqual(
            list(table.columns),
            [
                "Stratum_type",
                "Stratum",
                "n",
                "Metric",
                "g0",
                "g1",
                "g1_relative_reduction_pct",
            ],
        )
        self.assertEqual(len(table), 11)
        self.assertEqual(
            table.groupby("Stratum_type")["Metric"].count().to_dict(),
            {"pre_conf_tercile": 3, "task": 8},
        )
        self.assertEqual(
            set(table.loc[table["Stratum_type"] == "pre_conf_tercile", "Metric"]),
            {"model_predicted_team_mse"},
        )


if __name__ == "__main__":
    unittest.main()
