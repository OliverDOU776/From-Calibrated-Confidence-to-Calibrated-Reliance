"""Unit tests for HAIID loading and feature derivation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from calibrated_reliance.data import (
    MAIN_TASKS,
    derive_haiid_features,
    load_haiid,
    prepare_haiid,
    validate_haiid,
)


def raw_rows() -> pd.DataFrame:
    """Create a tiny raw HAIID-like frame with filter controls."""

    return pd.DataFrame(
        {
            "task_name": ["art", "sarcasm", "cities", "census", "art", "dermatology"],
            "task_instance_id": [f"item-{index}" for index in range(6)],
            "advice_source": ["ai", "human", "ai", "human", "ai", "ai"],
            "advice": [-0.8, 0.8, 0.6, -0.4, 0.5, 0.2],
            "participant_id": [f"person-{index // 2}" for index in range(6)],
            "response_1": [-0.6, 0.0, 0.4, 0.9, -0.2, 0.1],
            "response_2": [-0.7, 0.4, 0.5, 0.3, 0.1, 0.2],
            "perceived_accuracy": [80, 80, 80, 80, 65, 80],
        }
    )


class DataTests(unittest.TestCase):
    """Exercise validation, filtering, and deterministic derivations."""

    def test_prepare_filters_to_four_tasks_and_pa80(self) -> None:
        prepared = prepare_haiid(raw_rows())
        self.assertEqual(len(prepared), 4)
        self.assertEqual(set(prepared["task_name"]), set(MAIN_TASKS))
        self.assertTrue((prepared["perceived_accuracy"] == 80).all())

    def test_derived_features_match_signed_correctness_encoding(self) -> None:
        prepared = derive_haiid_features(raw_rows().iloc[:4])
        np.testing.assert_allclose(
            prepared["prob_correct_1"],
            (prepared["response_1"] + 1.0) / 2.0,
        )
        np.testing.assert_allclose(
            prepared["advice_prob"],
            (prepared["advice"] + 1.0) / 2.0,
        )
        np.testing.assert_array_equal(
            prepared["correct_post"],
            (prepared["response_2"] > 0.0).astype(int),
        )
        np.testing.assert_array_equal(
            prepared["shifted_toward"],
            np.array([1, 1, 1, 1]),
        )
        self.assertEqual(
            set(prepared["pre_conf_tercile"].astype(str)),
            {"low", "mid", "high"},
        )

    def test_load_accepts_file_or_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "haiid_dataset.csv"
            raw_rows().to_csv(source, index=False)
            from_file = load_haiid(source)
            from_directory = load_haiid(directory)
        pd.testing.assert_frame_equal(from_file, from_directory)

    def test_validation_rejects_missing_and_out_of_range_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_haiid(raw_rows().drop(columns="response_2"))

        invalid = raw_rows()
        invalid.loc[0, "advice"] = 1.1
        with self.assertRaisesRegex(ValueError, r"within \[-1, 1\]"):
            validate_haiid(invalid)

    def test_empty_filtered_cohort_is_an_error(self) -> None:
        frame = raw_rows()
        frame["perceived_accuracy"] = 65
        with self.assertRaisesRegex(ValueError, "No HAIID rows remain"):
            prepare_haiid(frame)


if __name__ == "__main__":
    unittest.main()
