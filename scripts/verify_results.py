#!/usr/bin/env python3
"""Fail-closed verification of generated accepted-paper result tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED = ROOT / "results" / "generated"
DEFAULT_REFERENCE = ROOT / "results" / "reference" / "tables"
DEFAULT_ATOL = 1e-10
DEFAULT_RTOL = 1e-9

CORE_TABLES = (
    "tab_revision_main_policy_eval.csv",
    "tab_revision_task_user_state_sensitivity.csv",
    "tab_revision_support_coverage.csv",
    "tab_revision_clipping_shift.csv",
    "tab_revision_rce_binning_sensitivity.csv",
    "tab_revision_jackknife_rce.csv",
    "tab_revision_self_confidence_noise.csv",
    "tab_revision_subgroup_definition_sensitivity.csv",
    "tab_revision_private_information_decomposition.csv",
    "tab_revision_bounded_frontier.csv",
    "tab_revision_simple_interfaces.csv",
    "tab_revision_pilot_power.csv",
)
FULL_ONLY_TABLES = (
    "tab_revision_reliance_model_sensitivity.csv",
    "tab_revision_split_sensitivity.csv",
)


class VerificationError(RuntimeError):
    """Raised when a generated table differs from its accepted reference."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse verification profile, paths, and numerical tolerances."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare generated accepted-paper CSV tables with versioned references. "
            "Strings must match exactly; numeric values use explicit tolerances."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("core", "full"),
        default="core",
        help="Tables to verify (default: core).",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=DEFAULT_GENERATED,
        help=f"Generated-table directory (default: {DEFAULT_GENERATED}).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"Reference-table directory (default: {DEFAULT_REFERENCE}).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"Absolute tolerance for numeric columns (default: {DEFAULT_ATOL:g}).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"Relative tolerance for numeric columns (default: {DEFAULT_RTOL:g}).",
    )
    return parser.parse_args(argv)


def expected_tables(profile: str) -> tuple[str, ...]:
    """Return the complete table manifest for a verification profile."""

    return CORE_TABLES if profile == "core" else CORE_TABLES + FULL_ONLY_TABLES


def validate_tolerances(atol: float, rtol: float) -> None:
    """Reject negative or non-finite comparison tolerances."""

    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise VerificationError("--atol and --rtol must be finite, non-negative numbers.")


def compare_numeric_column(
    filename: str,
    column: str,
    generated: pd.Series,
    reference: pd.Series,
    atol: float,
    rtol: float,
) -> None:
    """Compare a numeric column with explicit absolute and relative tolerance."""

    generated_values = generated.to_numpy(dtype=float)
    reference_values = reference.to_numpy(dtype=float)
    matches = np.isclose(
        generated_values,
        reference_values,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )
    if bool(np.all(matches)):
        return

    first = int(np.flatnonzero(~matches)[0])
    abs_diff = np.abs(generated_values - reference_values)
    finite_diff = abs_diff[np.isfinite(abs_diff)]
    max_diff = float(np.max(finite_diff)) if finite_diff.size else float("nan")
    raise VerificationError(
        f"{filename}: numeric mismatch in column '{column}' at row {first}: "
        f"generated={generated_values[first]!r}, reference={reference_values[first]!r}, "
        f"max_abs_diff={max_diff!r}, atol={atol:g}, rtol={rtol:g}"
    )


def compare_exact_column(
    filename: str,
    column: str,
    generated: pd.Series,
    reference: pd.Series,
) -> None:
    """Compare a string, boolean, or other non-numeric column exactly."""

    generated_missing = generated.isna().to_numpy()
    reference_missing = reference.isna().to_numpy()
    missing_matches = generated_missing == reference_missing
    value_matches = (
        generated.astype("string").fillna("<NA>").to_numpy()
        == reference.astype("string").fillna("<NA>").to_numpy()
    )
    matches = missing_matches & value_matches
    if bool(np.all(matches)):
        return
    first = int(np.flatnonzero(~matches)[0])
    raise VerificationError(
        f"{filename}: exact mismatch in column '{column}' at row {first}: "
        f"generated={generated.iloc[first]!r}, reference={reference.iloc[first]!r}"
    )


def compare_table(
    generated_path: Path,
    reference_path: Path,
    atol: float,
    rtol: float,
) -> None:
    """Compare one generated CSV against its reference, column by column."""

    try:
        generated = pd.read_csv(generated_path)
    except Exception as exc:
        raise VerificationError(f"Could not read generated table {generated_path}: {exc}") from exc
    try:
        reference = pd.read_csv(reference_path)
    except Exception as exc:
        raise VerificationError(f"Could not read reference table {reference_path}: {exc}") from exc

    filename = generated_path.name
    if list(generated.columns) != list(reference.columns):
        raise VerificationError(
            f"{filename}: column schema/order mismatch. "
            f"generated={list(generated.columns)!r}, reference={list(reference.columns)!r}"
        )
    if len(generated) != len(reference):
        raise VerificationError(
            f"{filename}: row-count mismatch: "
            f"generated={len(generated)}, reference={len(reference)}"
        )

    for column in reference.columns:
        generated_column = generated[column]
        reference_column = reference[column]
        generated_numeric = is_numeric_dtype(generated_column.dtype)
        reference_numeric = is_numeric_dtype(reference_column.dtype)
        if generated_numeric != reference_numeric:
            raise VerificationError(
                f"{filename}: type-class mismatch in column '{column}': "
                f"generated={generated_column.dtype}, reference={reference_column.dtype}"
            )
        if generated_numeric:
            compare_numeric_column(
                filename,
                column,
                generated_column,
                reference_column,
                atol,
                rtol,
            )
        else:
            compare_exact_column(filename, column, generated_column, reference_column)


def verify(
    profile: str,
    generated_dir: Path,
    reference_dir: Path,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> tuple[str, ...]:
    """Verify all required tables and return the successfully checked manifest."""

    validate_tolerances(atol, rtol)
    generated_dir = generated_dir.expanduser().resolve()
    reference_dir = reference_dir.expanduser().resolve()
    manifest = expected_tables(profile)

    missing_generated = [
        str(generated_dir / filename)
        for filename in manifest
        if not (generated_dir / filename).is_file()
    ]
    missing_reference = [
        str(reference_dir / filename)
        for filename in manifest
        if not (reference_dir / filename).is_file()
    ]
    if missing_generated or missing_reference:
        messages = []
        if missing_generated:
            messages.append("Missing generated table(s):\n  " + "\n  ".join(missing_generated))
        if missing_reference:
            messages.append("Missing reference table(s):\n  " + "\n  ".join(missing_reference))
        raise VerificationError("\n".join(messages))

    for filename in manifest:
        compare_table(
            generated_dir / filename,
            reference_dir / filename,
            atol=atol,
            rtol=rtol,
        )
        print(f"PASS {filename}")
    print(
        f"Verified {len(manifest)} {profile} table(s) "
        f"(atol={atol:g}, rtol={rtol:g})."
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; return nonzero if any expected file or value fails."""

    args = parse_args(argv)
    try:
        verify(
            args.profile,
            args.generated,
            args.reference,
            atol=args.atol,
            rtol=args.rtol,
        )
    except VerificationError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
