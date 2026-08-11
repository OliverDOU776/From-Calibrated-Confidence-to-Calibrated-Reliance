#!/usr/bin/env python3
"""Reproduce the accepted-paper HAIID diagnostics.

The implementation is the public-release counterpart of the canonical
``scripts/revision_diagnostics.py`` analysis.  It intentionally writes only to
the requested generated-output directory and never mutates manuscript assets.

``core`` generates the primary policy, support, observed plug-in RCE,
task/user-state sensitivity, subgroup, bounded-policy, interface,
private-information, and planning diagnostics.  ``full`` additionally fits the
slower reliance-model sensitivity suite and reruns the analysis under
alternative train/validation splits.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "HAIID" / "haiid_dataset.csv"
DEFAULT_OUTPUT = ROOT / "results" / "generated"
MAIN_TASKS = ["art", "sarcasm", "cities", "census"]
RNG = 42

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
DISPLAY_SHIFT_FIGURE = "fig_revision_display_shift.png"

REQUIRED_COLUMNS = {
    "task_name",
    "perceived_accuracy",
    "response_1",
    "response_2",
    "advice",
    "advice_source",
    "participant_id",
    "task_instance_id",
}

NUMERIC_FEATURES = [
    "displayed",
    "prob_correct_1",
    "is_ai",
    "abs_gap_pre_advice",
    "display_x_self",
]
CATEGORICAL_FEATURES = ["task_name", "advice_source"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the deterministic reproduction run."""

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce accepted-paper HAIID diagnostics. The core profile writes "
            "12 tables plus the display-shift figure; full adds model and split sensitivity."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("core", "full"),
        default="core",
        help="Reproduction scope (default: core).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"HAIID CSV path (default: {DEFAULT_DATA}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for generated tables and figure (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


def is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether *path* is equal to or contained in *parent*."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_paths(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Resolve inputs and reject missing data or protected output locations."""

    data_path = data_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"HAIID data file not found: {data_path}. "
            "Download it first or pass --data /path/to/haiid_dataset.csv."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")

    protected = (
        ROOT / "paper_assets",
        ROOT / "paper",
        ROOT / "results" / "reference",
    )
    for protected_dir in protected:
        if is_relative_to(output_dir, protected_dir.resolve()):
            raise ValueError(
                f"Refusing to write generated results inside protected directory: {protected_dir}"
            )
    return data_path, output_dir


def prepare_data(data_path: Path) -> pd.DataFrame:
    """Load HAIID and construct the accepted-paper analysis variables."""

    df = pd.read_csv(data_path, dtype={"job_title": str})
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"HAIID input is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("HAIID input contains no rows.")

    numeric_columns = ("response_1", "response_2", "advice", "perceived_accuracy")
    for column in numeric_columns:
        try:
            values = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"HAIID column '{column}' must be numeric.") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"HAIID column '{column}' must contain only finite values.")
        df[column] = values

    for column in ("response_1", "response_2", "advice"):
        values = df[column].to_numpy(dtype=float)
        if np.any((values < -1.0) | (values > 1.0)):
            raise ValueError(f"HAIID column '{column}' must lie within [-1, 1].")
    perceived_accuracy = df["perceived_accuracy"].to_numpy(dtype=float)
    if np.any((perceived_accuracy < 0.0) | (perceived_accuracy > 100.0)):
        raise ValueError("HAIID column 'perceived_accuracy' must lie within [0, 100].")

    for column in ("task_name", "advice_source", "participant_id", "task_instance_id"):
        if df[column].isna().any() or (df[column].astype(str).str.strip() == "").any():
            raise ValueError(f"HAIID column '{column}' must not contain missing or blank values.")

    df["task_name"] = df["task_name"].astype(str).str.strip().str.lower()
    df["advice_source"] = df["advice_source"].astype(str).str.strip().str.lower()
    unknown_sources = sorted(set(df["advice_source"]).difference({"ai", "human"}))
    if unknown_sources:
        raise ValueError(
            "HAIID column 'advice_source' must contain only 'ai' or 'human'; "
            f"observed {unknown_sources}."
        )

    df = df[(df["task_name"].isin(MAIN_TASKS)) & (df["perceived_accuracy"] == 80)].copy()
    if df.empty:
        raise ValueError(
            "HAIID input contains no rows for the four main tasks at perceived_accuracy=80."
        )
    observed_tasks = set(df["task_name"].unique())
    missing_tasks = sorted(set(MAIN_TASKS).difference(observed_tasks))
    if missing_tasks:
        raise ValueError(
            "HAIID input is missing required main task(s) at perceived_accuracy=80: "
            + ", ".join(missing_tasks)
        )
    df["prob_correct_1"] = (df["response_1"] + 1.0) / 2.0
    df["advice_prob"] = (df["advice"] + 1.0) / 2.0
    df["correct_pre"] = (df["response_1"] > 0).astype(int)
    df["correct_post"] = (df["response_2"] > 0).astype(int)
    df["ai_correct"] = (df["advice"] > 0).astype(int)
    df["is_ai"] = (df["advice_source"] == "ai").astype(int)
    df["shifted_toward"] = (
        (df["response_2"] - df["response_1"]) * (df["advice"] - df["response_1"]) > 0
    ).astype(int)
    df["abs_gap_pre_advice"] = (df["advice"] - df["response_1"]).abs()
    df["disagreement"] = (np.sign(df["response_1"]) != np.sign(df["advice"])).astype(int)

    # g2 is intentionally based on pre-advice self-confidence only.  It is not
    # the obsolete task-by-source subgroup definition from earlier drafts.
    df["pre_conf_tercile"] = pd.qcut(
        df["prob_correct_1"], q=3, labels=["low", "mid", "high"], duplicates="drop"
    ).astype(str)
    df["pre_conf_median"] = pd.qcut(
        df["prob_correct_1"], q=2, labels=["low", "high"], duplicates="drop"
    ).astype(str)
    df["pre_conf_quartile"] = pd.qcut(
        df["prob_correct_1"], q=4, labels=["q1", "q2", "q3", "q4"], duplicates="drop"
    ).astype(str)
    df["pre_conf_raw5"] = pd.cut(
        df["prob_correct_1"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
        labels=["1", "2", "3", "4", "5"],
    ).astype(str)
    return df


def feature_frame(df: pd.DataFrame, displayed: np.ndarray) -> pd.DataFrame:
    """Construct reliance-model features for a candidate displayed confidence."""

    return pd.DataFrame(
        {
            "displayed": displayed,
            "prob_correct_1": df["prob_correct_1"].to_numpy(),
            "is_ai": df["is_ai"].to_numpy(),
            "abs_gap_pre_advice": df["abs_gap_pre_advice"].to_numpy(),
            "display_x_self": displayed * df["prob_correct_1"].to_numpy(),
            "task_name": df["task_name"].to_numpy(),
            "advice_source": df["advice_source"].to_numpy(),
            "participant_id": df["participant_id"].to_numpy(),
            "task_instance_id": df["task_instance_id"].to_numpy(),
        }
    )


@dataclass
class RelianceSpec:
    """A named reliance estimator and any additional categorical effects."""

    name: str
    estimator: object
    extra_categories: tuple[str, ...] = ()


def make_pipeline(spec: RelianceSpec) -> Pipeline:
    """Build the preprocessing and classifier pipeline for one reliance model."""

    categorical = list(CATEGORICAL_FEATURES) + list(spec.extra_categories)
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
        ]
    )
    return Pipeline([("pre", pre), ("model", spec.estimator)])


def fit_reliance(spec: RelianceSpec, train_df: pd.DataFrame) -> Pipeline:
    """Fit a reliance model on observed displayed advice confidence."""

    pipe = make_pipeline(spec)
    x_train = feature_frame(train_df, train_df["advice_prob"].to_numpy())
    pipe.fit(x_train, train_df["shifted_toward"].to_numpy())
    return pipe


def predict_shift(pipe: Pipeline, df: pd.DataFrame, displayed: np.ndarray) -> np.ndarray:
    """Predict movement toward advice under a candidate display policy."""

    x = feature_frame(df, np.clip(displayed, 0, 1))
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(x)[:, 1]
    transformed = pipe.named_steps["pre"].transform(x)
    return pipe.named_steps["model"].predict_proba(transformed)[:, 1]


def counterfactual_response(
    pipe: Pipeline, df: pd.DataFrame, displayed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the post-advice response and reliance probability off policy."""

    p_shift = predict_shift(pipe, df, displayed)
    r1 = df["response_1"].to_numpy()
    advice = df["advice"].to_numpy()
    return r1 + p_shift * (advice - r1), p_shift


def rce_from_values(
    displayed: np.ndarray,
    team_score: np.ndarray,
    bins: int = 10,
    scheme: str = "equal_width",
) -> float:
    """Compute binned RCE from displayed confidence and an outcome score."""

    displayed = np.clip(np.asarray(displayed), 0, 1)
    team_score = np.asarray(team_score)
    if scheme == "equal_width":
        idx = np.clip((displayed * bins).astype(int), 0, bins - 1)
    elif scheme == "equal_mass":
        try:
            idx = pd.qcut(displayed, q=bins, labels=False, duplicates="drop")
            idx = np.asarray(idx, dtype=float)
        except ValueError:
            return float("nan")
    else:
        raise ValueError(f"Unknown RCE binning scheme: {scheme}")

    rce = 0.0
    total = 0
    for bin_id in sorted(pd.Series(idx).dropna().unique()):
        mask = idx == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        rce += count * abs(team_score[mask].mean() - displayed[mask].mean())
        total += count
    return rce / total if total else float("nan")


def team_mse_and_rce(
    pipe: Pipeline, df: pd.DataFrame, displayed: np.ndarray, bins: int = 10
) -> tuple[float, float, float]:
    """Return off-policy model-based team MSE, RCE, and mean reliance."""

    response_2, p_shift = counterfactual_response(pipe, df, displayed)
    mse = float(np.mean((response_2 - 1.0) ** 2))
    team_score = np.clip((response_2 + 1.0) / 2.0, 0, 1)
    rce = rce_from_values(displayed, team_score, bins=bins, scheme="equal_width")
    return mse, rce, float(np.mean(p_shift))


def g0(df: pd.DataFrame) -> np.ndarray:
    """Direct-display baseline."""

    return df["advice_prob"].to_numpy()


def g1(df: pd.DataFrame) -> np.ndarray:
    """Global human-aware affine remapping."""

    return np.clip(2.4 * df["advice_prob"].to_numpy() - 0.5, 0, 1)


def g3(df: pd.DataFrame) -> np.ndarray:
    """Bounded global mapping with the accepted-paper guardrail."""

    return np.clip(g1(df), 0.15, 0.85)


def bounded_g1(df: pd.DataFrame, delta: float) -> np.ndarray:
    """Bound g1 both around raw confidence and away from 0/1 extremes."""

    confidence = df["advice_prob"].to_numpy()
    return np.clip(np.clip(g1(df), confidence - delta, confidence + delta), 0.05, 0.95)


def train_g2(train_df: pd.DataFrame, group_col: str = "pre_conf_tercile") -> tuple:
    """Fit subgroup isotonic maps using pre-advice self-confidence groups."""

    models: dict[str, IsotonicRegression] = {}
    global_model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    global_model.fit(train_df["advice_prob"], train_df["correct_post"])
    for group, subgroup in train_df.groupby(group_col):
        if len(subgroup) < 100 or subgroup["advice_prob"].nunique() < 3:
            models[str(group)] = global_model
            continue
        model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        model.fit(subgroup["advice_prob"], subgroup["correct_post"])
        models[str(group)] = model
    return models, global_model, group_col


def apply_g2(
    df: pd.DataFrame,
    g2_bundle: tuple,
    noise: str | None = None,
    seed: int = RNG,
) -> np.ndarray:
    """Apply g2, optionally perturbing subgroup reports for sensitivity tests."""

    models, global_model, group_col = g2_bundle
    rng = np.random.default_rng(seed)
    groups = df[group_col].astype(str).to_numpy().copy()
    if noise is not None:
        unique = np.array(sorted(pd.Series(groups).dropna().unique()))
        if noise.startswith("flip"):
            fraction = float(noise.replace("flip", ""))
            mask = rng.random(len(groups)) < fraction
            groups[mask] = rng.choice(unique, size=mask.sum())
        elif noise == "all_low":
            groups[:] = unique[0]
        elif noise == "all_high":
            groups[:] = unique[-1]
        elif noise == "random":
            groups[:] = rng.choice(unique, size=len(groups))
        else:
            raise ValueError(f"Unknown g2 noise scenario: {noise}")
    confidence = df["advice_prob"].to_numpy()
    displayed = np.zeros(len(df))
    for group in np.unique(groups):
        mask = groups == group
        model = models.get(str(group), global_model)
        displayed[mask] = model.predict(confidence[mask])
    return np.clip(displayed, 0, 1)


def eval_policy_table(
    pipe: Pipeline, df: pd.DataFrame, policies: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Evaluate policies using explicitly model-based off-policy outcomes."""

    rows = []
    for name, displayed in policies.items():
        mse, rce, reliance = team_mse_and_rce(pipe, df, displayed)
        rows.append(
            {
                "Policy": name,
                "Team_MSE": mse,
                "RCE_model_based": rce,
                "Mean_predicted_reliance": reliance,
                "Mean_display": float(np.mean(displayed)),
                "Display_sd": float(np.std(displayed)),
            }
        )
    return pd.DataFrame(rows)


def relative_reduction_pct(baseline: float, candidate: float) -> float:
    """Return the percentage reduction from a positive lower-is-better baseline.

    Positive values mean that the candidate has lower error than the baseline;
    negative values mean that it is worse.  The explicit sign convention keeps
    MSE and RCE comparisons interpretable in the same long-format table.
    """

    if not np.isfinite(baseline) or not np.isfinite(candidate):
        raise ValueError("Relative-reduction inputs must be finite.")
    if baseline <= 0:
        raise ValueError("Relative-reduction baseline must be positive.")
    return 100.0 * (baseline - candidate) / baseline


def task_user_state_sensitivity(
    pipe: Pipeline,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare g1 with g0 by task and pre-advice self-confidence tercile.

    Model-predicted team MSE uses the fitted baseline reliance model.  The
    task-level RCE rows instead use observed post-advice correctness as the
    plug-in outcome with the canonical 10 equal-width confidence bins.  This
    distinction is retained in the metric names rather than blending the two
    estimands.
    """

    rows: list[dict[str, object]] = []
    stratum_specs = (
        ("task", "task_name", MAIN_TASKS, True),
        ("pre_conf_tercile", "pre_conf_tercile", ("low", "mid", "high"), False),
    )
    for stratum_type, column, levels, include_plugin_rce in stratum_specs:
        for level in levels:
            subgroup = val_df.loc[val_df[column] == level].copy()
            if subgroup.empty:
                raise ValueError(
                    f"Canonical sensitivity stratum is empty: {stratum_type}={level}"
                )
            direct = g0(subgroup)
            remapped = g1(subgroup)
            g0_mse = team_mse_and_rce(pipe, subgroup, direct)[0]
            g1_mse = team_mse_and_rce(pipe, subgroup, remapped)[0]
            rows.append(
                {
                    "Stratum_type": stratum_type,
                    "Stratum": level,
                    "n": len(subgroup),
                    "Metric": "model_predicted_team_mse",
                    "g0": g0_mse,
                    "g1": g1_mse,
                    "g1_relative_reduction_pct": relative_reduction_pct(g0_mse, g1_mse),
                }
            )

            if include_plugin_rce:
                observed = subgroup["correct_post"].to_numpy()
                g0_rce = rce_from_values(
                    direct, observed, bins=10, scheme="equal_width"
                )
                g1_rce = rce_from_values(
                    remapped, observed, bins=10, scheme="equal_width"
                )
                rows.append(
                    {
                        "Stratum_type": stratum_type,
                        "Stratum": level,
                        "n": len(subgroup),
                        "Metric": "plugin_observed_outcome_rce",
                        "g0": g0_rce,
                        "g1": g1_rce,
                        "g1_relative_reduction_pct": relative_reduction_pct(
                            g0_rce, g1_rce
                        ),
                    }
                )
    return pd.DataFrame(rows)


def support_and_shift(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    policies: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure empirical display support and policy-induced display shifts."""

    observed = train_df["advice_prob"].to_numpy()
    edges = np.linspace(0, 1, 21)
    observed_counts, _ = np.histogram(observed, bins=edges)
    sparse_cut = max(50, math.ceil(0.01 * len(train_df)))
    well_bins = observed_counts >= sparse_cut
    sparse_bins = (observed_counts > 0) & (observed_counts < sparse_cut)

    support_rows = []
    shift_rows = []
    raw_validation = val_df["advice_prob"].to_numpy()
    for name, displayed in policies.items():
        displayed = np.clip(displayed, 0, 1)
        idx = np.clip(
            np.digitize(displayed, edges, right=False) - 1,
            0,
            len(observed_counts) - 1,
        )
        nearest = np.min(np.abs(displayed[:, None] - observed[None, :]), axis=1)
        support_rows.append(
            {
                "Policy": name,
                "well_supported_pct": 100 * float(np.mean(well_bins[idx])),
                "sparse_pct": 100 * float(np.mean(sparse_bins[idx])),
                "outside_practical_support_pct": 100
                * float(np.mean(observed_counts[idx] == 0)),
                "mean_nearest_neighbor_distance": float(np.mean(nearest)),
            }
        )
        shift_rows.append(
            {
                "Policy": name,
                "mean_display": float(np.mean(displayed)),
                "sd_display": float(np.std(displayed)),
                "mean_abs_shift_from_c": float(np.mean(np.abs(displayed - raw_validation))),
                "p05_display": float(np.quantile(displayed, 0.05)),
                "p50_display": float(np.quantile(displayed, 0.50)),
                "p95_display": float(np.quantile(displayed, 0.95)),
                "pct_at_0": 100 * float(np.mean(np.isclose(displayed, 0))),
                "pct_at_1": 100 * float(np.mean(np.isclose(displayed, 1))),
                "pct_at_0_or_1": 100
                * float(np.mean(np.isclose(displayed, 0) | np.isclose(displayed, 1))),
                "pct_at_015_or_085": 100
                * float(np.mean(np.isclose(displayed, 0.15) | np.isclose(displayed, 0.85))),
            }
        )
    return pd.DataFrame(support_rows), pd.DataFrame(shift_rows)


def plot_display_shift(
    val_df: pd.DataFrame,
    policies: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Plot raw and policy-displayed confidence distributions."""

    raw = val_df["advice_prob"].to_numpy()
    bins = np.linspace(0, 1, 21)
    colors = {"g0": "#4D4D4D", "g1": "#D55E00", "g2": "#0072B2", "g3": "#009E73"}
    labels = {
        "g0": "Direct display",
        "g1": "Global remap",
        "g2": "Subgroup remap",
        "g3": "Bounded global",
    }

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), sharex=True, sharey=True)
    for axis, name in zip(axes.flat, ["g0", "g1", "g2", "g3"]):
        displayed = policies.get(name, raw)
        raw_weights = np.ones_like(raw, dtype=float) * 100.0 / len(raw)
        display_weights = np.ones_like(displayed, dtype=float) * 100.0 / len(displayed)
        axis.hist(raw, bins=bins, weights=raw_weights, color="#D0D0D0", alpha=0.8, label="raw c")
        axis.hist(
            displayed,
            bins=bins,
            weights=display_weights,
            histtype="step",
            linewidth=2.4,
            color=colors[name],
            label="displayed d",
        )
        axis.set_title(f"{name}: {labels[name]}", fontsize=10)
        axis.set_xlim(0, 1)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)

        boundary = np.mean((displayed <= 1e-12) | (displayed >= 1 - 1e-12)) * 100.0
        annotation = f"0/1 boundary: {boundary:.1f}%"
        if name == "g3":
            guard = (
                np.mean(np.isclose(displayed, 0.15) | np.isclose(displayed, 0.85)) * 100.0
            )
            annotation += f"\nguard: {guard:.1f}%"
            axis.axvline(0.15, color="#666666", linestyle=":", linewidth=1)
            axis.axvline(0.85, color="#666666", linestyle=":", linewidth=1)
        axis.text(
            0.03,
            0.95,
            annotation,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#BDBDBD",
                "alpha": 0.92,
            },
        )

    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(0.5, 0.03, "Confidence value", ha="center")
    fig.text(0.03, 0.5, "Validation cases (%)", va="center", rotation="vertical")
    fig.tight_layout(rect=(0.05, 0.06, 1.0, 1.0))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def model_specs() -> list[RelianceSpec]:
    """Return accepted-paper reliance-model sensitivity specifications."""

    return [
        RelianceSpec(
            "Logistic baseline",
            LogisticRegression(max_iter=1500, random_state=RNG),
        ),
        RelianceSpec(
            "Logistic + participant intercepts",
            LogisticRegression(max_iter=1500, C=1.0, random_state=RNG, solver="lbfgs"),
            ("participant_id",),
        ),
        RelianceSpec(
            "Logistic + item intercepts",
            LogisticRegression(max_iter=1500, C=1.0, random_state=RNG, solver="lbfgs"),
            ("task_instance_id",),
        ),
        RelianceSpec(
            "Logistic + participant + item intercepts",
            LogisticRegression(max_iter=1500, C=1.0, random_state=RNG, solver="lbfgs"),
            ("participant_id", "task_instance_id"),
        ),
        RelianceSpec(
            "Random forest",
            RandomForestClassifier(
                n_estimators=240,
                max_depth=10,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=RNG,
            ),
        ),
        RelianceSpec(
            "Gradient boosted trees",
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.04,
                max_leaf_nodes=24,
                random_state=RNG,
            ),
        ),
    ]


def reliance_model_sensitivity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    policies: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Evaluate off-policy conclusions across reliance-model specifications."""

    rows = []
    for spec in model_specs():
        pipe = fit_reliance(spec, train_df)
        y_val = val_df["shifted_toward"].to_numpy()
        p_val = predict_shift(pipe, val_df, val_df["advice_prob"].to_numpy())
        metrics: dict[str, object] = {
            "Reliance_model": spec.name,
            "AUC": roc_auc_score(y_val, p_val),
            "Brier": brier_score_loss(y_val, p_val),
            "Log_loss": log_loss(y_val, np.clip(p_val, 1e-6, 1 - 1e-6)),
        }
        for policy_name, displayed in policies.items():
            mse, rce, _ = team_mse_and_rce(pipe, val_df, displayed)
            metrics[f"{policy_name}_MSE"] = mse
            metrics[f"{policy_name}_RCE"] = rce
        ordered = sorted(policies, key=lambda key: metrics[f"{key}_RCE"])
        metrics["RCE_ranking"] = " < ".join(ordered)
        metrics["g2_beats_g0"] = metrics["g2_RCE"] < metrics["g0_RCE"]
        metrics["g3_beats_g0"] = metrics["g3_RCE"] < metrics["g0_RCE"]
        rows.append(metrics)
    return pd.DataFrame(rows)


def split_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Rerun the model-based policy comparison under three split definitions."""

    rows = []
    split_defs = [
        ("random stratified", None),
        ("participant-disjoint", "participant_id"),
        ("item-disjoint", "task_instance_id"),
    ]
    base_spec = model_specs()[0]
    for split_name, group_col in split_defs:
        if group_col is None:
            train_idx, val_idx = train_test_split(
                df.index,
                test_size=0.3,
                random_state=RNG,
                stratify=df["task_name"],
            )
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RNG)
            train_pos, val_pos = next(splitter.split(df, groups=df[group_col]))
            train_idx, val_idx = df.index[train_pos], df.index[val_pos]
        train_df, val_df = df.loc[train_idx].copy(), df.loc[val_idx].copy()
        pipe = fit_reliance(base_spec, train_df)
        local_g2 = train_g2(train_df)
        policies = {
            "g0": g0(val_df),
            "g1": g1(val_df),
            "g2": apply_g2(val_df, local_g2),
            "g3": g3(val_df),
        }
        metrics: dict[str, object] = {
            "Split": split_name,
            "n_train": len(train_df),
            "n_val": len(val_df),
        }
        for name, displayed in policies.items():
            mse, rce, _ = team_mse_and_rce(pipe, val_df, displayed)
            metrics[f"{name}_MSE"] = mse
            metrics[f"{name}_RCE"] = rce
        metrics["RCE_ranking"] = " < ".join(
            sorted(policies, key=lambda key: metrics[f"{key}_RCE"])
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def rce_binning_sensitivity(
    val_df: pd.DataFrame, policies: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Compute observed-outcome plug-in RCE across binning choices."""

    rows = []
    observed = val_df["correct_post"].to_numpy()
    for scheme in ["equal_width", "equal_mass"]:
        for bins in [5, 10, 15, 20]:
            row: dict[str, object] = {"Binning": scheme, "Bins": bins}
            for name, displayed in policies.items():
                row[f"{name}_RCE_observed"] = rce_from_values(
                    displayed, observed, bins=bins, scheme=scheme
                )
            row["Ranking"] = " < ".join(
                sorted(policies, key=lambda key: row[f"{key}_RCE_observed"])
            )
            rows.append(row)
    return pd.DataFrame(rows)


def jackknife_rce(
    val_df: pd.DataFrame, policies: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Bias-correct observed plug-in RCE with participant jackknifing."""

    rows = []
    observed = val_df["correct_post"].to_numpy()
    groups = val_df["participant_id"].to_numpy()
    unique_groups = np.unique(groups)
    for name, displayed in policies.items():
        plugin = rce_from_values(displayed, observed, bins=10, scheme="equal_width")
        leave = []
        for group in unique_groups:
            mask = groups != group
            leave.append(
                rce_from_values(
                    displayed[mask], observed[mask], bins=10, scheme="equal_width"
                )
            )
        leave_values = np.asarray(leave)
        jack_mean = leave_values.mean()
        corrected = len(unique_groups) * plugin - (len(unique_groups) - 1) * jack_mean
        standard_error = math.sqrt(
            (len(unique_groups) - 1)
            / len(unique_groups)
            * np.sum((leave_values - jack_mean) ** 2)
        )
        rows.append(
            {
                "Policy": name,
                "plugin_RCE": plugin,
                "participant_jackknife_RCE": corrected,
                "jackknife_SE": standard_error,
            }
        )
    return pd.DataFrame(rows)


def subgroup_noise_and_gaming(
    pipe: Pipeline, val_df: pd.DataFrame, g2_bundle: tuple
) -> pd.DataFrame:
    """Stress-test the self-confidence-tercile g2 policy under reporting noise."""

    scenarios = [
        ("observed", None),
        ("10pct random subgroup noise", "flip0.10"),
        ("20pct random subgroup noise", "flip0.20"),
        ("30pct random subgroup noise", "flip0.30"),
        ("complete randomization", "random"),
        ("all report low", "all_low"),
        ("all report high", "all_high"),
    ]
    baseline_mse, baseline_rce, _ = team_mse_and_rce(pipe, val_df, g0(val_df))
    rows = []
    for label, noise in scenarios:
        displayed = apply_g2(val_df, g2_bundle, noise=noise)
        mse, rce, _ = team_mse_and_rce(pipe, val_df, displayed)
        rows.append(
            {
                "Scenario": label,
                "g2_MSE": mse,
                "g2_RCE": rce,
                "RCE_vs_g0": rce - baseline_rce,
                "MSE_vs_g0": mse - baseline_mse,
                "g2_better_RCE_than_g0": rce < baseline_rce,
            }
        )
    return pd.DataFrame(rows)


def subgroup_definition_sensitivity(
    pipe: Pipeline, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> pd.DataFrame:
    """Compare alternative pre-advice self-confidence discretizations for g2."""

    rows = []
    baseline_mse, baseline_rce, _ = team_mse_and_rce(pipe, val_df, g0(val_df))
    for group_col in [
        "pre_conf_tercile",
        "pre_conf_median",
        "pre_conf_quartile",
        "pre_conf_raw5",
    ]:
        bundle = train_g2(train_df, group_col=group_col)
        displayed = apply_g2(val_df, bundle)
        mse, rce, _ = team_mse_and_rce(pipe, val_df, displayed)
        rows.append(
            {
                "Subgroup_definition": group_col,
                "n_groups": train_df[group_col].nunique(),
                "g2_MSE": mse,
                "g2_RCE": rce,
                "RCE_vs_g0": rce - baseline_rce,
                "MSE_vs_g0": mse - baseline_mse,
            }
        )
    return pd.DataFrame(rows)


def private_information_decomposition(
    pipe: Pipeline,
    val_df: pd.DataFrame,
    policies: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Decompose model-based policy outcomes by human/AI correctness state."""

    cases = {
        "human wrong / AI correct": (val_df["correct_pre"] == 0)
        & (val_df["ai_correct"] == 1),
        "human correct / AI wrong": (val_df["correct_pre"] == 1)
        & (val_df["ai_correct"] == 0),
        "both correct": (val_df["correct_pre"] == 1) & (val_df["ai_correct"] == 1),
        "both wrong": (val_df["correct_pre"] == 0) & (val_df["ai_correct"] == 0),
        "high-conf human wrong / AI correct": (val_df["pre_conf_tercile"] == "high")
        & (val_df["correct_pre"] == 0)
        & (val_df["ai_correct"] == 1),
        "high-conf human correct / AI wrong": (val_df["pre_conf_tercile"] == "high")
        & (val_df["correct_pre"] == 1)
        & (val_df["ai_correct"] == 0),
    }
    rows = []
    for case_name, mask in cases.items():
        subgroup = val_df.loc[mask].copy()
        if subgroup.empty:
            continue
        row: dict[str, object] = {"Case": case_name, "n": len(subgroup)}
        for policy_name, all_displayed in policies.items():
            displayed = all_displayed[mask.to_numpy()]
            _, p_shift = counterfactual_response(pipe, subgroup, displayed)
            row[f"{policy_name}_predicted_reliance"] = float(np.mean(p_shift))
            row[f"{policy_name}_MSE"] = team_mse_and_rce(pipe, subgroup, displayed)[0]
        rows.append(row)
    return pd.DataFrame(rows)


def bounded_frontier(
    pipe: Pipeline, val_df: pd.DataFrame, g2_bundle: tuple
) -> pd.DataFrame:
    """Evaluate model-based outcomes along the bounded display-shift frontier."""

    policies = {
        "g0": g0(val_df),
        "g1": g1(val_df),
        "g1_delta_0.30": bounded_g1(val_df, 0.30),
        "g1_delta_0.20": bounded_g1(val_df, 0.20),
        "g1_delta_0.15": bounded_g1(val_df, 0.15),
        "g1_delta_0.10": bounded_g1(val_df, 0.10),
        "g2": apply_g2(val_df, g2_bundle),
        "g2_bounded_0.20": np.clip(
            np.clip(
                apply_g2(val_df, g2_bundle),
                g0(val_df) - 0.20,
                g0(val_df) + 0.20,
            ),
            0.05,
            0.95,
        ),
        "g3": g3(val_df),
    }
    raw = g0(val_df)
    high_mask = val_df["pre_conf_tercile"].to_numpy() == "high"
    rows = []
    for name, displayed in policies.items():
        mse, rce, _ = team_mse_and_rce(pipe, val_df, displayed)
        high_mse = team_mse_and_rce(pipe, val_df.loc[high_mask], displayed[high_mask])[0]
        rows.append(
            {
                "Policy": name,
                "max_abs_display_shift": float(np.max(np.abs(displayed - raw))),
                "mean_abs_display_shift": float(np.mean(np.abs(displayed - raw))),
                "pct_at_0_or_1": 100
                * float(np.mean(np.isclose(displayed, 0) | np.isclose(displayed, 1))),
                "Team_MSE": mse,
                "RCE_model_based": rce,
                "High_conf_MSE": high_mse,
            }
        )
    return pd.DataFrame(rows)


def simple_interface_proxies(pipe: Pipeline, val_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate simpler interface proxies using model-based off-policy outcomes."""

    confidence = g0(val_df)
    policies = {
        "raw confidence": confidence,
        "no confidence proxy": np.repeat(0.5, len(val_df)),
        "coarse verbal proxy": np.where(
            confidence < 0.4, 0.2, np.where(confidence < 0.7, 0.55, 0.85)
        ),
        "extremity compression": 0.5 + 0.6 * (confidence - 0.5),
        "thresholded high confidence": np.where(confidence > 0.7, confidence, 0.5),
        "bounded human-aware": bounded_g1(val_df, 0.20),
    }
    return eval_policy_table(pipe, val_df, policies)


def pilot_power_analysis() -> pd.DataFrame:
    """Return two-sample normal-approximation planning calculations."""

    specs = [
        ("RCE reduction", 0.188, 0.156, 0.10),
        ("Over-reliance reduction", 0.240, 0.195, 0.20),
        ("Accuracy improvement", 0.740, 0.750, 0.22),
        ("High-conf harm", 0.024, 0.000, 0.18),
    ]
    z_alpha = stats.norm.ppf(1 - 0.05 / 2)
    z_power = stats.norm.ppf(0.80)
    rows = []
    for metric, control, treatment, standard_deviation in specs:
        effect = abs(treatment - control)
        n = (
            math.inf
            if effect == 0
            else 2 * ((z_alpha + z_power) * standard_deviation / effect) ** 2
        )
        rows.append(
            {
                "Target_metric": metric,
                "Observed_control": control,
                "Observed_treatment": treatment,
                "Observed_effect_abs": effect,
                "Planning_SD_assumption": standard_deviation,
                "Approx_N_per_condition_for_80pct_power": (
                    int(math.ceil(n)) if math.isfinite(n) else "inf"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_table(table: pd.DataFrame, output_dir: Path, filename: str) -> None:
    """Write one generated CSV without touching reference or manuscript files."""

    table.to_csv(output_dir / filename, index=False)


def reproduce(profile: str, data_path: Path, output_dir: Path) -> tuple[str, ...]:
    """Run the requested reproduction profile and return generated filenames."""

    data_path, output_dir = validate_paths(data_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(data_path)
    train_idx, val_idx = train_test_split(
        data.index,
        test_size=0.3,
        random_state=RNG,
        stratify=data["task_name"],
    )
    train_df = data.loc[train_idx].copy()
    val_df = data.loc[val_idx].copy()
    g2_bundle = train_g2(train_df)
    policies = {
        "g0": g0(val_df),
        "g1": g1(val_df),
        "g2": apply_g2(val_df, g2_bundle),
        "g3": g3(val_df),
    }
    base_pipe = fit_reliance(model_specs()[0], train_df)

    main_eval = eval_policy_table(base_pipe, val_df, policies)
    support, shift = support_and_shift(train_df, val_df, policies)
    core_outputs = {
        "tab_revision_main_policy_eval.csv": main_eval,
        "tab_revision_task_user_state_sensitivity.csv": (
            task_user_state_sensitivity(base_pipe, val_df)
        ),
        "tab_revision_support_coverage.csv": support,
        "tab_revision_clipping_shift.csv": shift,
        "tab_revision_rce_binning_sensitivity.csv": rce_binning_sensitivity(
            val_df, policies
        ),
        "tab_revision_jackknife_rce.csv": jackknife_rce(val_df, policies),
        "tab_revision_self_confidence_noise.csv": subgroup_noise_and_gaming(
            base_pipe, val_df, g2_bundle
        ),
        "tab_revision_subgroup_definition_sensitivity.csv": (
            subgroup_definition_sensitivity(base_pipe, train_df, val_df)
        ),
        "tab_revision_private_information_decomposition.csv": (
            private_information_decomposition(base_pipe, val_df, policies)
        ),
        "tab_revision_bounded_frontier.csv": bounded_frontier(
            base_pipe, val_df, g2_bundle
        ),
        "tab_revision_simple_interfaces.csv": simple_interface_proxies(base_pipe, val_df),
        "tab_revision_pilot_power.csv": pilot_power_analysis(),
    }
    for filename in CORE_TABLES:
        write_table(core_outputs[filename], output_dir, filename)
    plot_display_shift(val_df, policies, output_dir / DISPLAY_SHIFT_FIGURE)

    generated = list(CORE_TABLES) + [DISPLAY_SHIFT_FIGURE]
    if profile == "full":
        full_outputs = {
            "tab_revision_reliance_model_sensitivity.csv": reliance_model_sensitivity(
                train_df, val_df, policies
            ),
            "tab_revision_split_sensitivity.csv": split_sensitivity(data),
        }
        for filename in FULL_ONLY_TABLES:
            write_table(full_outputs[filename], output_dir, filename)
        generated.extend(FULL_ONLY_TABLES)

    missing = [name for name in generated if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Reproduction did not create expected outputs: {', '.join(missing)}")

    print(f"Reproduction profile: {profile}")
    print(f"Input data: {data_path}")
    print(f"Output directory: {output_dir}")
    print(
        f"Analysis rows: {len(data)} "
        f"(train={len(train_df)}, validation={len(val_df)})"
    )
    print(f"Generated {len(generated)} file(s):")
    for filename in generated:
        print(f"  {filename}")
    return tuple(generated)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; return nonzero on any incomplete or invalid run."""

    args = parse_args(argv)
    try:
        reproduce(args.profile, args.data, args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
