"""LMB per-split gate diagnostics for the repeated-split analysis.

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/run_figure4_lmb_mechanism_frozen5.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR, PV_WORKDIR  # noqa: E402
import sys
from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = WORKDIR

ROOT = PV_WORKDIR

OUT_DIR = ROOT / "outputs" / "figure4_gate_mechanism" / "lmb_rerun"

from physicsgate.submitted.lmb import pipeline as experiment  # noqa: E402
from physicsgate.submitted.common.gate_fit import fit_gate_model  # noqa: E402
from physicsgate.submitted.lmb.common import CAPACITY_TARGET, DISCHARGE_ENERGY_TARGET, ENERGY_DENSITY_TARGET, NAME_COLUMN, VOLTAGE_TARGET  # noqa: E402

FROZEN_SEEDS = (16, 603, 978, 1254, 1869)

SOURCE_FEATURES = {
    "R1": "R2_direct_residual_disagreement",
    "R2": "R5_physics_out_of_range_amount",
    "R3": "R6_global_propagated_uncertainty_proxy",
}

FEATURE_SETS = [
    ("R1",),
    ("R2",),
    ("R3",),
    ("R1", "R2"),
    ("R1", "R3"),
    ("R2", "R3"),
    ("R1", "R2", "R3"),
]

CURRENT_CONTEXT: dict[str, Any] = {}

SAMPLE_ROWS: list[dict[str, Any]] = []

ABLATION_ROWS: list[dict[str, Any]] = []

FIXED_BLEND_ROWS: list[dict[str, Any]] = []


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def route_target(target: str, family: str) -> str:
    if family.startswith("VC_"):
        return {
            ENERGY_DENSITY_TARGET: "VC-ED",
            VOLTAGE_TARGET: "VC-V",
            CAPACITY_TARGET: "VC-C",
        }[target]
    return {
        ENERGY_DENSITY_TARGET: "DE-ED",
        DISCHARGE_ENERGY_TARGET: "DE-DE",
    }[target]


def checkpoint() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(SAMPLE_ROWS).to_csv(
        OUT_DIR / "Figure4_LMB_PerSample_GateDiagnostics.csv", index=False
    )
    pd.DataFrame(ABLATION_ROWS).to_csv(
        OUT_DIR / "Figure4_LMB_Reliability_Ablation.csv", index=False
    )
    pd.DataFrame(FIXED_BLEND_ROWS).to_csv(
        OUT_DIR / "Figure4_LMB_FixedBlend_Comparison.csv", index=False
    )


def group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    n_use = min(n_splits, len(unique))
    splitter = GroupKFold(n_splits=n_use)
    return list(splitter.split(np.arange(len(groups)), groups=groups))


def tune_subset(
    *,
    y_train: np.ndarray,
    direct_train: np.ndarray,
    direct_test: np.ndarray,
    residual_train: np.ndarray,
    residual_test: np.ndarray,
    reliability_train: pd.DataFrame,
    reliability_test: pd.DataFrame,
    groups: np.ndarray,
    inner_cv: int,
    alpha_grid: list[float],
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    folds = group_folds(groups, inner_cv)
    best_alpha = float(alpha_grid[0])
    best_score = np.inf
    for alpha in alpha_grid:
        validation = np.full(len(y_train), np.nan, dtype=float)
        for fit_idx, valid_idx in folds:
            _, pred, _ = fit_gate_model(
                y_train=y_train[fit_idx],
                direct_train=direct_train[fit_idx],
                direct_test=direct_train[valid_idx],
                residual_train=residual_train[fit_idx],
                residual_test=residual_train[valid_idx],
                reliability_train=reliability_train.iloc[fit_idx].reset_index(drop=True),
                reliability_test=reliability_train.iloc[valid_idx].reset_index(drop=True),
                features=features,
                alpha=float(alpha),
            )
            validation[valid_idx] = pred
        score = rmse(y_train, validation)
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    _, test_prediction, test_weight = fit_gate_model(
        y_train=y_train,
        direct_train=direct_train,
        direct_test=direct_test,
        residual_train=residual_train,
        residual_test=residual_test,
        reliability_train=reliability_train,
        reliability_test=reliability_test,
        features=features,
        alpha=best_alpha,
    )
    return test_prediction, test_weight, best_alpha, best_score


def capture_tuner(**kwargs: Any):
    prediction, selected, records = ORIGINAL_TUNER(**kwargs)
    context = dict(CURRENT_CONTEXT)
    if not context:
        raise RuntimeError("Missing LMB target context")

    y_train = np.asarray(kwargs["y_train"], dtype=float)
    y_test = np.asarray(context["y_test"], dtype=float)
    direct_train = np.asarray(kwargs["direct_train"], dtype=float)
    direct_test = np.asarray(kwargs["direct_test"], dtype=float)
    residual_train = np.asarray(kwargs["residual_train"], dtype=float)
    residual_test = np.asarray(kwargs["residual_test"], dtype=float)
    reliability_train = kwargs["reliability_train"].reset_index(drop=True)
    reliability_test = kwargs["reliability_test"].reset_index(drop=True)
    groups = np.asarray(kwargs["groups"], dtype=object)
    inner_cv = int(kwargs["inner_cv"])
    alpha_grid = [float(value) for value in kwargs["alpha_grid"]]

    full_source_features = [SOURCE_FEATURES[key] for key in ("R1", "R2", "R3")]
    _, full_prediction, full_weight = fit_gate_model(
        y_train=y_train,
        direct_train=direct_train,
        direct_test=direct_test,
        residual_train=residual_train,
        residual_test=residual_test,
        reliability_train=reliability_train,
        reliability_test=reliability_test,
        features=full_source_features,
        alpha=float(selected["alpha"]),
    )
    if not np.allclose(full_prediction, prediction, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Captured full-gate prediction does not match retained runner")

    seed = int(context["seed"])
    target = str(context["target"])
    family = str(context["family"])
    route = route_target(target, family)
    names = context["test_names"]
    for index in range(len(y_test)):
        SAMPLE_ROWS.append(
            {
                "dataset": "LMB",
                "target": target,
                "route_target": route,
                "seed": seed,
                "sample_index_within_test": index,
                "group_name": names[index],
                "y_true": y_test[index],
                "A_direct": direct_test[index],
                "A_residual": residual_test[index],
                "A_final": full_prediction[index],
                "w": full_weight[index],
                "R1_branch_disagreement": reliability_test.loc[
                    index, SOURCE_FEATURES["R1"]
                ],
                "R2_physics_extrapolation": reliability_test.loc[
                    index, SOURCE_FEATURES["R2"]
                ],
                "R3_propagated_uncertainty": reliability_test.loc[
                    index, SOURCE_FEATURES["R3"]
                ],
                "selected_gate_alpha": float(selected["alpha"]),
                "uses_test_target_for_selection": False,
            }
        )

    direct_rmse = rmse(y_test, direct_test)
    for paper_features in FEATURE_SETS:
        if paper_features == ("R1", "R2", "R3"):
            pred = full_prediction
            weights = full_weight
            alpha = float(selected["alpha"])
            score = float(selected["selection_RMSE"])
        else:
            source_features = [SOURCE_FEATURES[key] for key in paper_features]
            pred, weights, alpha, score = tune_subset(
                y_train=y_train,
                direct_train=direct_train,
                direct_test=direct_test,
                residual_train=residual_train,
                residual_test=residual_test,
                reliability_train=reliability_train,
                reliability_test=reliability_test,
                groups=groups,
                inner_cv=inner_cv,
                alpha_grid=alpha_grid,
                features=source_features,
            )
        method = "+".join(paper_features)
        method_rmse = rmse(y_test, pred)
        ABLATION_ROWS.append(
            {
                "dataset": "LMB",
                "target": target,
                "route_target": route,
                "seed": seed,
                "feature_set": method,
                "selected_gate_alpha": alpha,
                "train_oof_selection_RMSE": score,
                "test_RMSE": method_rmse,
                "direct_test_RMSE": direct_rmse,
                "paired_log2_gain_vs_direct": np.log2(direct_rmse / method_rmse),
                "mean_test_weight": float(np.mean(weights)),
            }
        )

    delta_train = residual_train - direct_train
    denominator = float(np.dot(delta_train, delta_train))
    fixed_alpha = (
        float(np.clip(np.dot(delta_train, y_train - direct_train) / denominator, 0.0, 1.0))
        if denominator > 1e-15
        else 0.0
    )
    fixed_prediction = direct_test + fixed_alpha * (residual_test - direct_test)
    comparison = {
        "Direct": direct_test,
        "Residual-only": residual_test,
        "Fixed scalar blend": fixed_prediction,
        "PhysicsGate": full_prediction,
    }
    for method, pred in comparison.items():
        method_rmse = rmse(y_test, pred)
        FIXED_BLEND_ROWS.append(
            {
                "dataset": "LMB",
                "target": target,
                "route_target": route,
                "seed": seed,
                "method": method,
                "fixed_alpha_train_oof": (
                    fixed_alpha if method == "Fixed scalar blend" else np.nan
                ),
                "test_RMSE": method_rmse,
                "direct_test_RMSE": direct_rmse,
                "paired_log2_gain_vs_direct": np.log2(direct_rmse / method_rmse),
            }
        )
    checkpoint()
    return prediction, selected, records


def evaluate_with_context(**kwargs: Any):
    CURRENT_CONTEXT.clear()
    CURRENT_CONTEXT.update(
        {
            "seed": int(kwargs["seed"]),
            "target": str(kwargs["target"]),
            "family": str(experiment.runner.FORMULA_FAMILY),
            "y_test": kwargs["test"][kwargs["target"]].to_numpy(dtype=float),
            "test_names": kwargs["test"][NAME_COLUMN].astype(str).to_numpy(),
        }
    )
    try:
        return ORIGINAL_EVALUATE(**kwargs)
    finally:
        CURRENT_CONTEXT.clear()


ORIGINAL_TUNER = experiment.ORIGINAL_GATE_TUNER

ORIGINAL_EVALUATE = experiment.runner.evaluate_target


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment.TABLE_DIR = OUT_DIR
    experiment.ORIGINAL_GATE_TUNER = capture_tuner
    experiment.runner.evaluate_target = evaluate_with_context
    sys.argv = [
        "run_figure4_lmb_mechanism_frozen5.py",
        "--seeds",
        *[str(seed) for seed in FROZEN_SEEDS],
        "--output-tag",
        "Figure4_LMB_Frozen5_Rerun",
        "--no-resume",
        "--n-jobs",
        "-1",
    ]
    experiment.main()
    checkpoint()
    print(f"Wrote {len(SAMPLE_ROWS)} LMB sample rows")
    print(f"Wrote {len(ABLATION_ROWS)} LMB ablation rows")
    print(f"Wrote {len(FIXED_BLEND_ROWS)} LMB fixed-blend rows")


if __name__ == "__main__":
    main()
