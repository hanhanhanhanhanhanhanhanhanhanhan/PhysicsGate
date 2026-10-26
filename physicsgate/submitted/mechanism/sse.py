"""SSE per-split gate diagnostics for the repeated-split analysis.

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/run_figure4_sse_mechanism_frozen5.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR, PV_WORKDIR  # noqa: E402
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO = WORKDIR

ROOT = PV_WORKDIR

OUT_DIR = ROOT / "outputs" / "figure4_gate_mechanism" / "sse_rerun"

from physicsgate.submitted.sse import pipeline as experiment  # noqa: E402
from physicsgate.submitted.common.gate_fit import fit_gate_model, split_positions  # noqa: E402

FROZEN_SEEDS = (14, 483, 655, 752, 2392)

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


def checkpoint() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(SAMPLE_ROWS).to_csv(
        OUT_DIR / "Figure4_SSE_PerSample_GateDiagnostics.csv", index=False
    )
    pd.DataFrame(ABLATION_ROWS).to_csv(
        OUT_DIR / "Figure4_SSE_Reliability_Ablation.csv", index=False
    )
    pd.DataFrame(FIXED_BLEND_ROWS).to_csv(
        OUT_DIR / "Figure4_SSE_FixedBlend_Comparison.csv", index=False
    )


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
    seed: int,
    alpha_grid: list[float],
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    truth = np.asarray(y_train, dtype=float).reshape(-1)
    direct = np.asarray(direct_train, dtype=float).reshape(-1)
    residual = np.asarray(residual_train, dtype=float).reshape(-1)
    folds = split_positions(groups, experiment.INNER_CV, seed)
    best_alpha = float(alpha_grid[0])
    best_score = np.inf
    for alpha in alpha_grid:
        validation = np.full_like(truth, np.nan, dtype=float)
        for fit_idx, valid_idx in folds:
            _, pred_valid, _ = fit_gate_model(
                y_train=truth[fit_idx],
                direct_train=direct[fit_idx],
                direct_test=direct[valid_idx],
                residual_train=residual[fit_idx],
                residual_test=residual[valid_idx],
                reliability_train=reliability_train.iloc[fit_idx].reset_index(drop=True),
                reliability_test=reliability_train.iloc[valid_idx].reset_index(drop=True),
                alpha=float(alpha),
                features=features,
            )
            validation[valid_idx] = pred_valid
        score = rmse(truth, validation)
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    _, prediction, weight = fit_gate_model(
        y_train=truth,
        direct_train=direct,
        direct_test=direct_test,
        residual_train=residual,
        residual_test=residual_test,
        reliability_train=reliability_train,
        reliability_test=reliability_test,
        alpha=best_alpha,
        features=features,
    )
    return prediction, weight, best_alpha, best_score


ORIGINAL_TUNER = experiment.tune_fixed_gate_alpha

ORIGINAL_RUN = experiment.run_seed_target


def capture_tuner(**kwargs: Any):
    prediction, retained_weight, selected_alpha, selected_score = ORIGINAL_TUNER(**kwargs)
    source_features = [SOURCE_FEATURES[name] for name in ("R1", "R2", "R3")]
    _, full_prediction, full_weight = fit_gate_model(
        y_train=kwargs["y_train"],
        direct_train=kwargs["direct_train"],
        direct_test=kwargs["direct_test"],
        residual_train=kwargs["residual_train"],
        residual_test=kwargs["residual_test"],
        reliability_train=kwargs["reliability_train"],
        reliability_test=kwargs["reliability_test"],
        alpha=float(selected_alpha),
        features=source_features,
    )
    if not np.allclose(full_prediction, prediction, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Captured SSE full-gate prediction does not match retained runner")
    if not np.allclose(full_weight, retained_weight, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Captured SSE gate weights do not match retained runner")

    target = str(CURRENT_CONTEXT["target"])
    target_label = str(experiment.TARGET_MODE[target])
    seed = int(CURRENT_CONTEXT["seed"])
    y_test = np.asarray(CURRENT_CONTEXT["y_test"], dtype=float)
    materials = np.asarray(CURRENT_CONTEXT["test_materials"], dtype=str)
    direct_test = np.asarray(kwargs["direct_test"], dtype=float)
    residual_test = np.asarray(kwargs["residual_test"], dtype=float)
    rel_test = kwargs["reliability_test"].reset_index(drop=True)
    direct_rmse = rmse(y_test, direct_test)

    for index in range(len(y_test)):
        SAMPLE_ROWS.append(
            {
                "dataset": "SSE",
                "target": target_label,
                "target_code": target,
                "seed": seed,
                "sample_index_within_test": index,
                "material_group": materials[index],
                "y_true": y_test[index],
                "A_direct": direct_test[index],
                "A_residual": residual_test[index],
                "A_final": full_prediction[index],
                "w": full_weight[index],
                "R1_branch_disagreement": rel_test.loc[index, source_features[0]],
                "R2_physics_extrapolation": rel_test.loc[index, source_features[1]],
                "R3_propagated_uncertainty": rel_test.loc[index, source_features[2]],
                "selected_gate_alpha": selected_alpha,
                "uses_test_target_for_selection": False,
            }
        )

    for paper_features in FEATURE_SETS:
        source_subset = [SOURCE_FEATURES[name] for name in paper_features]
        if paper_features == ("R1", "R2", "R3"):
            pred = full_prediction
            weight = full_weight
            alpha = float(selected_alpha)
            score = float(selected_score)
        else:
            pred, weight, alpha, score = tune_subset(
                **{
                    key: kwargs[key]
                    for key in (
                        "y_train",
                        "direct_train",
                        "direct_test",
                        "residual_train",
                        "residual_test",
                        "reliability_train",
                        "reliability_test",
                        "groups",
                        "seed",
                        "alpha_grid",
                    )
                },
                features=source_subset,
            )
        method_rmse = rmse(y_test, pred)
        ABLATION_ROWS.append(
            {
                "dataset": "SSE",
                "target": target_label,
                "target_code": target,
                "seed": seed,
                "feature_set": "+".join(paper_features),
                "selected_gate_alpha": alpha,
                "train_oof_selection_RMSE": score,
                "test_RMSE": method_rmse,
                "direct_test_RMSE": direct_rmse,
                "paired_log2_gain_vs_direct": np.log2(direct_rmse / method_rmse),
                "mean_test_weight": float(np.mean(weight)),
            }
        )

    direct_train = np.asarray(kwargs["direct_train"], dtype=float)
    residual_train = np.asarray(kwargs["residual_train"], dtype=float)
    y_train = np.asarray(kwargs["y_train"], dtype=float)
    delta_train = residual_train - direct_train
    denominator = float(np.dot(delta_train, delta_train))
    fixed_alpha = (
        float(np.clip(np.dot(delta_train, y_train - direct_train) / denominator, 0.0, 1.0))
        if denominator > 1e-15
        else 0.0
    )
    fixed_prediction = direct_test + fixed_alpha * (residual_test - direct_test)
    for method, pred in {
        "Direct": direct_test,
        "Residual-only": residual_test,
        "Fixed scalar blend": fixed_prediction,
        "PhysicsGate": full_prediction,
    }.items():
        value = rmse(y_test, pred)
        FIXED_BLEND_ROWS.append(
            {
                "dataset": "SSE",
                "target": target_label,
                "target_code": target,
                "seed": seed,
                "method": method,
                "fixed_alpha_train_oof": (
                    fixed_alpha if method == "Fixed scalar blend" else np.nan
                ),
                "test_RMSE": value,
                "direct_test_RMSE": direct_rmse,
                "paired_log2_gain_vs_direct": np.log2(direct_rmse / value),
            }
        )
    checkpoint()
    return prediction, retained_weight, selected_alpha, selected_score


def run_with_context(**kwargs: Any):
    seed = int(kwargs["seed"])
    target = str(kwargs["target"])
    data = kwargs["data"]
    target_col = experiment.TARGET_COLUMNS[target]
    valid = data.loc[data[target_col].notna()].reset_index(drop=True)
    train_idx, test_idx = experiment._split_group_indices(
        valid["Materials"].to_numpy(),
        test_size=float(kwargs["test_size"]),
        random_state=seed,
    )
    test = valid.iloc[test_idx].reset_index(drop=True)
    CURRENT_CONTEXT.clear()
    CURRENT_CONTEXT.update(
        {
            "seed": seed,
            "target": target,
            "y_test": test[target_col].to_numpy(dtype=float),
            "test_materials": test["Materials"].astype(str).to_numpy(),
        }
    )
    try:
        return ORIGINAL_RUN(**kwargs)
    finally:
        CURRENT_CONTEXT.clear()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment.tune_fixed_gate_alpha = capture_tuner
    experiment.run_seed_target = run_with_context
    data, feature_columns = experiment.load_and_prepare_data()
    alpha_grid = [
        float(value)
        for value in experiment.DEFAULT_GATE_ALPHAS.split(",")
        if value.strip()
    ]
    all_rows: list[dict[str, Any]] = []
    for seed in FROZEN_SEEDS:
        for target in experiment.TARGET_ORDER:
            rows = experiment.run_seed_target(
                seed=seed,
                target=target,
                data=data,
                feature_columns=feature_columns,
                alpha_grid=alpha_grid,
                test_size=experiment.TEST_SIZE,
                gate_weighting="delta2",
                include_r3=False,
                include_physresstack=False,
            )
            all_rows.extend(rows)
            print(f"[SSE seed={seed} target={target}] complete")
    pd.DataFrame(all_rows).to_csv(
        OUT_DIR / "Figure4_SSE_Frozen5_Rerun_Detailed.csv", index=False
    )
    checkpoint()
    print(f"Wrote {len(SAMPLE_ROWS)} SSE sample rows")
    print(f"Wrote {len(ABLATION_ROWS)} SSE ablation rows")
    print(f"Wrote {len(FIXED_BLEND_ROWS)} SSE fixed-blend rows")


if __name__ == "__main__":
    main()
