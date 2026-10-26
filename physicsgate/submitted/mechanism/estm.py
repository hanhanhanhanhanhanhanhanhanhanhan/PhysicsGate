"""ESTM per-split gate diagnostics for the repeated-split analysis.

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/run_figure4_estm_mechanism_frozen5.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR, PV_WORKDIR  # noqa: E402
import argparse
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO = WORKDIR

ROOT = PV_WORKDIR

OUT_DIR = ROOT / "outputs" / "figure4_gate_mechanism" / "estm_rerun"

from physicsgate.submitted.estm import stability as experiment  # noqa: E402
from physicsgate.submitted.common.gate_fit import fit_gate_model, split_positions  # noqa: E402

FROZEN_SEEDS = (4, 42, 99, 178, 2024)

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
        OUT_DIR / "Figure4_ESTM_PerSample_GateDiagnostics.csv", index=False
    )
    pd.DataFrame(ABLATION_ROWS).to_csv(
        OUT_DIR / "Figure4_ESTM_Reliability_Ablation.csv", index=False
    )
    pd.DataFrame(FIXED_BLEND_ROWS).to_csv(
        OUT_DIR / "Figure4_ESTM_FixedBlend_Comparison.csv", index=False
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
    inner_cv: int,
    alpha_grid: list[float],
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    truth = np.asarray(y_train, dtype=float).reshape(-1)
    direct = np.asarray(direct_train, dtype=float).reshape(-1)
    residual = np.asarray(residual_train, dtype=float).reshape(-1)
    folds = split_positions(groups, inner_cv, seed)
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
        finite = np.isfinite(validation)
        score = rmse(truth[finite], validation[finite])
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


ORIGINAL_TUNER = experiment.tune_r2_r5_r6_gate

ORIGINAL_EVALUATE = experiment.evaluate_fold


def capture_tuner(**kwargs: Any):
    prediction, selected, records = ORIGINAL_TUNER(**kwargs)
    source_features = [SOURCE_FEATURES[name] for name in ("R1", "R2", "R3")]
    _, full_prediction, full_weight = fit_gate_model(
        y_train=kwargs["y_train"],
        direct_train=kwargs["direct_train"],
        direct_test=kwargs["direct_test"],
        residual_train=kwargs["residual_train"],
        residual_test=kwargs["residual_test"],
        reliability_train=kwargs["reliability_train"],
        reliability_test=kwargs["reliability_test"],
        alpha=float(selected["alpha"]),
        features=source_features,
    )
    if not np.allclose(full_prediction, prediction, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Captured ESTM full-gate prediction does not match retained runner")

    target = str(CURRENT_CONTEXT["target"])
    target_label = str(experiment.TARGET_DISPLAY[target])
    seed = int(CURRENT_CONTEXT["seed"])
    fold_id = int(CURRENT_CONTEXT["fold_id"])
    y_test = np.asarray(CURRENT_CONTEXT["y_test"], dtype=float)
    formula_ids = np.asarray(CURRENT_CONTEXT["formula_ids"], dtype=str)
    row_indices = np.asarray(CURRENT_CONTEXT["row_indices"], dtype=int)
    direct_test = np.asarray(kwargs["direct_test"], dtype=float)
    residual_test = np.asarray(kwargs["residual_test"], dtype=float)
    rel_test = kwargs["reliability_test"].reset_index(drop=True)
    direct_rmse = rmse(y_test, direct_test)

    for index in range(len(y_test)):
        SAMPLE_ROWS.append(
            {
                "dataset": "ESTM",
                "target": target_label,
                "target_code": target,
                "seed": seed,
                "fold_id": fold_id,
                "source_row_index": row_indices[index],
                "formula_id": formula_ids[index],
                "y_true": y_test[index],
                "A_direct": direct_test[index],
                "A_residual": residual_test[index],
                "A_final": full_prediction[index],
                "w": full_weight[index],
                "R1_branch_disagreement": rel_test.loc[index, source_features[0]],
                "R2_physics_extrapolation": rel_test.loc[index, source_features[1]],
                "R3_propagated_uncertainty": rel_test.loc[index, source_features[2]],
                "selected_gate_alpha": selected["alpha"],
                "uses_test_target_for_selection": False,
            }
        )

    for paper_features in FEATURE_SETS:
        source_subset = [SOURCE_FEATURES[name] for name in paper_features]
        if paper_features == ("R1", "R2", "R3"):
            pred = full_prediction
            weight = full_weight
            alpha = float(selected["alpha"])
            score = float(selected["selection_RMSE"])
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
                        "inner_cv",
                        "alpha_grid",
                    )
                },
                features=source_subset,
            )
        value = rmse(y_test, pred)
        ABLATION_ROWS.append(
            {
                "dataset": "ESTM",
                "target": target_label,
                "target_code": target,
                "seed": seed,
                "fold_id": fold_id,
                "feature_set": "+".join(paper_features),
                "selected_gate_alpha": alpha,
                "train_oof_selection_RMSE": score,
                "test_RMSE": value,
                "direct_test_RMSE": direct_rmse,
                "paired_log2_gain_vs_direct": np.log2(direct_rmse / value),
                "mean_test_weight": float(np.mean(weight)),
                "n_test": len(y_test),
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
                "dataset": "ESTM",
                "target": target_label,
                "target_code": target,
                "seed": seed,
                "fold_id": fold_id,
                "method": method,
                "fixed_alpha_train_oof": (
                    fixed_alpha if method == "Fixed scalar blend" else np.nan
                ),
                "test_RMSE": value,
                "direct_test_RMSE": direct_rmse,
                "paired_log2_gain_vs_direct": np.log2(direct_rmse / value),
                "n_test": len(y_test),
            }
        )
    checkpoint()
    return prediction, selected, records


def evaluate_with_context(**kwargs: Any):
    target = str(kwargs["target"])
    test_idx = np.asarray(kwargs["test_idx"], dtype=int)
    data = kwargs["data"]
    CURRENT_CONTEXT.clear()
    CURRENT_CONTEXT.update(
        {
            "seed": int(kwargs["outer_seed"]),
            "fold_id": int(kwargs["fold_id"]),
            "target": target,
            "y_test": data[target].to_numpy(dtype=float)[test_idx],
            "formula_ids": data["formula_id"].astype(str).to_numpy()[test_idx],
            "row_indices": data["core_v2_row_index"].to_numpy(dtype=int)[test_idx],
        }
    )
    try:
        return ORIGINAL_EVALUATE(**kwargs)
    finally:
        CURRENT_CONTEXT.clear()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment.tune_r2_r5_r6_gate = capture_tuner
    experiment.evaluate_fold = evaluate_with_context
    args = argparse.Namespace(
        inner_cv=5,
        gate_inner_cv=5,
        n_splits=3,
        n_jobs=-1,
        base_model="xgb_fixed",
    )
    alpha_grid = [
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1000.0,
    ]
    frame = experiment.estm.complete_estm_frame(
        pd.read_csv(experiment.DEFAULT_INPUT_PATH)
    )
    X, _ = experiment.build_feature_matrix(
        frame.drop(columns=["core_v2_row_index"])
    )
    detail_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    total = len(FROZEN_SEEDS) * len(experiment.TARGET_COLUMNS) * args.n_splits
    done = 0
    for seed in FROZEN_SEEDS:
        folds = experiment.split_indices(
            frame,
            split_mode="paper_interpolation",
            n_splits=args.n_splits,
            random_state=seed,
        )
        for target in experiment.TARGET_COLUMNS:
            for fold_id, (train_idx, test_idx) in enumerate(folds):
                detail_rows.extend(
                    experiment.evaluate_fold(
                        data=frame,
                        X=X,
                        target=target,
                        outer_seed=seed,
                        fold_id=int(fold_id),
                        train_idx=train_idx,
                        test_idx=test_idx,
                        args=args,
                        alpha_grid=alpha_grid,
                        selection_rows=selection_rows,
                        audit_rows=audit_rows,
                    )
                )
                done += 1
                print(f"[ESTM {done}/{total}] seed={seed} target={target} fold={fold_id}")
    pd.DataFrame(detail_rows).to_csv(
        OUT_DIR / "Figure4_ESTM_Frozen5_Rerun_Detailed.csv", index=False
    )
    pd.DataFrame(selection_rows).to_csv(
        OUT_DIR / "Figure4_ESTM_Frozen5_Rerun_Selection.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        OUT_DIR / "Figure4_ESTM_Frozen5_Rerun_Audit.csv", index=False
    )
    checkpoint()
    print(f"Wrote {len(SAMPLE_ROWS)} ESTM sample rows")
    print(f"Wrote {len(ABLATION_ROWS)} ESTM ablation rows")
    print(f"Wrote {len(FIXED_BLEND_ROWS)} ESTM fixed-blend rows")


if __name__ == "__main__":
    main()
