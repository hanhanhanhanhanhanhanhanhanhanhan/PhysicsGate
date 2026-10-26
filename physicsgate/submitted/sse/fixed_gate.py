"""SSE fixed-gate helpers.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/tmp_sse_fixed_gate_multiseed_stability.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import sys
import time
import argparse
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

from physicsgate.submitted.common.gate_fit import fit_gate_model, metric_dict, split_positions  # noqa: E402

INNER_CV = 3

GATE_FEATURES = [
    "R2_direct_residual_disagreement",
    "R5_physics_out_of_range_amount",
    "R6_global_propagated_uncertainty_proxy",
]

DEFAULT_GATE_ALPHAS = "0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000"


def tune_fixed_gate_alpha(
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
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Select only Ridge alpha by train-side group CV; keep R2+R5+R6 fixed."""
    truth = np.asarray(y_train, dtype=float).reshape(-1)
    direct = np.asarray(direct_train, dtype=float).reshape(-1)
    residual = np.asarray(residual_train, dtype=float).reshape(-1)
    folds = split_positions(groups, INNER_CV, seed)
    best_alpha = float(alpha_grid[0])
    best_score = np.inf
    for alpha in alpha_grid:
        validation_prediction = np.full_like(truth, np.nan, dtype=float)
        for fit_idx, valid_idx in folds:
            _, pred_valid, _ = fit_gate_model(
                y_train=truth[fit_idx],
                direct_train=direct[fit_idx],
                direct_test=direct[valid_idx],
                residual_train=residual[fit_idx],
                residual_test=residual[valid_idx],
                reliability_train=reliability_train.iloc[fit_idx].reset_index(drop=True),
                reliability_test=reliability_train.iloc[valid_idx].reset_index(drop=True),
                features=GATE_FEATURES,
                alpha=float(alpha),
            )
            validation_prediction[valid_idx] = pred_valid
        score = metric_dict(truth, validation_prediction)["RMSE"]
        if score < best_score:
            best_score = float(score)
            best_alpha = float(alpha)
    _, test_prediction, test_weight = fit_gate_model(
        y_train=truth,
        direct_train=direct_train,
        direct_test=direct_test,
        residual_train=residual_train,
        residual_test=residual_test,
        reliability_train=reliability_train,
        reliability_test=reliability_test,
        features=GATE_FEATURES,
        alpha=best_alpha,
    )
    return test_prediction, test_weight, best_alpha, best_score


GATE_METHOD = "FixedGate_R2_R5_R6"


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        detail.groupby(
            ["dataset", "target", "target_code", "metric_level", "primary_metric_level", "method"],
            as_index=False,
        )
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index(drop=True)
    )
    direct = grouped[grouped["method"].eq("Direct")][["target", "metric_level", "RMSE_mean"]].rename(
        columns={"RMSE_mean": "direct_RMSE_mean"}
    )
    grouped = grouped.merge(direct, on=["target", "metric_level"], how="left")
    grouped["percent_RMSE_change_vs_direct_of_means"] = (
        (grouped["RMSE_mean"] - grouped["direct_RMSE_mean"]) / grouped["direct_RMSE_mean"] * 100.0
    )
    grouped = grouped.drop(columns=["direct_RMSE_mean"])

    # Per-seed percent change (paired within each seed), then mean/std across
    # seeds -- a more honest stability measure than the ratio-of-means above.
    pivot = detail.pivot_table(
        index=["target", "target_code", "metric_level", "primary_metric_level", "seed"],
        columns="method",
        values="RMSE",
    ).reset_index()
    pivot["pct_change_gate_vs_direct"] = (
        (pivot[GATE_METHOD] - pivot["Direct"]) / pivot["Direct"] * 100.0
    )
    pct_summary = pivot.groupby(
        ["target", "target_code", "metric_level", "primary_metric_level"], as_index=False
    ).agg(
        gate_pct_change_vs_direct_mean_per_seed=("pct_change_gate_vs_direct", "mean"),
        gate_pct_change_vs_direct_std_per_seed=("pct_change_gate_vs_direct", "std"),
    )
    grouped = grouped.merge(
        pct_summary,
        on=["target", "target_code", "metric_level", "primary_metric_level"],
        how="left",
    )
    return grouped.sort_values(["target", "metric_level", "RMSE_mean"]).reset_index(drop=True)
