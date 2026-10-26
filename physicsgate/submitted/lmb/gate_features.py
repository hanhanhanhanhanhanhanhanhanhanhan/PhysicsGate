"""LMB reliability features.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/tmp_lmb_physicsgate_r2_r5_r6_test.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = WORKDIR

from physicsgate.submitted.common.gate_fit import fit_gate_model, metric_dict, split_positions  # noqa: E402

METHOD_R2_R5_R6 = "DirectAnchored_PhysicsGate_R2_R5_R6_tuned"

R2_R5_R6_FEATURES = [
    "R2_direct_residual_disagreement",
    "R5_physics_out_of_range_amount",
    "R6_global_propagated_uncertainty_proxy",
]


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def tune_r2_r5_r6_gate(
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
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Tune the minimal R2+R5+R6 gate by train-side OOF RMSE only."""
    records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    y = _as_array(y_train)
    folds = split_positions(groups, inner_cv, seed)
    for alpha in alpha_grid:
        val_pred = np.full_like(y, np.nan, dtype=float)
        for fit_idx, valid_idx in folds:
            _, pred_valid, _ = fit_gate_model(
                y_train=y[fit_idx],
                direct_train=_as_array(direct_train)[fit_idx],
                direct_test=_as_array(direct_train)[valid_idx],
                residual_train=_as_array(residual_train)[fit_idx],
                residual_test=_as_array(residual_train)[valid_idx],
                reliability_train=reliability_train.iloc[fit_idx].reset_index(drop=True),
                reliability_test=reliability_train.iloc[valid_idx].reset_index(drop=True),
                features=R2_R5_R6_FEATURES,
                alpha=alpha,
            )
            val_pred[valid_idx] = pred_valid
        score = metric_dict(y[np.isfinite(val_pred)], val_pred[np.isfinite(val_pred)])["RMSE"]
        record = {
            "method": METHOD_R2_R5_R6,
            "feature_set": "R2_R5_R6",
            "features": json.dumps(R2_R5_R6_FEATURES),
            "alpha": alpha,
            "selection_RMSE": score,
            "selection_metric": "group_cv_train_oof_RMSE",
        }
        records.append(record)
        if best is None or score < best["selection_RMSE"]:
            best = record
    assert best is not None
    _train_pred, test_pred, test_w = fit_gate_model(
        y_train=y_train,
        direct_train=direct_train,
        direct_test=direct_test,
        residual_train=residual_train,
        residual_test=residual_test,
        reliability_train=reliability_train,
        reliability_test=reliability_test,
        features=R2_R5_R6_FEATURES,
        alpha=float(best["alpha"]),
    )
    best = dict(best)
    best["mean_test_weight"] = float(np.mean(test_w))
    best["median_test_weight"] = float(np.median(test_w))
    return test_pred, best, records
