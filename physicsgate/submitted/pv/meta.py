"""PV paper-matched helpers.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/run_pv_aux_targets_paper_matched_wide_meta_a2.py`` in the original analysis code; only imports and paths were adapted.
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
from numpy.typing import NDArray

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

from physicsgate.submitted.pv import base_models as pvgated  # noqa: E402
from physicsgate.submitted.common.metrics import mae, r2, rmse  # noqa: E402

ANALYSIS_SUBSET = "paper_matched_pce_aligned_aux_targets_wide_meta"


def _as_array(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def nrmse(y_true: Any, y_pred: Any) -> float:
    truth = _as_array(y_true)
    error = float(rmse(truth, y_pred))
    spread = float(np.std(truth))
    return float(error / spread) if spread > 0 else np.nan


def metric_dict(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "RMSE": float(rmse(y_true, y_pred)),
        "MAE": float(mae(y_true, y_pred)),
        "nRMSE": nrmse(y_true, y_pred),
        "R2": float(r2(y_true, y_pred)),
    }


def select_meta(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    *,
    y_test: NDArray[np.float64],
    target: str,
    role: str,
    random_state: int,
    n_jobs: int,
    inner_cv: int,
    meta_rows: list[dict[str, Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, float]:
    oof, test, selected, _params, score, rows = pvgated.select_meta_model(
        X_train,
        y_train,
        X_test,
        role=role,
        fold_id=0,
        random_state=random_state,
        n_jobs=n_jobs,
        n_splits=inner_cv,
        y_test_for_report=y_test,
    )
    for row in rows:
        row["target"] = target
        row["analysis_subset"] = ANALYSIS_SUBSET
    meta_rows.extend(rows)
    return oof, test, selected, score
