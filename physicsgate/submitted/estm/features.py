"""ESTM composition features and splits.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/thermoelectric/run_estm_direct_baselines.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

ROOT = WORKDIR

DEFAULT_INPUT_PATH = ROOT / "data" / "processed" / "estm_clean.csv"


def infer_element_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("elem_")]
    if not columns:
        raise ValueError("No elem_* composition columns found. Run the ESTM loader first.")
    return sorted(columns, key=lambda name: int(name.split("_")[1]))


def build_feature_matrix(frame: pd.DataFrame) -> tuple[NDArray[np.float64], list[str]]:
    """Return sparse composition fraction features plus temperature."""
    element_columns = infer_element_columns(frame)
    feature_columns = [*element_columns, "temperature_K"]
    X = frame.loc[:, feature_columns].to_numpy(dtype=float)
    return X, feature_columns


def split_indices(
    frame: pd.DataFrame,
    *,
    split_mode: str,
    n_splits: int,
    random_state: int,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Create row-wise Table 2 interpolation or diagnostic formula-group splits."""
    if split_mode == "paper_interpolation":
        splitter = KFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=int(random_state),
        )
        return [
            (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
            for train, test in splitter.split(frame)
        ]
    if split_mode == "formula_group":
        groups = frame["formula_id"].to_numpy()
        n_group_splits = min(int(n_splits), len(np.unique(groups)))
        if n_group_splits < 2:
            raise ValueError("At least two formula groups are required for GroupKFold")
        splitter = GroupKFold(n_splits=n_group_splits)
        return [
            (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
            for train, test in splitter.split(frame, groups=groups)
        ]
    raise ValueError("split_mode must be paper_interpolation or formula_group")


def require_xgboost():
    try:
        from xgboost import XGBRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "xgboost is required for the ESTM Table 2 XGB benchmark"
        ) from exc
    return XGBRegressor


def build_xgb_estimator(
    *,
    random_state: int,
    n_jobs: int,
    **params: Any,
) -> BaseEstimator:
    """Build an XGBRegressor with a squared-error objective for all ESTM targets."""
    XGBRegressor = require_xgboost()
    defaults: dict[str, Any] = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "verbosity": 0,
    }
    defaults.update(params)
    return XGBRegressor(**defaults)
