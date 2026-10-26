"""ESTM base learners and OOF predictions.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/run_estm_fresh_fair_benchmark.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

from physicsgate.submitted.estm.features import build_xgb_estimator  # noqa: E402
from physicsgate.submitted.estm.zt import xgb_available  # noqa: E402
from physicsgate.submitted.common.metrics import rmse  # noqa: E402

DATASET = "ESTM"

TARGET = "ZT"

FAIR_SUBSET = "physics_complete_fair_comparison"


def _as_array(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def safe_json(value: Any) -> str:
    def convert(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, BaseEstimator):
            return item.__class__.__name__
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        return repr(item)

    return json.dumps(convert(value), sort_keys=True)


def build_base_tree_estimator(
    *,
    random_state: int,
    n_jobs: int,
    base_model: str,
) -> tuple[BaseEstimator, str, str]:
    """Return a strong direct/auxiliary tree model without test-set tuning."""
    if base_model == "xgb_fixed" and xgb_available():
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.045,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 2,
            "reg_lambda": 1.0,
            "reg_alpha": 1e-4,
            "gamma": 0.0,
        }
        estimator = build_xgb_estimator(
            random_state=random_state,
            n_jobs=n_jobs,
            **params,
        )
        return estimator, "xgb_fixed_strong", safe_json(params)
    params = {
        "n_estimators": 500,
        "max_features": 0.8,
        "min_samples_leaf": 1,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    return ExtraTreesRegressor(**params), "extra_trees_fallback", safe_json(params)


def oof_and_test_predictions(
    estimator: BaseEstimator,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    *,
    n_splits: int,
    random_state: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    folds = KFold(n_splits=min(n_splits, len(y_train)), shuffle=True, random_state=random_state)
    oof = np.full(y_train.shape, np.nan, dtype=float)
    for fold_index, (fit_idx, valid_idx) in enumerate(folds.split(X_train)):
        model = clone(estimator)
        if hasattr(model, "set_params"):
            try:
                model.set_params(random_state=random_state + fold_index * 17)
            except ValueError:
                pass
        model.fit(X_train[fit_idx], y_train[fit_idx])
        oof[valid_idx] = _as_array(model.predict(X_train[valid_idx]))
    if not np.all(np.isfinite(oof)):
        raise ValueError("Every train row must receive an OOF prediction")
    full = clone(estimator)
    full.fit(X_train, y_train)
    test = _as_array(full.predict(X_test))
    return oof, test


def meta_candidate_factories(random_state: int, n_jobs: int) -> dict[str, Callable[[], BaseEstimator]]:
    factories: dict[str, Callable[[], BaseEstimator]] = {
        "RidgeCV": lambda: make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])),
        ),
        "ElasticNetCV": lambda: make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                alphas=np.array([0.001, 0.01, 0.1, 1.0, 10.0]),
                cv=3,
                max_iter=100000,
                tol=1e-3,
                random_state=random_state,
            ),
        ),
        "RandomForestRegressor": lambda: RandomForestRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "ExtraTreesRegressor": lambda: ExtraTreesRegressor(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "HistGradientBoostingRegressor": lambda: HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.01,
            random_state=random_state,
        ),
    }
    if xgb_available():
        factories["XGBRegressor"] = lambda: build_xgb_estimator(
            random_state=random_state,
            n_jobs=n_jobs,
            n_estimators=240,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            reg_alpha=1e-4,
        )
    return factories


def select_meta_model(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    *,
    role: str,
    fold_id: int,
    random_state: int,
    n_jobs: int,
    n_splits: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best_name = ""
    best_score = float("inf")
    best_oof: NDArray[np.float64] | None = None
    factories = meta_candidate_factories(random_state, n_jobs)
    folds = KFold(n_splits=min(n_splits, len(y_train)), shuffle=True, random_state=random_state)
    for name, factory in factories.items():
        oof = np.full(y_train.shape, np.nan, dtype=float)
        for inner_fold, (fit_idx, valid_idx) in enumerate(folds.split(X_train)):
            model = factory()
            if hasattr(model, "set_params"):
                try:
                    model.set_params(random_state=random_state + inner_fold * 19)
                except ValueError:
                    pass
            model.fit(X_train[fit_idx], y_train[fit_idx])
            oof[valid_idx] = _as_array(model.predict(X_train[valid_idx]))
        score = float(rmse(y_train, oof))
        rows.append(
            {
                "dataset": DATASET,
                "target": TARGET,
                "analysis_subset": FAIR_SUBSET,
                "fold_id": int(fold_id),
                "role": role,
                "candidate_meta_learner": name,
                "selection_metric": "inner_oof_RMSE",
                "validation_score": score,
                "selected": False,
                "selection_uses_test": False,
            }
        )
        if score < best_score:
            best_score = score
            best_name = name
            best_oof = oof.copy()
    if best_oof is None:
        raise ValueError(f"No meta model candidates evaluated for role {role}")
    final = factories[best_name]()
    final.fit(X_train, y_train)
    test_pred = _as_array(final.predict(X_test))
    for row in rows:
        row["selected"] = row["candidate_meta_learner"] == best_name
        row["selected_meta_learner"] = best_name
    return best_oof, test_pred, best_name, best_score, rows
