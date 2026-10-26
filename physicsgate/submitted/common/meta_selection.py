"""Meta-learner and prediction-model selection.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/run_sse_estm_pv_wide_meta_a2_eval.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import sys
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

from physicsgate.submitted.common.metrics import rmse  # noqa: E402


def _as_array(values: object) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def _elasticnet_cv(
    *,
    random_state: int,
    l1_ratio: list[float],
    alphas: NDArray[np.float64],
    cv: int,
) -> ElasticNetCV:
    return ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=alphas,
        cv=cv,
        max_iter=1000000,
        tol=1e-3,
        random_state=random_state,
        selection="cyclic",
    )


def candidate_factories(random_state: int, n_jobs: int) -> dict[str, Callable[[], BaseEstimator]]:
    # Keep this deliberately smaller than the LMB wide search.  Each candidate
    # is evaluated by inner OOF for multiple branches, so an overly broad grid
    # makes this cross-dataset smoke test impractical without changing the
    # underlying method definitions.
    factories: dict[str, Callable[[], BaseEstimator]] = {
        "RidgeCV": lambda: make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.array([1e-4, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])),
        ),
        "ElasticNetCV_broad": lambda: make_pipeline(
            StandardScaler(),
            _elasticnet_cv(
                random_state=random_state,
                l1_ratio=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                alphas=np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]),
                cv=3,
            ),
        ),
    }
    for depth in (None, 5):
        for leaf in (1, 3):
            depth_label = "none" if depth is None else str(depth)
            factories[f"ExtraTrees_depth{depth_label}_leaf{leaf}"] = (
                lambda depth=depth, leaf=leaf: ExtraTreesRegressor(
                    n_estimators=80,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    max_features=1.0,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
            )
    for leaf_nodes, learning_rate in ((15, 0.05), (31, 0.03)):
        factories[f"HistGradientBoosting_leaf{leaf_nodes}_lr{learning_rate}"] = (
            lambda leaf_nodes=leaf_nodes, learning_rate=learning_rate: HistGradientBoostingRegressor(
                max_iter=120,
                learning_rate=learning_rate,
                max_leaf_nodes=leaf_nodes,
                l2_regularization=0.03,
                random_state=random_state,
            )
        )
    factories["RandomForest_depth5_leaf3"] = lambda: RandomForestRegressor(
        n_estimators=80,
        max_depth=5,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    factories["GradientBoosting_depth2"] = lambda: GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=2,
        random_state=random_state,
    )
    try:
        from xgboost import XGBRegressor  # type: ignore

        factories["XGB_depth2"] = lambda: XGBRegressor(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
        )
    except Exception:
        pass
    return factories


def _splits(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    groups: NDArray[np.object_] | None,
    n_splits: int,
    random_state: int,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            splitter = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
            return [
                (fit.astype(int), hold.astype(int))
                for fit, hold in splitter.split(X, y, groups)
            ]
    splitter = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=random_state)
    return [(fit.astype(int), hold.astype(int)) for fit, hold in splitter.split(X, y)]


def select_prediction_model(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    *,
    dataset: str,
    unit_id: str,
    role: str,
    groups: NDArray[np.object_] | None,
    random_state: int,
    n_jobs: int,
    inner_cv: int,
    selection_rows: list[dict[str, object]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, float]:
    factories = candidate_factories(random_state, n_jobs)
    splits = _splits(X_train, y_train, groups=groups, n_splits=inner_cv, random_state=random_state)
    start = len(selection_rows)
    best_name = ""
    best_score = float("inf")
    best_oof: NDArray[np.float64] | None = None
    for name, factory in factories.items():
        oof = np.full(y_train.shape, np.nan, dtype=float)
        for inner_fold, (fit_idx, hold_idx) in enumerate(splits):
            model = factory()
            try:
                model.set_params(random_state=random_state + inner_fold * 19)
            except (AttributeError, ValueError):
                pass
            model.fit(X_train[fit_idx], y_train[fit_idx])
            oof[hold_idx] = _as_array(model.predict(X_train[hold_idx]))
        score = float(rmse(y_train, oof))
        selection_rows.append(
            {
                "dataset": dataset,
                "unit_id": unit_id,
                "role": role,
                "candidate_model": name,
                "inner_oof_RMSE": score,
                "selected": False,
                "uses_group_inner_cv": groups is not None,
            }
        )
        if score < best_score:
            best_name = name
            best_score = score
            best_oof = oof.copy()
    if best_oof is None:
        raise RuntimeError(f"No model selected for {dataset} {unit_id} {role}")
    final = factories[best_name]()
    final.fit(X_train, y_train)
    for row in selection_rows[start:]:
        row["selected"] = row["candidate_model"] == best_name
    return best_oof, _as_array(final.predict(X_test)), best_name, best_score
