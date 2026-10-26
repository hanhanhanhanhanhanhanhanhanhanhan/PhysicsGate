"""LMB method helpers.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/lmb/run_lmb_90_10_clean_methods_a2.py`` in the original analysis code; only imports and paths were adapted.
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = WORKDIR

from physicsgate.submitted.lmb.common import metrics_dict  # noqa: E402


def _as_array(values: object) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def _elasticnet_cv(
    *,
    random_state: int,
    l1_ratio: list[float],
    alphas: NDArray[np.float64],
    cv: int,
) -> ElasticNetCV:
    """Return a stable ElasticNetCV candidate for low-dimensional meta features."""
    return ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=alphas,
        cv=cv,
        max_iter=1000000,
        tol=1e-3,
        random_state=random_state,
        selection="cyclic",
    )


def _candidate_factories(
    random_state: int,
    n_jobs: int,
    *,
    search_level: str,
) -> dict[str, Callable[[], BaseEstimator]]:
    factories: dict[str, Callable[[], BaseEstimator]] = {
        "RidgeCV": lambda: make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])),
        ),
        "ElasticNetCV": lambda: make_pipeline(
            StandardScaler(),
            _elasticnet_cv(
                random_state=random_state,
                l1_ratio=[0.1, 0.5, 0.9],
                alphas=np.array([0.001, 0.01, 0.1, 1.0, 10.0]),
                cv=3,
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
    if search_level == "standard":
        return factories
    if search_level != "wide":
        raise ValueError(f"Unsupported search_level: {search_level}")

    factories["RidgeCV_broad"] = lambda: make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.array([1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])),
    )
    factories["ElasticNetCV_broad"] = lambda: make_pipeline(
        StandardScaler(),
        _elasticnet_cv(
            random_state=random_state,
            l1_ratio=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            alphas=np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]),
            cv=5,
        ),
    )
    for depth in (None, 3, 5, 8, 12):
        for leaf in (1, 2, 5):
            label_depth = "none" if depth is None else str(depth)
            factories[f"RandomForest_depth{label_depth}_leaf{leaf}"] = (
                lambda depth=depth, leaf=leaf: RandomForestRegressor(
                    n_estimators=260,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    max_features=1.0,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
            )
            factories[f"ExtraTrees_depth{label_depth}_leaf{leaf}"] = (
                lambda depth=depth, leaf=leaf: ExtraTreesRegressor(
                    n_estimators=260,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    max_features=1.0,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
            )
    for leaf_nodes in (7, 15, 31):
        for learning_rate in (0.03, 0.05, 0.08):
            factories[f"HistGradientBoosting_leaf{leaf_nodes}_lr{learning_rate}"] = (
                lambda leaf_nodes=leaf_nodes, learning_rate=learning_rate: HistGradientBoostingRegressor(
                    max_iter=260,
                    learning_rate=learning_rate,
                    max_leaf_nodes=leaf_nodes,
                    l2_regularization=0.03,
                    random_state=random_state,
                )
            )
    for depth in (2, 3):
        for learning_rate in (0.03, 0.05, 0.08):
            factories[f"GradientBoosting_depth{depth}_lr{learning_rate}"] = (
                lambda depth=depth, learning_rate=learning_rate: GradientBoostingRegressor(
                    n_estimators=260,
                    learning_rate=learning_rate,
                    max_depth=depth,
                    random_state=random_state,
                )
            )
    try:
        from xgboost import XGBRegressor  # type: ignore

        for depth in (2, 3):
            for learning_rate in (0.03, 0.05):
                factories[f"XGB_depth{depth}_lr{learning_rate}"] = (
                    lambda depth=depth, learning_rate=learning_rate: XGBRegressor(
                        n_estimators=260,
                        max_depth=depth,
                        learning_rate=learning_rate,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        tree_method="hist",
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                )
    except Exception:
        pass
    return factories


def _select_prediction_model(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    *,
    random_state: int,
    n_jobs: int,
    inner_cv: int,
    search_level: str,
    role: str,
    selection_rows: list[dict[str, object]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, float]:
    factories = _candidate_factories(random_state, n_jobs, search_level=search_level)
    splits = list(
        KFold(n_splits=min(inner_cv, len(y_train)), shuffle=True, random_state=random_state).split(
            X_train
        )
    )
    best_name = ""
    best_score = float("inf")
    best_oof: NDArray[np.float64] | None = None
    row_start = len(selection_rows)
    for name, factory in factories.items():
        oof = np.full(y_train.shape, np.nan, dtype=float)
        for inner_fold, (fit_idx, valid_idx) in enumerate(splits):
            model = factory()
            try:
                model.set_params(random_state=random_state + inner_fold * 19)
            except (AttributeError, ValueError):
                pass
            model.fit(X_train[fit_idx], y_train[fit_idx])
            oof[valid_idx] = _as_array(model.predict(X_train[valid_idx]))
        score = float(metrics_dict(y_train, oof)["RMSE"])
        selection_rows.append(
            {
                "role": role,
                "candidate_model": name,
                "search_level": search_level,
                "inner_oof_RMSE": score,
                "selected": False,
            }
        )
        if score < best_score:
            best_name = name
            best_score = score
            best_oof = oof.copy()
    if best_oof is None:
        raise RuntimeError("No candidate model was selected")
    final = factories[best_name]()
    final.fit(X_train, y_train)
    for row in selection_rows[row_start:]:
        row["selected"] = row["candidate_model"] == best_name
    return best_oof, _as_array(final.predict(X_test)), best_name, best_score
