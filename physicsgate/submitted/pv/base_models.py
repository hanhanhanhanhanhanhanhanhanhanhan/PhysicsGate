"""PV base learners and meta selection.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/run_pv_fresh_fair_benchmark_v2_gated.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
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

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

from physicsgate.submitted.pv.common import FEATURE_COLUMNS, build_rf_model, encode_features_for_target  # noqa: E402
from physicsgate.submitted.common.metrics import rmse  # noqa: E402

DATASET = "PV"

TARGET = "PCE_percent"

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


def split_indices(n_rows: int, *, test_size: float, random_state: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    indices = np.arange(n_rows)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def meta_candidate_factories(random_state: int, n_jobs: int) -> dict[str, Callable[[], BaseEstimator]]:
    factories: dict[str, Callable[[], BaseEstimator]] = {
        "RidgeCV": lambda: make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.array([0.01, 0.1, 1.0, 10.0, 100.0])),
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
    }
    for depth in (2, 3, 4):
        for leaf in (5, 10):
            factories[f"RandomForest_depth{depth}_leaf{leaf}"] = (
                lambda depth=depth, leaf=leaf: RandomForestRegressor(
                    n_estimators=180,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
            )
            factories[f"ExtraTrees_depth{depth}_leaf{leaf}"] = (
                lambda depth=depth, leaf=leaf: ExtraTreesRegressor(
                    n_estimators=180,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
            )
    for leaf_nodes in (15, 31):
        factories[f"HistGradientBoosting_leafnodes{leaf_nodes}"] = (
            lambda leaf_nodes=leaf_nodes: HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.05,
                max_leaf_nodes=leaf_nodes,
                l2_regularization=0.1,
                random_state=random_state,
            )
        )
    try:
        from xgboost import XGBRegressor  # type: ignore

        for depth in (2, 3):
            factories[f"XGB_depth{depth}"] = (
                lambda depth=depth: XGBRegressor(
                    n_estimators=180,
                    max_depth=depth,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
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
    y_test_for_report: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, str, float, list[dict[str, Any]]]:
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
                "selected_hyperparameters": safe_json(factory()),
                "candidate_scores_json": safe_json({"inner_oof_RMSE": score}),
                "selection_metric": "inner_oof_RMSE",
                "validation_score": score,
                "selected": False,
                "selected_meta_learner": "",
                "test_score_after_selection": np.nan,
                "selection_uses_test": False,
            }
        )
        if score < best_score:
            best_score = score
            best_name = name
            best_oof = oof.copy()
    if best_oof is None:
        raise ValueError(f"No meta candidates evaluated for role {role}")
    final = factories[best_name]()
    final.fit(X_train, y_train)
    test_pred = _as_array(final.predict(X_test))
    selected_params = safe_json(final)
    test_score = float(rmse(y_test_for_report, test_pred)) if y_test_for_report is not None else np.nan
    for row in rows:
        row["selected"] = row["candidate_meta_learner"] == best_name
        row["selected_meta_learner"] = best_name
        if row["selected"]:
            row["test_score_after_selection"] = test_score
    return best_oof, test_pred, best_name, selected_params, best_score, rows


def base_candidate_factories(
    target: str,
    *,
    random_state: int,
    n_jobs: int,
    rf_n_estimators: int | None,
) -> list[tuple[str, Callable[[], BaseEstimator], dict[str, Any]]]:
    n_estimators = rf_n_estimators or 300
    candidates: list[tuple[str, Callable[[], BaseEstimator], dict[str, Any]]] = []
    paper_params = {
        "model": "paper_rf",
        "rf_n_estimators": rf_n_estimators,
    }
    candidates.append(
        (
            "paper_rf",
            lambda: build_rf_model(
                target,
                random_state=random_state,
                n_jobs=n_jobs,
                rf_n_estimators=rf_n_estimators,
            ),
            paper_params,
        )
    )
    for max_features in ("sqrt", 0.5, 0.8, 1.0):
        params = {
            "model": "ExtraTreesRegressor",
            "n_estimators": max(300, n_estimators),
            "max_features": max_features,
            "min_samples_leaf": 1,
        }
        candidates.append(
            (
                f"extra_trees_mf{max_features}_leaf1",
                lambda params=params: ExtraTreesRegressor(
                    n_estimators=params["n_estimators"],
                    max_features=params["max_features"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
                params,
            )
        )
    for max_features, leaf in (("sqrt", 2), (0.5, 2), (0.8, 4)):
        params = {
            "model": "RandomForestRegressor",
            "n_estimators": max(300, n_estimators),
            "max_features": max_features,
            "min_samples_leaf": leaf,
            "max_depth": None,
        }
        candidates.append(
            (
                f"random_forest_mf{max_features}_leaf{leaf}",
                lambda params=params: RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_features=params["max_features"],
                    min_samples_leaf=params["min_samples_leaf"],
                    max_depth=params["max_depth"],
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
                params,
            )
        )
    for learning_rate, max_leaf_nodes, l2 in ((0.03, 31, 0.1), (0.05, 31, 0.1), (0.08, 15, 1.0)):
        params = {
            "model": "HistGradientBoostingRegressor",
            "max_iter": 300,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "l2_regularization": l2,
        }
        candidates.append(
            (
                f"hgb_lr{learning_rate}_leaf{max_leaf_nodes}_l2{l2}",
                lambda params=params: HistGradientBoostingRegressor(
                    max_iter=params["max_iter"],
                    learning_rate=params["learning_rate"],
                    max_leaf_nodes=params["max_leaf_nodes"],
                    l2_regularization=params["l2_regularization"],
                    random_state=random_state,
                ),
                params,
            )
        )
    for learning_rate, depth in ((0.03, 3), (0.05, 3)):
        params = {
            "model": "GradientBoostingRegressor",
            "n_estimators": 300,
            "learning_rate": learning_rate,
            "max_depth": depth,
        }
        candidates.append(
            (
                f"gradient_boosting_lr{learning_rate}_depth{depth}",
                lambda params=params: GradientBoostingRegressor(
                    n_estimators=params["n_estimators"],
                    learning_rate=params["learning_rate"],
                    max_depth=params["max_depth"],
                    random_state=random_state,
                ),
                params,
            )
        )
    try:
        from xgboost import XGBRegressor  # type: ignore

        for n_est, depth, lr, reg_lambda, min_child_weight in (
            (300, 3, 0.05, 1.0, 1),
            (600, 3, 0.03, 5.0, 3),
            (600, 4, 0.05, 5.0, 3),
            (900, 3, 0.02, 10.0, 5),
        ):
            params = {
                "model": "XGBRegressor",
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": reg_lambda,
                "min_child_weight": min_child_weight,
            }
            candidates.append(
                (
                    f"xgb_n{n_est}_d{depth}_lr{lr}_l2{reg_lambda}_mcw{min_child_weight}",
                    lambda params=params: XGBRegressor(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        learning_rate=params["learning_rate"],
                        subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"],
                        reg_lambda=params["reg_lambda"],
                        min_child_weight=params["min_child_weight"],
                        objective="reg:squarederror",
                        tree_method="hist",
                        random_state=random_state,
                        n_jobs=n_jobs,
                    ),
                    params,
                )
            )
    except Exception:
        pass
    return candidates


def tune_base_model(
    data: pd.DataFrame,
    *,
    target: str,
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    random_state: int,
    n_jobs: int,
    inner_cv: int,
    rf_n_estimators: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, str, list[dict[str, Any]]]:
    X_raw = data[FEATURE_COLUMNS]
    y = data[target].astype(float)
    y_train = y.iloc[train_idx].to_numpy(dtype=float)
    y_test = y.iloc[test_idx].to_numpy(dtype=float)
    X_train_raw = X_raw.iloc[train_idx]
    X_test_raw = X_raw.iloc[test_idx]
    folds = KFold(n_splits=min(inner_cv, len(train_idx)), shuffle=True, random_state=random_state)
    candidates = base_candidate_factories(
        target,
        random_state=random_state,
        n_jobs=n_jobs,
        rf_n_estimators=rf_n_estimators,
    )
    tuning_rows: list[dict[str, Any]] = []
    best_name = ""
    best_params = ""
    best_score = float("inf")
    best_factory: Callable[[], BaseEstimator] | None = None
    for candidate_name, factory, params in candidates:
        candidate_oof = np.full(y_train.shape, np.nan, dtype=float)
        for fold_id, (fit_pos, valid_pos) in enumerate(folds.split(X_train_raw)):
            fit_index = train_idx[fit_pos]
            valid_index = train_idx[valid_pos]
            X_fit, X_valid, _, _ = encode_features_for_target(
                X_raw.iloc[fit_index],
                X_raw.iloc[valid_index],
                y.iloc[fit_index],
                protocol="leakage_safe",
            )
            model = factory()
            if hasattr(model, "set_params"):
                try:
                    model.set_params(random_state=random_state + fold_id * 17)
                except ValueError:
                    pass
            model.fit(X_fit, y.iloc[fit_index])
            candidate_oof[valid_pos] = _as_array(model.predict(X_valid))
        score = float(rmse(y_train, candidate_oof))
        row = {
            "dataset": DATASET,
            "target": target,
            "analysis_subset": FAIR_SUBSET,
            "role": f"base_model_{target}",
            "candidate_model": candidate_name,
            "candidate_params_json": safe_json(params),
            "selection_metric": "inner_oof_RMSE",
            "validation_score": score,
            "selected": False,
            "selected_model": "",
            "selected_params_json": "",
            "test_score_after_selection": np.nan,
            "base_model_selected_by_test": False,
        }
        tuning_rows.append(row)
        if score < best_score:
            best_score = score
            best_name = candidate_name
            best_params = safe_json(params)
            best_factory = factory
    if best_factory is None:
        raise ValueError(f"No base model candidates evaluated for {target}")

    oof = np.full(y_train.shape, np.nan, dtype=float)
    for fold_id, (fit_pos, valid_pos) in enumerate(folds.split(X_train_raw)):
        fit_index = train_idx[fit_pos]
        valid_index = train_idx[valid_pos]
        X_fit, X_valid, _, _ = encode_features_for_target(
            X_raw.iloc[fit_index],
            X_raw.iloc[valid_index],
            y.iloc[fit_index],
            protocol="leakage_safe",
        )
        model = best_factory()
        if hasattr(model, "set_params"):
            try:
                model.set_params(random_state=random_state + fold_id * 17)
            except ValueError:
                pass
        model.fit(X_fit, y.iloc[fit_index])
        oof[valid_pos] = _as_array(model.predict(X_valid))
    if not np.all(np.isfinite(oof)):
        raise ValueError(f"Every training row must receive an OOF prediction for {target}")
    X_train, X_test, _, _ = encode_features_for_target(
        X_train_raw,
        X_test_raw,
        y.iloc[train_idx],
        protocol="leakage_safe",
    )
    full_model = best_factory()
    full_model.fit(X_train, y.iloc[train_idx])
    test_pred = _as_array(full_model.predict(X_test))
    test_score = float(rmse(y_test, test_pred))
    for row in tuning_rows:
        row["selected"] = row["candidate_model"] == best_name
        row["selected_model"] = best_name
        row["selected_params_json"] = best_params
        if row["selected"]:
            row["test_score_after_selection"] = test_score
    return oof, test_pred, best_name, best_params, tuning_rows
