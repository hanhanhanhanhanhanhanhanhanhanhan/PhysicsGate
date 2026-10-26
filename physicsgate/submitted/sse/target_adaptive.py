"""SSE target definitions, grouped splits and ordinary stacking.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/arrhenius/run_sse_target_adaptive.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT = WORKDIR

from physicsgate.submitted.sse.iterative import TARGET_COLUMNS, TARGET_ORDER, _as_1d, _as_2d, _direct_features, _fit_and_predict_test, _group_oof_predictions, _split_group_indices, physics_prediction_for_target  # noqa: E402
from physicsgate.submitted.common.model_selection import DEFAULT_SCORING, extract_selected_feature_names, fit_model_by_strategy  # noqa: E402


def _model_metadata(
    *,
    X: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    model_strategy: str,
    role: str,
    n_iter: int,
    inner_cv: int,
    scoring: str,
    random_state: int,
    n_jobs: int,
    feature_selection: str,
) -> tuple[Any, str, str]:
    result = fit_model_by_strategy(
        X,
        y,
        groups,
        model_strategy=model_strategy,
        model_names=None,
        n_iter=n_iter,
        inner_cv=inner_cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        feature_selection=feature_selection,
    )
    return result.estimator, f"{role}:{result.selected_model}", result.selected_params_json


def direct_features_for_target(
    target: str,
    X_base: ArrayLike,
    T: ArrayLike,
) -> NDArray[np.float64]:
    """Apply target-specific direct-feature rules."""
    if target == "L":
        return _direct_features(X_base, T)
    if target in {"P", "E"}:
        return _as_2d(X_base, name="X_base")
    raise ValueError(f"Unknown target: {target}")


def residual_feature_names_for_target(
    target: str,
    base_feature_names: Sequence[str],
) -> list[str]:
    auxiliary_names = [
        f"{name}_pred" for name in TARGET_ORDER if name != target
    ]
    return [
        *base_feature_names,
        "T_K",
        f"{target}_physics_prediction",
        *auxiliary_names,
    ]


def _ordinary_base_models(random_state: int) -> list[tuple[Any, str]]:
    return [
        (Ridge(alpha=0.5), "ridge_alpha_0.5"),
        (KNeighborsRegressor(n_neighbors=3), "knn_3"),
        (DecisionTreeRegressor(max_depth=4, random_state=random_state), "tree_depth_4"),
    ]


def _ordinary_stacking_oof_and_test(
    *,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    groups_train: ArrayLike,
    X_test: NDArray[np.float64],
    random_state: int,
    inner_cv: int,
) -> NDArray[np.float64]:
    oof_columns = []
    test_columns = []
    for estimator, _ in _ordinary_base_models(random_state):
        oof_columns.append(
            _group_oof_predictions(
                estimator,
                X_train,
                y_train,
                groups_train,
                n_splits=inner_cv,
            )
        )
        test_columns.append(_fit_and_predict_test(estimator, X_train, y_train, X_test))
    train_features = np.column_stack(oof_columns)
    test_features = np.column_stack(test_columns)
    meta = clone(Ridge(alpha=1.0))
    oof_prediction = _group_oof_predictions(
        meta,
        train_features,
        y_train,
        groups_train,
        n_splits=inner_cv,
    )
    test_prediction = _fit_and_predict_test(meta, train_features, y_train, test_features)
    return oof_prediction, test_prediction


def _residual_features(
    X_base: ArrayLike,
    T: ArrayLike,
    *,
    target: str,
    physics_prediction: ArrayLike,
    predictions: Mapping[str, ArrayLike],
) -> NDArray[np.float64]:
    direct_with_temperature = _direct_features(X_base, T)
    physics = _as_1d(physics_prediction, name="physics_prediction")
    auxiliaries = [
        _as_1d(predictions[name], name=f"{name}_prediction")
        for name in TARGET_ORDER
        if name != target
    ]
    return np.column_stack([direct_with_temperature, physics, *auxiliaries])


def _fit_residual_prediction(
    *,
    target: str,
    X_train_base: NDArray[np.float64],
    T_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    groups_train: ArrayLike,
    physics_oof: NDArray[np.float64],
    current_oof: Mapping[str, NDArray[np.float64]],
    X_test_base: NDArray[np.float64],
    T_test: NDArray[np.float64],
    physics_test: NDArray[np.float64],
    current_test: Mapping[str, NDArray[np.float64]],
    model_strategy: str,
    inner_cv: int,
    n_iter: int,
    scoring: str,
    random_state: int,
    n_jobs: int,
    feature_selection: str,
    feature_names: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, str, list[str]]:
    residual_features_oof = _residual_features(
        X_train_base,
        T_train,
        target=target,
        physics_prediction=physics_oof,
        predictions=current_oof,
    )
    residual_targets = y_train - physics_oof
    estimator, selected_model, params_json = _model_metadata(
        X=residual_features_oof,
        y=residual_targets,
        groups=groups_train,
        model_strategy=model_strategy,
        role="residual",
        n_iter=n_iter,
        inner_cv=inner_cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        feature_selection=feature_selection,
    )
    selected_features = extract_selected_feature_names(estimator, feature_names)
    residual_oof = physics_oof + _group_oof_predictions(
        estimator,
        residual_features_oof,
        residual_targets,
        groups_train,
        n_splits=inner_cv,
    )
    residual_features_test = _residual_features(
        X_test_base,
        T_test,
        target=target,
        physics_prediction=physics_test,
        predictions=current_test,
    )
    residual_test = physics_test + _fit_and_predict_test(
        estimator,
        residual_features_oof,
        residual_targets,
        residual_features_test,
    )
    return residual_oof, residual_test, selected_model, params_json, selected_features


def _fit_physres_prediction(
    *,
    y_train: NDArray[np.float64],
    groups_train: ArrayLike,
    direct_oof: NDArray[np.float64],
    physics_oof: NDArray[np.float64],
    residual_oof: NDArray[np.float64],
    direct_test: NDArray[np.float64],
    physics_test: NDArray[np.float64],
    residual_test: NDArray[np.float64],
    model_strategy: str,
    inner_cv: int,
    n_iter: int,
    scoring: str,
    random_state: int,
    n_jobs: int,
    feature_selection: str,
    feature_names: Sequence[str],
) -> tuple[NDArray[np.float64], str, str, list[str]]:
    train_features = np.column_stack([direct_oof, physics_oof, residual_oof])
    test_features = np.column_stack([direct_test, physics_test, residual_test])
    estimator, selected_model, params_json = _model_metadata(
        X=train_features,
        y=y_train,
        groups=groups_train,
        model_strategy=model_strategy,
        role="meta",
        n_iter=n_iter,
        inner_cv=inner_cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        feature_selection=feature_selection,
    )
    selected_features = extract_selected_feature_names(estimator, feature_names)
    # OOF meta predictions are intentionally generated for leakage checks and
    # parity with training, even though final reporting uses held-out test rows.
    _group_oof_predictions(
        estimator,
        train_features,
        y_train,
        groups_train,
        n_splits=inner_cv,
    )
    prediction = _fit_and_predict_test(estimator, train_features, y_train, test_features)
    return prediction, selected_model, params_json, selected_features
