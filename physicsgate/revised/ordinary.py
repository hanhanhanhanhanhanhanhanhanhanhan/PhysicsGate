"""Ordinary stacking of ESTM and PV as defined in the Methods (correction C2).

The meta-learner combines out-of-fold predictions of base learners fitted
directly to the target from the inputs of Direct: the Direct model itself and
four of the families of the LMB ordinary stacking, with the same settings
(ridge, random forest, extra trees and histogram gradient boosting). The
meta-learner is selected as before. The submitted code gave the meta-learner the Direct, auxiliary and
Physics-only predictions instead.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def base_factories(random_state: int, n_jobs: int) -> dict[str, Callable[[], BaseEstimator]]:
    return {
        "RidgeCV": lambda: make_pipeline(StandardScaler(),
                                         RidgeCV(alphas=np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0]))),
        "RandomForestRegressor": lambda: RandomForestRegressor(n_estimators=160, max_depth=6, min_samples_leaf=2,
                                                               random_state=random_state, n_jobs=n_jobs),
        "ExtraTreesRegressor": lambda: ExtraTreesRegressor(n_estimators=160, max_depth=8, min_samples_leaf=1,
                                                           random_state=random_state, n_jobs=n_jobs),
        "HistGradientBoostingRegressor": lambda: HistGradientBoostingRegressor(
            max_iter=160, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.01, random_state=random_state),
    }


def estm_base_predictions(X_train, y_train, X_test, *, random_state: int, n_jobs: int, inner_cv: int,
                          oof_and_test) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """OOF and test predictions of the four further base learners on the ESTM Direct inputs."""
    oof, test = [], []
    for k, factory in enumerate(base_factories(random_state, n_jobs).values()):
        o, t = oof_and_test(factory(), X_train, y_train, X_test, n_splits=inner_cv, random_state=random_state + k * 7)
        oof.append(o)
        test.append(t)
    return oof, test


def pv_base_predictions(data: pd.DataFrame, target: str, train_idx, test_idx, *, feature_columns, encode,
                        random_state: int, n_jobs: int, inner_cv: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """As ``estm_base_predictions`` for PV, with the target encoding fitted within each inner fold as for Direct."""
    X_raw = data[feature_columns]
    y = data[target].astype(float)
    X_train_raw, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
    folds = list(KFold(n_splits=min(inner_cv, len(train_idx)), shuffle=True, random_state=random_state)
                 .split(X_train_raw))
    encoded = []
    for fit_pos, valid_pos in folds:
        X_fit, X_valid, _, _ = encode(X_raw.iloc[train_idx[fit_pos]], X_raw.iloc[train_idx[valid_pos]],
                                      y.iloc[train_idx[fit_pos]], protocol="leakage_safe")
        encoded.append((fit_pos, valid_pos, X_fit, X_valid))
    X_train, X_test, _, _ = encode(X_train_raw, X_test_raw, y.iloc[train_idx], protocol="leakage_safe")
    oof_list, test_list = [], []
    for k, factory in enumerate(base_factories(random_state, n_jobs).values()):
        oof = np.full(len(train_idx), np.nan, dtype=float)
        for fold_id, (fit_pos, valid_pos, X_fit, X_valid) in enumerate(encoded):
            model = factory()
            try:
                model.set_params(random_state=random_state + k * 7 + fold_id * 17)
            except (AttributeError, ValueError):
                pass
            model.fit(X_fit, y.iloc[train_idx[fit_pos]])
            oof[valid_pos] = np.asarray(model.predict(X_valid), dtype=float).reshape(-1)
        if not np.all(np.isfinite(oof)):
            raise RuntimeError(f"Incomplete OOF prediction for {target}")
        full = factory()
        full.fit(X_train, y.iloc[train_idx])
        oof_list.append(oof)
        test_list.append(np.asarray(full.predict(X_test), dtype=float).reshape(-1))
    return oof_list, test_list
