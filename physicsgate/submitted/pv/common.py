"""PV columns and encoding.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/photovoltaic/common.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS_RAW = [
    "Cell_architecture",
    "Cell_flexible",
    "Cell_semitransparent",
    "Substrate_stack_sequence",
    "ETL_stack_sequence",
    "ETL_thickness",
    "ETL_deposition_procedure",
    "Perovskite_dimension_2D",
    "Perovskite_dimension_2D3D_mixture",
    "Perovskite_dimension_3D",
    "Perovskite_dimension_3D_with_2D_capping_layer",
    "Perovskite_composition_perovskite_ABC3_structure",
    "Perovskite_composition_long_form",
    "Perovskite_thickness",
    "Perovskite_composition_inorganic",
    "Perovskite_band_gap",
    "Perovskite_band_gap_graded",
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_solvents",
    "Perovskite_deposition_quenching_induced_crystallisation",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Perovskite_deposition_solvent_annealing",
    "HTL_stack_sequence",
    "HTL_thickness_list",
    "HTL_deposition_procedure",
    "Backcontact_stack_sequence",
    "Backcontact_thickness_list",
    "Backcontact_deposition_procedure",
]

TARGET_COLUMNS_RAW = {
    "Voc_V": "JV_default_Voc",
    "Jsc_mA_cm2": "JV_default_Jsc",
    "FF_raw": "JV_default_FF",
    "PCE_percent": "JV_default_PCE",
}

TARGET_COLUMNS = ["Voc_V", "Jsc_mA_cm2", "FF_fraction", "PCE_percent"]

NUMERIC_FEATURES_RAW = [
    "ETL_thickness",
    "Perovskite_thickness",
    "Perovskite_band_gap",
    "HTL_thickness_list",
    "Backcontact_thickness_list",
]

BOOLEAN_FEATURES_RAW = [
    "Cell_flexible",
    "Cell_semitransparent",
    "Perovskite_dimension_2D",
    "Perovskite_dimension_2D3D_mixture",
    "Perovskite_dimension_3D",
    "Perovskite_dimension_3D_with_2D_capping_layer",
    "Perovskite_composition_perovskite_ABC3_structure",
    "Perovskite_composition_inorganic",
    "Perovskite_band_gap_graded",
    "Perovskite_deposition_quenching_induced_crystallisation",
    "Perovskite_deposition_solvent_annealing",
]

PAPER_RF_PARAMS = {
    "Voc_V": {
        "n_estimators": 100,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "log2",
        "max_depth": None,
    },
    "Jsc_mA_cm2": {
        "n_estimators": 200,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "log2",
        "max_depth": None,
    },
    "FF_fraction": {
        "n_estimators": 200,
        "min_samples_split": 10,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "max_depth": 30,
    },
    "PCE_percent": {
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "max_depth": 30,
    },
}


def safe_name(name: str) -> str:
    """Convert a raw column name to a stable snake_case identifier."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")
    cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cleaned)
    return re.sub(r"_+", "_", cleaned).lower()


FEATURE_COLUMNS = [safe_name(name) for name in FEATURE_COLUMNS_RAW]

NUMERIC_FEATURES = [safe_name(name) for name in NUMERIC_FEATURES_RAW]

BOOLEAN_FEATURES = [safe_name(name) for name in BOOLEAN_FEATURES_RAW]

CATEGORICAL_FEATURES = [
    name
    for name in FEATURE_COLUMNS
    if name not in set(NUMERIC_FEATURES) | set(BOOLEAN_FEATURES)
]


def pce_from_terms(
    voc_v: Iterable[float] | np.ndarray,
    jsc_ma_cm2: Iterable[float] | np.ndarray,
    ff_fraction: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Compute PCE percent from photovoltaic device terms."""
    return np.asarray(voc_v, dtype=float) * np.asarray(jsc_ma_cm2, dtype=float) * np.asarray(
        ff_fraction, dtype=float
    )


def metrics_dict(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Return MSE, RMSE, MAE, nRMSE, and R2."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    target_range = float(np.max(y_true_arr) - np.min(y_true_arr))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "nrmse": float(rmse / target_range) if target_range > 0 else np.nan,
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }


class SimpleTargetEncoder:
    """Small target-mean encoder used when category_encoders is unavailable.

    The upstream notebooks use `category_encoders.TargetEncoder`. The execution
    environment for this repository does not require that optional package, so
    this local encoder implements the same benchmark idea: category means are
    fitted target-specifically and unseen categories map to the global mean.
    """

    def __init__(self) -> None:
        self.global_mean_: float | None = None
        self.maps_: dict[str, dict[object, float]] = {}

    def fit(self, frame: pd.DataFrame, y: Iterable[float]) -> "SimpleTargetEncoder":
        y_series = pd.Series(np.asarray(y, dtype=float), index=frame.index)
        self.global_mean_ = float(y_series.mean())
        self.maps_ = {}
        for column in frame.columns:
            values = frame[column].fillna("__missing__")
            grouped = y_series.groupby(values).mean()
            self.maps_[column] = grouped.to_dict()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.global_mean_ is None:
            raise RuntimeError("SimpleTargetEncoder must be fitted before transform.")
        encoded = pd.DataFrame(index=frame.index)
        for column in frame.columns:
            mapping = self.maps_.get(column, {})
            encoded[column] = frame[column].fillna("__missing__").map(mapping).fillna(
                self.global_mean_
            )
        return encoded.astype(float)

    def fit_transform(self, frame: pd.DataFrame, y: Iterable[float]) -> pd.DataFrame:
        return self.fit(frame, y).transform(frame)


def encode_features_for_target(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    *,
    protocol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleTargetEncoder, MinMaxScaler]:
    """Encode categorical features and scale features for one target."""
    if protocol not in {"paper_matched", "leakage_safe"}:
        raise ValueError(f"Unknown protocol: {protocol}")
    encoder = SimpleTargetEncoder()
    scaler = MinMaxScaler()

    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    if CATEGORICAL_FEATURES:
        X_train_enc[CATEGORICAL_FEATURES] = encoder.fit_transform(
            X_train[CATEGORICAL_FEATURES], y_train
        )
        X_test_enc[CATEGORICAL_FEATURES] = encoder.transform(X_test[CATEGORICAL_FEATURES])
    for column in BOOLEAN_FEATURES:
        X_train_enc[column] = X_train_enc[column].astype(float)
        X_test_enc[column] = X_test_enc[column].astype(float)
    X_train_enc[FEATURE_COLUMNS] = scaler.fit_transform(X_train_enc[FEATURE_COLUMNS])
    X_test_enc[FEATURE_COLUMNS] = scaler.transform(X_test_enc[FEATURE_COLUMNS])
    return X_train_enc[FEATURE_COLUMNS], X_test_enc[FEATURE_COLUMNS], encoder, scaler


def build_rf_model(
    target: str,
    *,
    random_state: int,
    n_jobs: int,
    rf_n_estimators: int | None = None,
) -> RandomForestRegressor:
    """Return the paper/GitHub RandomForestRegressor for a target."""
    params = dict(PAPER_RF_PARAMS[target])
    if rf_n_estimators is not None:
        params["n_estimators"] = rf_n_estimators
    return RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, **params)
