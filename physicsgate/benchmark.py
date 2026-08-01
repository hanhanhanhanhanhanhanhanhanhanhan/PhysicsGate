"""Unified fresh-refit benchmark for the four retained scientific domains.

This runner is intentionally compact. It enforces the information-flow
contract shared by all domains: auxiliary predictions and second-stage
training features are out-of-fold, and outer-test labels are used only for
final metrics. Retained figure-source CSVs remain the exact numerical source
for the reported figures; this module provides a clean raw-data refit path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .equations import ARRHENIUS_KB_EV_PER_K
from .gate import PhysicsGate, reliability_features


EPS = 1e-12
ESTM_POSITIVE_FLOOR = 1e-300
FINITE_DIFFERENCE_ABSOLUTE_STEP = 1e-6
FINITE_DIFFERENCE_RELATIVE_STEP = 1e-4
METHODS = (
    "Direct",
    "Physics-only",
    "Residual-only",
    "Ordinary stacking",
    "PhysResStack",
    "PhysicsGate",
)
RIDGE_ALPHAS = np.logspace(-5, 5, 21)


@dataclass(frozen=True)
class Endpoint:
    code: str
    target: str
    auxiliaries: tuple[str, ...]
    route: str


ENDPOINTS = {
    "SSE": (
        Endpoint("ln_sigma_T", "ln_sigma_T", ("ln_sigma0", "Ea"), "arrhenius"),
        Endpoint("Ea", "Ea", ("ln_sigma_T", "ln_sigma0"), "arrhenius"),
        Endpoint("ln_sigma0", "ln_sigma0", ("ln_sigma_T", "Ea"), "arrhenius"),
    ),
    "ESTM": (
        Endpoint(
            "Seebeck",
            "seebeck_uV_per_K",
            ("ZT", "log10_electrical_conductivity", "log10_thermal_conductivity"),
            "thermoelectric",
        ),
        Endpoint(
            "ZT",
            "ZT",
            (
                "seebeck_uV_per_K",
                "log10_electrical_conductivity",
                "log10_thermal_conductivity",
            ),
            "thermoelectric",
        ),
        Endpoint(
            "log10_sigma",
            "log10_electrical_conductivity",
            ("ZT", "seebeck_uV_per_K", "log10_thermal_conductivity"),
            "thermoelectric",
        ),
        Endpoint(
            "log10_kappa",
            "log10_thermal_conductivity",
            ("ZT", "seebeck_uV_per_K", "log10_electrical_conductivity"),
            "thermoelectric",
        ),
    ),
    "PV": (
        Endpoint("FF", "FF_fraction", ("PCE_percent", "Voc_V", "Jsc_mA_cm2"), "pv"),
        Endpoint("Jsc", "Jsc_mA_cm2", ("PCE_percent", "Voc_V", "FF_fraction"), "pv"),
        Endpoint("PCE", "PCE_percent", ("Voc_V", "Jsc_mA_cm2", "FF_fraction"), "pv"),
        Endpoint("Voc", "Voc_V", ("PCE_percent", "Jsc_mA_cm2", "FF_fraction"), "pv"),
    ),
    "LMB": (
        Endpoint(
            "ED_DE_m",
            "energy_density_wh_kg",
            ("discharge_energy_wh",),
            "lmb_de_mass",
        ),
        Endpoint(
            "DE",
            "discharge_energy_wh",
            ("energy_density_wh_kg",),
            "lmb_de_mass",
        ),
        Endpoint(
            "ED_QV_m",
            "energy_density_wh_kg",
            ("discharge_capacity_ah", "normal_discharge_voltage_v"),
            "lmb_vc_mass",
        ),
        Endpoint(
            "Q",
            "discharge_capacity_ah",
            ("energy_density_wh_kg", "normal_discharge_voltage_v"),
            "lmb_vc_mass",
        ),
        Endpoint(
            "V",
            "normal_discharge_voltage_v",
            ("energy_density_wh_kg", "discharge_capacity_ah"),
            "lmb_vc_mass",
        ),
    ),
}


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    spread = float(np.std(y_true, ddof=0))
    return {
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "nRMSE": rmse / spread if spread > 0 else np.nan,
    }


def _feature_pipeline(X: pd.DataFrame, model: object) -> object:
    numeric = list(X.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in X.columns if column not in numeric]
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return make_pipeline(transformer, model)


def _direct_model(X: pd.DataFrame, seed: int, n_jobs: int) -> object:
    return _feature_pipeline(
        X,
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        ),
    )


def _ordinary_models(X: pd.DataFrame, seed: int, n_jobs: int) -> list[object]:
    return [
        _direct_model(X, seed, n_jobs),
        _feature_pipeline(
            X,
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=seed + 1,
                n_jobs=n_jobs,
            ),
        ),
        _feature_pipeline(X, GradientBoostingRegressor(random_state=seed + 2)),
    ]


def _folds(
    n_rows: int,
    *,
    groups: np.ndarray | None,
    seed: int,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_rows)
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        unique = np.unique(groups)
        if len(unique) >= 2:
            count = min(n_splits, len(unique))
            return [
                (np.asarray(train), np.asarray(valid))
                for train, valid in GroupKFold(n_splits=count).split(
                    indices, groups=groups
                )
            ]
    count = min(n_splits, n_rows)
    if count < 2:
        raise ValueError("At least two training rows are required")
    return [
        (np.asarray(train), np.asarray(valid))
        for train, valid in KFold(
            n_splits=count, shuffle=True, random_state=seed
        ).split(indices)
    ]


def _oof_and_test(
    estimator: object,
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    *,
    groups: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_train = np.asarray(y_train, dtype=float)
    oof = np.full(len(y_train), np.nan)
    for train_idx, valid_idx in _folds(
        len(y_train), groups=groups, seed=seed
    ):
        model = clone(estimator)
        model.fit(X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx], y_train[train_idx])
        valid_X = X_train.iloc[valid_idx] if isinstance(X_train, pd.DataFrame) else X_train[valid_idx]
        oof[valid_idx] = np.asarray(model.predict(valid_X), dtype=float)
    fitted = clone(estimator).fit(X_train, y_train)
    test = np.asarray(fitted.predict(X_test), dtype=float)
    return oof, test


def _positive(values: np.ndarray, floor: float = EPS) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), floor)


def _physics(
    dataset: str,
    endpoint: Endpoint,
    predictions: dict[str, np.ndarray],
    context: pd.DataFrame,
    *,
    direct_target: np.ndarray,
    target_range: tuple[float, float],
) -> np.ndarray:
    target = endpoint.target
    if dataset == "SSE":
        T = _positive(context["T_K"].to_numpy())
        if target == "ln_sigma_T":
            result = predictions["ln_sigma0"] - predictions["Ea"] / (
                ARRHENIUS_KB_EV_PER_K * T
            )
        elif target == "Ea":
            result = ARRHENIUS_KB_EV_PER_K * T * (
                predictions["ln_sigma0"] - predictions["ln_sigma_T"]
            )
        else:
            result = predictions["ln_sigma_T"] + predictions["Ea"] / (
                ARRHENIUS_KB_EV_PER_K * T
            )
        return np.asarray(result, dtype=float)

    if dataset == "ESTM":
        T = _positive(context["temperature_K"].to_numpy(), floor=EPS)
        if target == "ZT":
            S = predictions["seebeck_uV_per_K"] * 1e-6
            sigma = np.power(
                10.0,
                np.clip(predictions["log10_electrical_conductivity"], -30.0, 30.0),
            )
            kappa = _positive(
                np.power(
                    10.0,
                    np.clip(predictions["log10_thermal_conductivity"], -30.0, 30.0),
                ),
                floor=ESTM_POSITIVE_FLOOR,
            )
            result = (
                S**2
                * sigma
                * T
                / kappa
            )
            return np.asarray(result, dtype=float)
        if target == "seebeck_uV_per_K":
            sigma = _positive(
                np.power(
                    10.0,
                    np.clip(predictions["log10_electrical_conductivity"], -30.0, 30.0),
                ),
                floor=ESTM_POSITIVE_FLOOR,
            )
            kappa = _positive(
                np.power(
                    10.0,
                    np.clip(predictions["log10_thermal_conductivity"], -30.0, 30.0),
                ),
                floor=ESTM_POSITIVE_FLOOR,
            )
            magnitude = np.sqrt(
                _positive(predictions["ZT"], floor=ESTM_POSITIVE_FLOOR)
                * kappa
                / (sigma * T)
            )
            sign = np.sign(direct_target)
            sign[sign == 0.0] = 1.0
            result = sign * magnitude * 1e6
        elif target == "log10_electrical_conductivity":
            S = predictions["seebeck_uV_per_K"] * 1e-6
            kappa = _positive(
                np.power(
                    10.0,
                    np.clip(predictions["log10_thermal_conductivity"], -30.0, 30.0),
                ),
                floor=ESTM_POSITIVE_FLOOR,
            )
            sigma = (
                _positive(predictions["ZT"], floor=ESTM_POSITIVE_FLOOR)
                * kappa
                / (_positive(S**2, floor=ESTM_POSITIVE_FLOOR) * T)
            )
            result = np.log10(_positive(sigma, floor=ESTM_POSITIVE_FLOOR))
        else:
            S = predictions["seebeck_uV_per_K"] * 1e-6
            sigma = np.power(
                10.0,
                np.clip(predictions["log10_electrical_conductivity"], -30.0, 30.0),
            )
            kappa = (
                S**2
                * sigma
                * T
                / _positive(predictions["ZT"], floor=ESTM_POSITIVE_FLOOR)
            )
            result = np.log10(_positive(kappa, floor=ESTM_POSITIVE_FLOOR))
        return np.clip(np.asarray(result, dtype=float), *target_range)

    if dataset == "PV":
        if target == "PCE_percent":
            return (
                predictions["Voc_V"]
                * predictions["Jsc_mA_cm2"]
                * predictions["FF_fraction"]
            )
        if target == "Voc_V":
            result = _positive(predictions["PCE_percent"]) / (
                _positive(predictions["Jsc_mA_cm2"])
                * _positive(predictions["FF_fraction"])
            )
        elif target == "Jsc_mA_cm2":
            result = _positive(predictions["PCE_percent"]) / (
                _positive(predictions["Voc_V"])
                * _positive(predictions["FF_fraction"])
            )
        else:
            result = _positive(predictions["PCE_percent"]) / (
                _positive(predictions["Voc_V"])
                * _positive(predictions["Jsc_mA_cm2"])
            )
        return np.clip(np.asarray(result, dtype=float), *target_range)

    mass = _positive(context["mass_kg"].to_numpy())
    if endpoint.route == "lmb_de_mass":
        if target == "energy_density_wh_kg":
            return predictions["discharge_energy_wh"] / mass
        return predictions["energy_density_wh_kg"] * mass
    if target == "energy_density_wh_kg":
        return (
            predictions["discharge_capacity_ah"]
            * predictions["normal_discharge_voltage_v"]
            / mass
        )
    if target == "discharge_capacity_ah":
        return (
            predictions["energy_density_wh_kg"]
            * mass
            / _positive(predictions["normal_discharge_voltage_v"])
        )
    return (
        predictions["energy_density_wh_kg"]
        * mass
        / _positive(predictions["discharge_capacity_ah"])
    )


def _propagated_auxiliary_error_proxy(
    dataset: str,
    endpoint: Endpoint,
    predictions: dict[str, np.ndarray],
    context: pd.DataFrame,
    *,
    direct_target: np.ndarray,
    target_range: tuple[float, float],
    auxiliary_rmse: dict[str, float],
) -> np.ndarray:
    """Propagate outer-training OOF auxiliary RMSE by first-order sensitivity."""
    if dataset == "LMB":
        return _lmb_propagated_auxiliary_error_proxy(
            endpoint,
            predictions,
            context,
            auxiliary_rmse,
        )

    base = _physics(
        dataset,
        endpoint,
        predictions,
        context,
        direct_target=direct_target,
        target_range=target_range,
    )
    squared_error = np.zeros(len(direct_target), dtype=float)
    for auxiliary in endpoint.auxiliaries:
        values = np.asarray(predictions[auxiliary], dtype=float)
        step = np.maximum(
            FINITE_DIFFERENCE_ABSOLUTE_STEP,
            FINITE_DIFFERENCE_RELATIVE_STEP * np.abs(values),
        )
        plus = {name: np.asarray(values).copy() for name, values in predictions.items()}
        minus = {name: np.asarray(values).copy() for name, values in predictions.items()}
        plus[auxiliary] += step
        minus[auxiliary] -= step
        upper = _physics(
            dataset,
            endpoint,
            plus,
            context,
            direct_target=direct_target,
            target_range=target_range,
        )
        lower = _physics(
            dataset,
            endpoint,
            minus,
            context,
            direct_target=direct_target,
            target_range=target_range,
        )
        with np.errstate(all="ignore"):
            derivative = (upper - lower) / (2.0 * step)
            one_sided = (upper - base) / step
        nonfinite = ~np.isfinite(derivative)
        derivative[nonfinite] = one_sided[nonfinite]
        derivative[~np.isfinite(derivative)] = 0.0
        sigma = auxiliary_rmse.get(auxiliary, 0.0)
        if not np.isfinite(sigma):
            sigma = 0.0
        squared_error += (derivative * sigma) ** 2
    return np.sqrt(squared_error)


def _lmb_propagated_auxiliary_error_proxy(
    endpoint: Endpoint,
    predictions: dict[str, np.ndarray],
    context: pd.DataFrame,
    auxiliary_rmse: dict[str, float],
) -> np.ndarray:
    """Analytic first-order propagation for the retained LMB equations."""
    mass = _positive(context["mass_kg"].to_numpy())
    target = endpoint.target
    if endpoint.route == "lmb_de_mass":
        if target == "energy_density_wh_kg":
            return np.full_like(mass, auxiliary_rmse["discharge_energy_wh"]) / mass
        return np.full_like(mass, auxiliary_rmse["energy_density_wh_kg"]) * mass

    energy_density = np.asarray(predictions["energy_density_wh_kg"], dtype=float)
    if target == "energy_density_wh_kg":
        voltage = np.asarray(predictions["normal_discharge_voltage_v"], dtype=float)
        capacity = np.asarray(predictions["discharge_capacity_ah"], dtype=float)
        return np.sqrt(
            ((capacity / mass) * auxiliary_rmse["normal_discharge_voltage_v"]) ** 2
            + ((voltage / mass) * auxiliary_rmse["discharge_capacity_ah"]) ** 2
        )
    if target == "discharge_capacity_ah":
        voltage = _positive(predictions["normal_discharge_voltage_v"])
        return np.sqrt(
            ((mass / voltage) * auxiliary_rmse["energy_density_wh_kg"]) ** 2
            + (
                (energy_density * mass / voltage**2)
                * auxiliary_rmse["normal_discharge_voltage_v"]
            )
            ** 2
        )
    capacity = _positive(predictions["discharge_capacity_ah"])
    return np.sqrt(
        ((mass / capacity) * auxiliary_rmse["energy_density_wh_kg"]) ** 2
        + (
            (energy_density * mass / capacity**2)
            * auxiliary_rmse["discharge_capacity_ah"]
        )
        ** 2
    )


def _select_gate(
    *,
    y: np.ndarray,
    direct: np.ndarray,
    residual: np.ndarray,
    reliability: pd.DataFrame,
    groups: np.ndarray | None,
    seed: int,
) -> tuple[PhysicsGate, float]:
    best_alpha = float(RIDGE_ALPHAS[0])
    best_rmse = np.inf
    for alpha in RIDGE_ALPHAS:
        prediction = np.full(len(y), np.nan)
        for train_idx, valid_idx in _folds(
            len(y), groups=groups, seed=seed
        ):
            gate = PhysicsGate(alpha=float(alpha)).fit(
                y_oof=y[train_idx],
                direct_oof=direct[train_idx],
                residual_oof=residual[train_idx],
                reliability_oof=reliability.iloc[train_idx],
            )
            prediction[valid_idx] = gate.predict(
                direct=direct[valid_idx],
                residual=residual[valid_idx],
                reliability=reliability.iloc[valid_idx],
            )
        score = float(np.sqrt(mean_squared_error(y, prediction)))
        if score < best_rmse:
            best_rmse = score
            best_alpha = float(alpha)
    gate = PhysicsGate(alpha=best_alpha).fit(
        y_oof=y,
        direct_oof=direct,
        residual_oof=residual,
        reliability_oof=reliability,
    )
    return gate, best_alpha


def _outer_splits(
    dataset: str,
    frame: pd.DataFrame,
    *,
    group_column: str | None,
    split_id: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    indices = np.arange(len(frame))
    if dataset == "ESTM":
        splitter = KFold(n_splits=3, shuffle=True, random_state=split_id)
        return [
            (fold, np.asarray(train), np.asarray(test))
            for fold, (train, test) in enumerate(splitter.split(indices))
        ]
    if dataset in {"SSE", "LMB"}:
        if group_column is None:
            raise ValueError(f"{dataset} requires a group column")
        test_size = 0.25 if dataset == "SSE" else 0.10
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=split_id
        )
        train, test = next(
            splitter.split(indices, groups=frame[group_column].astype(str))
        )
        return [(0, np.asarray(train), np.asarray(test))]
    train, test = train_test_split(
        indices, test_size=0.30, random_state=split_id, shuffle=True
    )
    return [(0, np.asarray(train), np.asarray(test))]


def _evaluate_endpoint(
    dataset: str,
    endpoint: Endpoint,
    frame: pd.DataFrame,
    *,
    features: list[str],
    context_columns: list[str],
    group_column: str | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_id: int,
    fold_id: int,
    n_jobs: int,
) -> list[dict[str, object]]:
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    required = [endpoint.target, *endpoint.auxiliaries, *context_columns]
    train = train.dropna(subset=required).reset_index(drop=True)
    test = test.dropna(subset=required).reset_index(drop=True)
    if len(train) < 20 or len(test) < 2:
        raise ValueError(f"Insufficient complete rows for {dataset}/{endpoint.code}")
    X_train = train[features]
    X_test = test[features]
    groups = (
        train[group_column].astype(str).to_numpy()
        if group_column is not None
        else None
    )
    y_train = train[endpoint.target].to_numpy(dtype=float)
    y_test = test[endpoint.target].to_numpy(dtype=float)
    target_range = (float(np.min(y_train)), float(np.max(y_train)))

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for offset, target in enumerate((endpoint.target, *endpoint.auxiliaries)):
        model = _direct_model(X_train, split_id + fold_id * 101 + offset, n_jobs)
        oof, held_out = _oof_and_test(
            model,
            X_train,
            train[target].to_numpy(dtype=float),
            X_test,
            groups=groups,
            seed=split_id + offset,
        )
        predictions[target] = {"oof": oof, "test": held_out}
    direct_oof = predictions[endpoint.target]["oof"]
    direct_test = predictions[endpoint.target]["test"]
    aux_oof = {name: predictions[name]["oof"] for name in endpoint.auxiliaries}
    aux_test = {name: predictions[name]["test"] for name in endpoint.auxiliaries}
    context_train = train[context_columns] if context_columns else pd.DataFrame(index=train.index)
    context_test = test[context_columns] if context_columns else pd.DataFrame(index=test.index)
    physics_oof = _physics(
        dataset,
        endpoint,
        aux_oof,
        context_train,
        direct_target=direct_oof,
        target_range=target_range,
    )
    physics_test = _physics(
        dataset,
        endpoint,
        aux_test,
        context_test,
        direct_target=direct_test,
        target_range=target_range,
    )

    residual_X_train = np.column_stack(
        [direct_oof, physics_oof, *[aux_oof[name] for name in endpoint.auxiliaries]]
    )
    residual_X_test = np.column_stack(
        [direct_test, physics_test, *[aux_test[name] for name in endpoint.auxiliaries]]
    )
    residual_delta_oof, residual_delta_test = _oof_and_test(
        make_pipeline(
            StandardScaler(),
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=split_id + 700,
                n_jobs=n_jobs,
            ),
        ),
        residual_X_train,
        y_train - physics_oof,
        residual_X_test,
        groups=groups,
        seed=split_id + 701,
    )
    residual_oof = physics_oof + residual_delta_oof
    residual_test = physics_test + residual_delta_test

    ordinary_train: list[np.ndarray] = []
    ordinary_test: list[np.ndarray] = []
    for offset, model in enumerate(_ordinary_models(X_train, split_id + 900, n_jobs)):
        oof, held_out = _oof_and_test(
            model,
            X_train,
            y_train,
            X_test,
            groups=groups,
            seed=split_id + 901 + offset,
        )
        ordinary_train.append(oof)
        ordinary_test.append(held_out)
    ordinary_meta = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    ordinary_meta.fit(np.column_stack(ordinary_train), y_train)
    ordinary_prediction = ordinary_meta.predict(np.column_stack(ordinary_test))

    physres_meta = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    physres_meta.fit(
        np.column_stack([direct_oof, physics_oof, residual_oof]), y_train
    )
    physres_prediction = physres_meta.predict(
        np.column_stack([direct_test, physics_test, residual_test])
    )

    auxiliary_rmse = {
        name: float(
            np.sqrt(
                mean_squared_error(
                    train[name].to_numpy(dtype=float), predictions[name]["oof"]
                )
            )
        )
        for name in endpoint.auxiliaries
    }
    propagated_error_oof = _propagated_auxiliary_error_proxy(
        dataset,
        endpoint,
        aux_oof,
        context_train,
        direct_target=direct_oof,
        target_range=target_range,
        auxiliary_rmse=auxiliary_rmse,
    )
    propagated_error_test = _propagated_auxiliary_error_proxy(
        dataset,
        endpoint,
        aux_test,
        context_test,
        direct_target=direct_test,
        target_range=target_range,
        auxiliary_rmse=auxiliary_rmse,
    )
    reliability_oof = reliability_features(
        direct=direct_oof,
        physics=physics_oof,
        residual=residual_oof,
        training_targets=y_train,
        propagated_auxiliary_error=propagated_error_oof,
    )
    reliability_test = reliability_features(
        direct=direct_test,
        physics=physics_test,
        residual=residual_test,
        training_targets=y_train,
        propagated_auxiliary_error=propagated_error_test,
    )
    gate, gate_alpha = _select_gate(
        y=y_train,
        direct=direct_oof,
        residual=residual_oof,
        reliability=reliability_oof,
        groups=groups,
        seed=split_id + 1200,
    )
    gate_prediction = gate.predict(
        direct=direct_test,
        residual=residual_test,
        reliability=reliability_test,
    )

    method_predictions = {
        "Direct": direct_test,
        "Physics-only": physics_test,
        "Residual-only": residual_test,
        "Ordinary stacking": ordinary_prediction,
        "PhysResStack": physres_prediction,
        "PhysicsGate": gate_prediction,
    }
    rows: list[dict[str, object]] = []
    direct_rmse = metric_dict(y_test, direct_test)["RMSE"]
    for method in METHODS:
        prediction = np.asarray(method_predictions[method], dtype=float)
        metrics = metric_dict(y_test, prediction)
        rows.append(
            {
                "dataset": dataset,
                "endpoint": endpoint.code,
                "target": endpoint.target,
                "route": endpoint.route,
                "split_id": split_id,
                "fold_id": fold_id,
                "method": method,
                "n_train": len(train),
                "n_test": len(test),
                **metrics,
                "physics_residual": float(np.mean(np.abs(prediction - physics_test))),
                "percent_RMSE_change_vs_direct": 100.0
                * (metrics["RMSE"] - direct_rmse)
                / direct_rmse,
                "selected_gate_alpha": gate_alpha if method == "PhysicsGate" else np.nan,
                "uses_true_test_auxiliary_values": False,
                "uses_oof_second_stage_predictions": True,
            }
        )
    return rows


def run_benchmark(
    dataset: str,
    processed_path: str | Path,
    *,
    split_ids: Iterable[int],
    output_path: str | Path,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run all retained endpoints and six methods for one dataset."""
    dataset = dataset.upper()
    if dataset not in ENDPOINTS:
        raise ValueError(f"Unknown dataset: {dataset}")
    processed_path = Path(processed_path)
    schema_path = processed_path.with_suffix(".schema.json")
    frame = pd.read_csv(processed_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema["dataset"].upper() != dataset:
        raise ValueError("Processed schema dataset does not match requested dataset")
    features = list(schema["feature_columns"])
    contexts = list(schema["context_columns"])
    group_column = schema["group_column"]
    rows: list[dict[str, object]] = []
    for split_id in split_ids:
        for fold_id, train_idx, test_idx in _outer_splits(
            dataset,
            frame,
            group_column=group_column,
            split_id=int(split_id),
        ):
            for endpoint in ENDPOINTS[dataset]:
                rows.extend(
                    _evaluate_endpoint(
                        dataset,
                        endpoint,
                        frame,
                        features=features,
                        context_columns=contexts,
                        group_column=group_column,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        split_id=int(split_id),
                        fold_id=fold_id,
                        n_jobs=n_jobs,
                    )
                )
    result = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


__all__ = ["ENDPOINTS", "METHODS", "run_benchmark"]
