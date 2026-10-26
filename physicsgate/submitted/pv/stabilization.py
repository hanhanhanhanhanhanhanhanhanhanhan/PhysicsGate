"""Train-range stabilization helpers and the stabilization-sensitivity run (Supplementary Figure S3).

Submitted-protocol code. Definitions are copied verbatim from
``figma5/scripts/run_si_stabilization_sensitivity.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR, PV_WORKDIR  # noqa: E402
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = PV_WORKDIR

REPO = WORKDIR

OUTPUT = ROOT / "outputs" / "si_stabilization_sensitivity"

SEED_PLAN = ROOT / "outputs" / "figure4_gate_mechanism" / "seed_plans"

PV_FIGURE4_DIR = ROOT / "outputs" / "figure4_gate_mechanism" / "50seed" / "pv"

PV_BASE_SELECTION = PV_FIGURE4_DIR / "Figure4_PV_50Seed_Rerun_BaseSelection.csv"

PV_META_SELECTION = PV_FIGURE4_DIR / "Figure4_PV_50Seed_Rerun_Selection.csv"

from physicsgate.submitted.estm import stability as estm_exp  # noqa: E402
from physicsgate.submitted.pv import selection as pv_exp  # noqa: E402
from physicsgate.submitted.common.propagated_error import global_auxiliary_sigmas, propagated_sigma_finite_difference  # noqa: E402
from physicsgate.submitted.common.gate_features import reliability_features as pv_reliability_features  # noqa: E402
from physicsgate.submitted.pv.common import FEATURE_COLUMNS as PV_FEATURE_COLUMNS, TARGET_COLUMNS as PV_TARGET_COLUMNS  # noqa: E402

ESTM_VARIANTS = ("retained_train_range", "numeric_only_no_train_range")

PV_VARIANTS = ("retained_numeric_only", "matched_train_range")

GATE_METHOD = "PhysicsGate"


def selected_pv_models() -> tuple[dict[tuple[int, str], str], dict[tuple[int, str], str]]:
    base = pd.read_csv(PV_BASE_SELECTION)
    base = base.loc[base["selected"].astype(str).str.lower().eq("true")]
    base_map = {
        (int(row.seed), str(row.target)): str(row.candidate_model)
        for row in base.itertuples(index=False)
    }
    meta = pd.read_csv(PV_META_SELECTION)
    meta = meta.loc[
        meta["selected"].astype(str).str.lower().eq("true")
        & meta["selection_stage"].astype(str).eq("Ares_old_h")
    ]
    meta_map = {
        (int(row.seed), str(row.target_code)): str(row.candidate_meta_learner)
        for row in meta.itertuples(index=False)
    }
    return base_map, meta_map


def as_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def rmse(y_true: Any, y_pred: Any) -> float:
    truth = as_array(y_true)
    pred = as_array(y_pred)
    return float(np.sqrt(np.mean(np.square(truth - pred))))


def metric_row(
    *,
    dataset: str,
    target: str,
    target_code: str,
    seed: int,
    fold_id: int,
    variant: str,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    direct_rmse: float,
    selected_alpha: float | None = None,
    mean_weight: float | None = None,
) -> dict[str, Any]:
    metric_fn = estm_exp.metric_dict if dataset == "ESTM" else pv_exp.paper.metric_dict
    row: dict[str, Any] = {
        "dataset": dataset,
        "target": target,
        "target_code": target_code,
        "seed": int(seed),
        "fold_id": int(fold_id),
        "stabilization_variant": variant,
        "method": method,
        "selected_gate_alpha": selected_alpha,
        "mean_test_weight": mean_weight,
        "n_test": int(len(y_true)),
        "uses_true_test_auxiliary_values": False,
        "uses_test_target_for_selection": False,
    }
    row.update(metric_fn(y_true, y_pred))
    row["direct_RMSE"] = direct_rmse
    row["percent_RMSE_change_vs_direct"] = (
        (float(row["RMSE"]) / direct_rmse - 1.0) * 100.0
    )
    return row


def train_range_clip(
    target: str,
    values: np.ndarray,
    y_train: np.ndarray,
    *,
    forward_target: str,
) -> np.ndarray:
    result = as_array(values)
    if target == forward_target:
        return result
    finite = as_array(y_train)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return result
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower >= upper:
        return result
    return np.clip(result, lower, upper)


def clipping_audit(
    raw: np.ndarray,
    stabilized: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, float]:
    raw_arr = as_array(raw)
    stabilized_arr = as_array(stabilized)
    train_arr = as_array(y_train)
    return {
        "train_target_min": float(np.min(train_arr)),
        "train_target_max": float(np.max(train_arr)),
        "n_predictions": int(len(raw_arr)),
        "n_changed_by_stabilization": int(
            np.count_nonzero(~np.isclose(raw_arr, stabilized_arr, rtol=0.0, atol=1e-12))
        ),
        "fraction_changed_by_stabilization": float(
            np.mean(~np.isclose(raw_arr, stabilized_arr, rtol=0.0, atol=1e-12))
        ),
        "raw_prediction_min": float(np.min(raw_arr)),
        "raw_prediction_max": float(np.max(raw_arr)),
        "stabilized_prediction_min": float(np.min(stabilized_arr)),
        "stabilized_prediction_max": float(np.max(stabilized_arr)),
    }


def estm_phi(
    *,
    target: str,
    aux_columns: list[str],
    y_train: np.ndarray,
    use_train_range: bool,
) -> Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray]:
    def phi(values: np.ndarray, known: dict[str, np.ndarray]) -> np.ndarray:
        predictions = {
            column: values[:, index] for index, column in enumerate(aux_columns)
        }
        raw = estm_exp.estm.compute_estm_physics_prediction(
            target,
            predictions,
            known["temperature_K"],
            direct_target_prediction=known["direct_target_prediction"],
        )
        if use_train_range:
            return train_range_clip(target, raw, y_train, forward_target="ZT")
        return as_array(raw)

    return phi


def run_estm_seed(seed: int, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = estm_exp.estm.complete_estm_frame(
        pd.read_csv(estm_exp.DEFAULT_INPUT_PATH)
    )
    X, _ = estm_exp.build_feature_matrix(frame.drop(columns=["core_v2_row_index"]))
    y_all = {
        column: frame[column].to_numpy(dtype=float)
        for column in estm_exp.TARGET_COLUMNS
    }
    temperature = frame["temperature_K"].to_numpy(dtype=float)
    folds = estm_exp.split_indices(
        frame,
        split_mode="paper_interpolation",
        n_splits=args.estm_n_splits,
        random_state=seed,
    )
    alpha_grid = [
        0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
        1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0,
    ]
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    model_args = SimpleNamespace(
        n_jobs=args.n_jobs,
        base_model="xgb_fixed",
        inner_cv=args.inner_cv,
        gate_inner_cv=args.gate_inner_cv,
    )

    for target in estm_exp.TARGET_COLUMNS:
        aux_columns = [column for column in estm_exp.TARGET_COLUMNS if column != target]
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y_all[target][train_idx]
            y_test = y_all[target][test_idx]
            t_train = temperature[train_idx]
            t_test = temperature[test_idx]
            fold_seed = int(seed + fold_id * 101)

            base_predictions: dict[str, dict[str, np.ndarray]] = {}
            for target_index, column in enumerate(estm_exp.TARGET_COLUMNS):
                model_seed = seed + fold_id * 101 + target_index * 13
                estimator, _, _ = estm_exp.build_base_tree_estimator(
                    random_state=model_seed,
                    n_jobs=model_args.n_jobs,
                    base_model=model_args.base_model,
                )
                oof, test = estm_exp.oof_and_test_predictions(
                    estimator,
                    X_train,
                    y_all[column][train_idx],
                    X_test,
                    n_splits=model_args.inner_cv,
                    random_state=model_seed,
                )
                base_predictions[column] = {
                    "oof": as_array(oof),
                    "test": as_array(test),
                }

            direct_oof = base_predictions[target]["oof"]
            direct_test = base_predictions[target]["test"]
            aux_oof = {
                column: base_predictions[column]["oof"] for column in aux_columns
            }
            aux_test = {
                column: base_predictions[column]["test"] for column in aux_columns
            }
            raw_phys_oof = as_array(
                estm_exp.estm.compute_estm_physics_prediction(
                    target,
                    aux_oof,
                    t_train,
                    direct_target_prediction=direct_oof,
                )
            )
            raw_phys_test = as_array(
                estm_exp.estm.compute_estm_physics_prediction(
                    target,
                    aux_test,
                    t_test,
                    direct_target_prediction=direct_test,
                )
            )
            direct_value = rmse(y_test, direct_test)

            for variant in ESTM_VARIANTS:
                use_range = variant == "retained_train_range"
                phys_oof = (
                    train_range_clip(target, raw_phys_oof, y_train, forward_target="ZT")
                    if use_range
                    else raw_phys_oof.copy()
                )
                phys_test = (
                    train_range_clip(target, raw_phys_test, y_train, forward_target="ZT")
                    if use_range
                    else raw_phys_test.copy()
                )

                residual_features_oof = np.column_stack(
                    [
                        X_train,
                        direct_oof,
                        phys_oof,
                        *[aux_oof[column] for column in aux_columns],
                        t_train,
                    ]
                )
                residual_features_test = np.column_stack(
                    [
                        X_test,
                        direct_test,
                        phys_test,
                        *[aux_test[column] for column in aux_columns],
                        t_test,
                    ]
                )
                residual_seed = seed + fold_id * 101 + 71
                residual_estimator, _, _ = estm_exp.build_base_tree_estimator(
                    random_state=residual_seed,
                    n_jobs=model_args.n_jobs,
                    base_model=model_args.base_model,
                )
                delta_oof, delta_test = estm_exp.oof_and_test_predictions(
                    residual_estimator,
                    residual_features_oof,
                    y_train - phys_oof,
                    residual_features_test,
                    n_splits=model_args.inner_cv,
                    random_state=residual_seed,
                )
                residual_oof = phys_oof + as_array(delta_oof)
                residual_test = phys_test + as_array(delta_test)

                aux_train_arr = np.column_stack(
                    [aux_oof[column] for column in aux_columns]
                )
                aux_test_arr = np.column_stack(
                    [aux_test[column] for column in aux_columns]
                )
                aux_true_train = np.column_stack(
                    [y_all[column][train_idx] for column in aux_columns]
                )
                aux_sigmas, _ = global_auxiliary_sigmas(
                    aux_pred_train=aux_train_arr,
                    aux_true_train=aux_true_train,
                    aux_names=aux_columns,
                )
                phi = estm_phi(
                    target=target,
                    aux_columns=aux_columns,
                    y_train=y_train,
                    use_train_range=use_range,
                )
                known_train = {
                    "temperature_K": t_train,
                    "direct_target_prediction": direct_oof,
                }
                known_test = {
                    "temperature_K": t_test,
                    "direct_target_prediction": direct_test,
                }
                sigma_oof, _ = propagated_sigma_finite_difference(
                    aux_pred=aux_train_arr,
                    known=known_train,
                    aux_sigmas=aux_sigmas,
                    phi=phi,
                )
                sigma_test, _ = propagated_sigma_finite_difference(
                    aux_pred=aux_test_arr,
                    known=known_test,
                    aux_sigmas=aux_sigmas,
                    phi=phi,
                )
                rel_oof = estm_exp.reliability_features(
                    direct=direct_oof,
                    physics=phys_oof,
                    residual=residual_oof,
                    y_train=y_train,
                    r4=None,
                    sigma_phys=sigma_oof,
                )
                rel_test = estm_exp.reliability_features(
                    direct=direct_test,
                    physics=phys_test,
                    residual=residual_test,
                    y_train=y_train,
                    r4=None,
                    sigma_phys=sigma_test,
                )
                groups = frame["formula_id"].astype(str).to_numpy()[train_idx]
                gate_test, selected, _ = estm_exp.tune_r2_r5_r6_gate(
                    y_train=y_train,
                    direct_train=direct_oof,
                    direct_test=direct_test,
                    residual_train=residual_oof,
                    residual_test=residual_test,
                    reliability_train=rel_oof,
                    reliability_test=rel_test,
                    groups=groups,
                    seed=fold_seed + 17017,
                    inner_cv=model_args.gate_inner_cv,
                    alpha_grid=alpha_grid,
                )

                display = estm_exp.TARGET_DISPLAY[target]
                for method, prediction in (
                    ("Direct", direct_test),
                    ("Physics-only", phys_test),
                    ("Ares_old", residual_test),
                    (GATE_METHOD, gate_test),
                ):
                    rows.append(
                        metric_row(
                            dataset="ESTM",
                            target=display,
                            target_code=target,
                            seed=seed,
                            fold_id=fold_id,
                            variant=variant,
                            method=method,
                            y_true=y_test,
                            y_pred=prediction,
                            direct_rmse=direct_value,
                            selected_alpha=(
                                float(selected["alpha"])
                                if method == GATE_METHOD
                                else None
                            ),
                            mean_weight=(
                                float(selected["mean_test_weight"])
                                if method == GATE_METHOD
                                else None
                            ),
                        )
                    )
                audit = clipping_audit(raw_phys_test, phys_test, y_train)
                audit.update(
                    {
                        "dataset": "ESTM",
                        "target_code": target,
                        "seed": seed,
                        "fold_id": fold_id,
                        "stabilization_variant": variant,
                        "forward_target": target == "ZT",
                        "uses_outer_train_target_range": use_range and target != "ZT",
                        "uses_true_test_auxiliary_values": False,
                        "uses_test_target_for_selection": False,
                    }
                )
                audits.append(audit)
            print(
                f"[ESTM] seed={seed} target={target} fold={fold_id}",
                flush=True,
            )
    return rows, audits


def pv_phi(
    *,
    target: str,
    aux_columns: list[str],
    y_train: np.ndarray,
    use_train_range: bool,
) -> Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray]:
    def phi(values: np.ndarray, _known: dict[str, np.ndarray]) -> np.ndarray:
        predictions = {
            column: values[:, index] for index, column in enumerate(aux_columns)
        }
        raw = pv_exp.pvmain.compute_pv_physics_prediction(target, predictions)
        if use_train_range:
            return train_range_clip(
                target, raw, y_train, forward_target="PCE_percent"
            )
        return as_array(raw)

    return phi


def fit_pv_base_once(
    *,
    data: pd.DataFrame,
    target: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    random_state: int,
    n_jobs: int,
    inner_cv: int,
    selected_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay one selected candidate without duplicating its OOF pass."""
    candidates = pv_exp.pvgated.base_candidate_factories(
        target,
        random_state=random_state,
        n_jobs=n_jobs,
        rf_n_estimators=None,
    )
    retained = [candidate for candidate in candidates if candidate[0] == selected_name]
    if len(retained) != 1:
        raise RuntimeError(
            f"Missing unique retained PV base candidate {selected_name!r} "
            f"for {target}"
        )
    _, factory, _ = retained[0]
    X_raw = data[PV_FEATURE_COLUMNS]
    y = data[target].astype(float)
    X_train_raw = X_raw.iloc[train_idx]
    X_test_raw = X_raw.iloc[test_idx]
    folds = KFold(
        n_splits=min(inner_cv, len(train_idx)),
        shuffle=True,
        random_state=random_state,
    )
    oof = np.full(len(train_idx), np.nan, dtype=float)
    for fold_id, (fit_pos, valid_pos) in enumerate(folds.split(X_train_raw)):
        fit_index = train_idx[fit_pos]
        valid_index = train_idx[valid_pos]
        X_fit, X_valid, _, _ = pv_exp.pvgated.encode_features_for_target(
            X_raw.iloc[fit_index],
            X_raw.iloc[valid_index],
            y.iloc[fit_index],
            protocol="leakage_safe",
        )
        model = factory()
        try:
            model.set_params(random_state=random_state + fold_id * 17)
        except (AttributeError, ValueError):
            pass
        model.fit(X_fit, y.iloc[fit_index])
        oof[valid_pos] = as_array(model.predict(X_valid))
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f"Incomplete OOF prediction for {target}")
    X_train, X_test, _, _ = pv_exp.pvgated.encode_features_for_target(
        X_train_raw,
        X_test_raw,
        y.iloc[train_idx],
        protocol="leakage_safe",
    )
    full_model = factory()
    full_model.fit(X_train, y.iloc[train_idx])
    test = as_array(full_model.predict(X_test))
    return oof, test


def run_pv_seed(seed: int, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.read_csv(pv_exp.DEFAULT_OUTPUT)
    data = (
        frame.dropna(subset=[*PV_TARGET_COLUMNS, *PV_FEATURE_COLUMNS])
        .copy()
        .reset_index(drop=True)
    )
    train_idx, test_idx = pv_exp.pvgated.split_indices(
        len(data),
        test_size=args.pv_test_size,
        random_state=seed,
    )
    base_model_map, meta_model_map = selected_pv_models()
    base_oof: dict[str, np.ndarray] = {}
    base_test: dict[str, np.ndarray] = {}
    for index, target in enumerate(PV_TARGET_COLUMNS):
        oof, test = fit_pv_base_once(
            data=data,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            random_state=seed + index * 31,
            n_jobs=args.n_jobs,
            inner_cv=args.inner_cv,
            selected_name=base_model_map[(seed, target)],
        )
        base_oof[target] = as_array(oof)
        base_test[target] = as_array(test)

    alpha_grid = np.logspace(-8.0, 8.0, 73)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for target_index, target in enumerate(PV_TARGET_COLUMNS):
        y_all = data[target].to_numpy(dtype=float)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        aux_columns = [column for column in PV_TARGET_COLUMNS if column != target]
        aux_oof = {column: base_oof[column] for column in aux_columns}
        aux_test = {column: base_test[column] for column in aux_columns}
        direct_oof = base_oof[target]
        direct_test = base_test[target]
        raw_phys_oof = as_array(
            pv_exp.pvmain.compute_pv_physics_prediction(target, aux_oof)
        )
        raw_phys_test = as_array(
            pv_exp.pvmain.compute_pv_physics_prediction(target, aux_test)
        )
        direct_value = rmse(y_test, direct_test)
        selected_meta_name = meta_model_map[(seed, target)]
        original_meta_factories = pv_exp.pvgated.meta_candidate_factories

        def retained_meta_factories(random_state: int, n_jobs: int):
            candidates = original_meta_factories(random_state, n_jobs)
            if selected_meta_name not in candidates:
                raise RuntimeError(
                    f"Missing retained PV residual candidate "
                    f"{selected_meta_name!r} for {(seed, target)}"
                )
            return {selected_meta_name: candidates[selected_meta_name]}

        pv_exp.pvgated.meta_candidate_factories = retained_meta_factories
        try:
            for variant in PV_VARIANTS:
                use_range = variant == "matched_train_range"
                phys_oof = (
                    train_range_clip(
                        target, raw_phys_oof, y_train, forward_target="PCE_percent"
                    )
                    if use_range
                    else raw_phys_oof.copy()
                )
                phys_test = (
                    train_range_clip(
                        target, raw_phys_test, y_train, forward_target="PCE_percent"
                    )
                    if use_range
                    else raw_phys_test.copy()
                )
                feature_oof = np.column_stack(
                    [direct_oof, phys_oof, *[aux_oof[column] for column in aux_columns]]
                )
                feature_test = np.column_stack(
                    [direct_test, phys_test, *[aux_test[column] for column in aux_columns]]
                )
                meta_rows: list[dict[str, Any]] = []
                delta_oof, delta_test, _, _ = pv_exp.paper.select_meta(
                    feature_oof,
                    y_train - phys_oof,
                    feature_test,
                    y_test=y_test - phys_test,
                    target=target,
                    role=f"{target}_{variant}_Ares_old_h",
                    random_state=seed + 1001 + target_index * 37,
                    n_jobs=args.n_jobs,
                    inner_cv=args.inner_cv,
                    meta_rows=meta_rows,
                )
                residual_oof = phys_oof + as_array(delta_oof)
                residual_test = phys_test + as_array(delta_test)

                aux_matrix = np.column_stack(
                    [aux_oof[column] for column in aux_columns]
                )
                aux_test_matrix = np.column_stack(
                    [aux_test[column] for column in aux_columns]
                )
                aux_true = np.column_stack(
                    [
                        data[column].to_numpy(dtype=float)[train_idx]
                        for column in aux_columns
                    ]
                )
                aux_sigmas, _ = global_auxiliary_sigmas(
                    aux_pred_train=aux_matrix,
                    aux_true_train=aux_true,
                    aux_names=aux_columns,
                )
                phi = pv_phi(
                    target=target,
                    aux_columns=aux_columns,
                    y_train=y_train,
                    use_train_range=use_range,
                )
                sigma_oof, _ = propagated_sigma_finite_difference(
                    aux_pred=aux_matrix,
                    known={},
                    aux_sigmas=aux_sigmas,
                    phi=phi,
                )
                sigma_test, _ = propagated_sigma_finite_difference(
                    aux_pred=aux_test_matrix,
                    known={},
                    aux_sigmas=aux_sigmas,
                    phi=phi,
                )
                rel_oof = pv_reliability_features(
                    direct=direct_oof,
                    physics=phys_oof,
                    residual=residual_oof,
                    y_train=y_train,
                    propagated_uncertainty=sigma_oof,
                )[pv_exp.FEATURES]
                rel_test = pv_reliability_features(
                    direct=direct_test,
                    physics=phys_test,
                    residual=residual_test,
                    y_train=y_train,
                    propagated_uncertainty=sigma_test,
                )[pv_exp.FEATURES]
                gate_test, alpha, mean_weight, _ = pv_exp.tune_gate(
                    y=y_train,
                    direct_oof=direct_oof,
                    direct_test=direct_test,
                    residual_oof=residual_oof,
                    residual_test=residual_test,
                    rel_oof=rel_oof,
                    rel_test=rel_test,
                    seed=seed + 9001 + target_index * 41,
                    alphas=alpha_grid,
                )

                display = pv_exp.pvmain.TARGET_DISPLAY[target]
                for method, prediction in (
                    ("Direct", direct_test),
                    ("Physics-only", phys_test),
                    ("Ares_old", residual_test),
                    (GATE_METHOD, gate_test),
                ):
                    rows.append(
                        metric_row(
                            dataset="PV",
                            target=display,
                            target_code=target,
                            seed=seed,
                            fold_id=0,
                            variant=variant,
                            method=method,
                            y_true=y_test,
                            y_pred=prediction,
                            direct_rmse=direct_value,
                            selected_alpha=alpha if method == GATE_METHOD else None,
                            mean_weight=mean_weight if method == GATE_METHOD else None,
                        )
                    )
                audit = clipping_audit(raw_phys_test, phys_test, y_train)
                audit.update(
                    {
                        "dataset": "PV",
                        "target_code": target,
                        "seed": seed,
                        "fold_id": 0,
                        "stabilization_variant": variant,
                        "forward_target": target == "PCE_percent",
                        "uses_outer_train_target_range": (
                            use_range and target != "PCE_percent"
                        ),
                        "retained_base_model": base_model_map[(seed, target)],
                        "retained_residual_model": selected_meta_name,
                        "uses_true_test_auxiliary_values": False,
                        "uses_test_target_for_selection": False,
                    }
                )
                audits.append(audit)
        finally:
            pv_exp.pvgated.meta_candidate_factories = original_meta_factories
        print(f"[PV] seed={seed} target={target}", flush=True)
    return rows, audits


def read_seed_list(dataset: str) -> list[int]:
    path = SEED_PLAN / f"Figure4_{dataset}_50Seed_List.txt"
    text = path.read_text(encoding="utf-8-sig").strip()
    return [
        int(value.strip())
        for value in text.replace("\n", ",").split(",")
        if value.strip()
    ]


def checkpoint(
    dataset: str,
    rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        OUTPUT / f"SI_Stabilization_{dataset}_Detailed.csv", index=False
    )
    pd.DataFrame(audits).to_csv(
        OUTPUT / f"SI_Stabilization_{dataset}_Audit.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ESTM", "PV"), required=True)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--gate-inner-cv", type=int, default=5)
    parser.add_argument("--estm-n-splits", type=int, default=3)
    parser.add_argument("--pv-test-size", type=float, default=0.30)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else read_seed_list(args.dataset)
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    detail_path = OUTPUT / f"SI_Stabilization_{args.dataset}_Detailed.csv"
    audit_path = OUTPUT / f"SI_Stabilization_{args.dataset}_Audit.csv"
    if args.resume and detail_path.exists():
        rows = pd.read_csv(detail_path).to_dict("records")
        audits = (
            pd.read_csv(audit_path).to_dict("records") if audit_path.exists() else []
        )
    else:
        rows = []
        audits = []
    completed = {
        int(seed)
        for seed, group in pd.DataFrame(rows).groupby("seed")
        if len(group) == (
            len(estm_exp.TARGET_COLUMNS) * args.estm_n_splits * len(ESTM_VARIANTS) * 4
            if args.dataset == "ESTM"
            else len(PV_TARGET_COLUMNS) * len(PV_VARIANTS) * 4
        )
    } if rows else set()

    runner = run_estm_seed if args.dataset == "ESTM" else run_pv_seed
    for position, seed in enumerate(seeds, start=1):
        if seed in completed:
            print(
                f"[{args.dataset} {position}/{len(seeds)}] seed={seed} complete",
                flush=True,
            )
            continue
        seed_rows, seed_audits = runner(seed, args)
        rows.extend(seed_rows)
        audits.extend(seed_audits)
        checkpoint(args.dataset, rows, audits)
        print(
            f"[{args.dataset} {position}/{len(seeds)}] seed={seed} checkpointed",
            flush=True,
        )
    print(f"Wrote paired sensitivity outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
