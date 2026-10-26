"""ESTM fixed-gate evaluation used by the repeated-split analysis.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/tmp_estm_fixed_gate_multiseed_stability.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

from physicsgate.submitted.estm import gate as estmgate  # noqa: E402
from physicsgate.submitted.estm import physics as estm  # noqa: E402
from physicsgate.submitted.common.propagated_error import global_auxiliary_sigmas, propagated_sigma_finite_difference  # noqa: E402
from physicsgate.submitted.estm.base_models import build_base_tree_estimator, oof_and_test_predictions  # noqa: E402
from physicsgate.submitted.common.gate_fit import metric_dict, reliability_features  # noqa: E402
from physicsgate.submitted.lmb.gate_features import METHOD_R2_R5_R6, tune_r2_r5_r6_gate  # noqa: E402
from physicsgate.submitted.estm.features import DEFAULT_INPUT_PATH, build_feature_matrix, split_indices  # noqa: E402

TARGET_COLUMNS = estm.TARGET_COLUMNS

TARGET_DISPLAY = estm.TARGET_DISPLAY


def evaluate_fold(
    *,
    data: pd.DataFrame,
    X: np.ndarray,
    target: str,
    outer_seed: int,
    fold_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    alpha_grid: list[float],
    selection_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fully independently rebuild all branches + the fixed gate for one (seed, target, fold)."""
    y_all = {column: data[column].to_numpy(dtype=float) for column in TARGET_COLUMNS}
    temperature = data["temperature_K"].to_numpy(dtype=float)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = y_all[target][train_idx]
    y_test = y_all[target][test_idx]
    t_train = temperature[train_idx]
    t_test = temperature[test_idx]
    unit_id = f"ESTM_{target}_seed_{outer_seed}_fold_{fold_id}"
    fold_seed = int(outer_seed + fold_id * 101)
    aux_columns = [column for column in TARGET_COLUMNS if column != target]

    # --- Direct branch (and auxiliary target branches, needed for the physics equation) ---
    base_predictions: dict[str, dict[str, np.ndarray]] = {}
    for target_index, column in enumerate(TARGET_COLUMNS):
        estimator, _selected_model, _selected_params = build_base_tree_estimator(
            random_state=outer_seed + fold_id * 101 + target_index * 13,
            n_jobs=args.n_jobs,
            base_model=args.base_model,
        )
        oof, test = oof_and_test_predictions(
            estimator,
            X_train,
            y_all[column][train_idx],
            X_test,
            n_splits=args.inner_cv,
            random_state=outer_seed + fold_id * 101 + target_index * 13,
        )
        base_predictions[column] = {"oof": oof, "test": test}

    direct_oof = base_predictions[target]["oof"]
    direct_test = base_predictions[target]["test"]
    aux_oof = {column: base_predictions[column]["oof"] for column in aux_columns}
    aux_test = {column: base_predictions[column]["test"] for column in aux_columns}

    # --- Physics-only branch: PREDICTED auxiliaries only, never true test auxiliaries ---
    phys_oof = estm.compute_estm_physics_prediction(
        target,
        aux_oof,
        t_train,
        direct_target_prediction=direct_oof,
    )
    phys_test = estm.compute_estm_physics_prediction(
        target,
        aux_test,
        t_test,
        direct_target_prediction=direct_test,
    )
    phys_oof = estm.stabilize_inverse_physics_branch(target, phys_oof, y_train)
    phys_test = estm.stabilize_inverse_physics_branch(target, phys_test, y_train)

    # --- Ares_old = Aphys + h(...) residual-correction branch ---
    residual_features_train = np.column_stack(
        [X_train, direct_oof, phys_oof, *[aux_oof[column] for column in aux_columns], t_train]
    )
    residual_features_test = np.column_stack(
        [X_test, direct_test, phys_test, *[aux_test[column] for column in aux_columns], t_test]
    )
    residual_estimator, _residual_model, _residual_params = build_base_tree_estimator(
        random_state=outer_seed + fold_id * 101 + 71,
        n_jobs=args.n_jobs,
        base_model=args.base_model,
    )
    residual_delta_oof, residual_delta_test = oof_and_test_predictions(
        residual_estimator,
        residual_features_train,
        y_train - phys_oof,
        residual_features_test,
        n_splits=args.inner_cv,
        random_state=outer_seed + fold_id * 101 + 71,
    )
    residual_oof = phys_oof + residual_delta_oof
    residual_test = phys_test + residual_delta_test

    train = estmgate.branch_frame(
        source=data,
        indices=train_idx,
        target=target,
        fold_id=fold_id,
        seed=fold_seed,
        y_true=y_train,
        direct=direct_oof,
        phys=phys_oof,
        residual=residual_oof,
        aux_predictions=aux_oof,
        train_or_test="train_oof",
    )
    test = estmgate.branch_frame(
        source=data,
        indices=test_idx,
        target=target,
        fold_id=fold_id,
        seed=fold_seed,
        y_true=y_test,
        direct=direct_test,
        phys=phys_test,
        residual=residual_test,
        aux_predictions=aux_test,
        train_or_test="test",
    )

    aux_train_arr = train[[f"pred_{column}" for column in aux_columns]].to_numpy(dtype=float)
    aux_test_arr = test[[f"pred_{column}" for column in aux_columns]].to_numpy(dtype=float)
    aux_true_train = np.column_stack([y_all[column][train_idx] for column in aux_columns])
    known_train = {"temperature_K": t_train, "direct_target_prediction": direct_oof}
    known_test = {"temperature_K": t_test, "direct_target_prediction": direct_test}

    aux_sigmas, sigma_rows = global_auxiliary_sigmas(
        aux_pred_train=aux_train_arr,
        aux_true_train=aux_true_train,
        aux_names=aux_columns,
    )
    phi = estmgate.phi_for_target(target, y_train)
    sigma_train, fd_train = propagated_sigma_finite_difference(
        aux_pred=aux_train_arr,
        known=known_train,
        aux_sigmas=aux_sigmas,
        phi=phi,
    )
    sigma_test, fd_test = propagated_sigma_finite_difference(
        aux_pred=aux_test_arr,
        known=known_test,
        aux_sigmas=aux_sigmas,
        phi=phi,
    )

    # R6-only reliability features (r4=None), from which the fixed R2+R5+R6 subset is taken.
    rel_train = reliability_features(
        direct=direct_oof,
        physics=phys_oof,
        residual=residual_oof,
        y_train=y_train,
        r4=None,
        sigma_phys=sigma_train,
    )
    rel_test = reliability_features(
        direct=direct_test,
        physics=phys_test,
        residual=residual_test,
        y_train=y_train,
        r4=None,
        sigma_phys=sigma_test,
    )

    groups = train["formula_id"].astype(str).to_numpy(dtype=object)
    display = TARGET_DISPLAY[target]

    def make_row(method: str, y_pred: np.ndarray, selected: dict[str, Any] | None, notes: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "dataset": "ESTM",
            "target": display,
            "target_code": target,
            "outer_seed": int(outer_seed),
            "fold_id": int(fold_id),
            "unit_id": unit_id,
            "method": method,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "selected_feature_set": "" if selected is None else selected.get("feature_set", ""),
            "selected_alpha": np.nan if selected is None else selected.get("alpha", np.nan),
            "selection_RMSE": np.nan if selected is None else selected.get("selection_RMSE", np.nan),
            "mean_test_weight": np.nan if selected is None else selected.get("mean_test_weight", np.nan),
            "uses_true_test_auxiliary_values": False,
            "uses_test_target_for_selection": False,
            "notes": notes,
        }
        row.update(metric_dict(y_test, y_pred))
        return row

    rows: list[dict[str, Any]] = [
        make_row("Direct", direct_test, None, "Independently rebuilt direct branch, fresh per seed."),
        make_row(
            "Physics-only",
            phys_test,
            None,
            "Physics branch from predicted auxiliaries only (Aphys), fresh per seed.",
        ),
        make_row(
            "Ares_old",
            residual_test,
            None,
            "Ares_old = Aphys + h(direct, phys, aux_preds, T); h selected by train OOF, fresh per seed.",
        ),
    ]

    pred_test, selected, records = tune_r2_r5_r6_gate(
        y_train=y_train,
        direct_train=direct_oof,
        direct_test=direct_test,
        residual_train=residual_oof,
        residual_test=residual_test,
        reliability_train=rel_train,
        reliability_test=rel_test,
        groups=groups,
        seed=fold_seed + 17017,
        inner_cv=args.gate_inner_cv,
        alpha_grid=alpha_grid,
    )
    for record in records:
        selection_rows.append(
            {
                "dataset": "ESTM",
                "target": display,
                "target_code": target,
                "outer_seed": int(outer_seed),
                "fold_id": int(fold_id),
                "unit_id": unit_id,
                **record,
                "selected": bool(
                    record["method"] == selected["method"]
                    and record["feature_set"] == selected["feature_set"]
                    and float(record["alpha"]) == float(selected["alpha"])
                ),
            }
        )
    rows.append(
        make_row(
            METHOD_R2_R5_R6,
            pred_test,
            selected,
            "Fixed two-branch PhysicsGate: A_final = A_direct + w*(Ares_old - A_direct); "
            "w = sigmoid(StandardScaler+Ridge(R2,R5,R6)) fit on train-side OOF, alpha selected "
            "by train-side formula-grouped CV only, feature set fixed to R2+R5+R6.",
        )
    )

    audit_rows.append(
        {
            "dataset": "ESTM",
            "target": display,
            "target_code": target,
            "outer_seed": int(outer_seed),
            "fold_id": int(fold_id),
            "unit_id": unit_id,
            "aux_sigmas_json": json.dumps(dict(zip(aux_columns, map(float, aux_sigmas)))),
            "fd_train_json": json.dumps(fd_train),
            "fd_test_json": json.dumps(fd_test),
            "sigma_rows_json": json.dumps(sigma_rows),
            "uses_true_test_auxiliary_values": False,
            "uses_test_target_for_selection": False,
            "n_train_formula_groups": int(len(np.unique(groups))),
            "gate_inner_cv_is_group_kfold_deterministic_not_reseeded": True,
        }
    )
    return rows
