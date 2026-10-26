"""SSE pipeline (entry point).

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/tmp_sse_mnr_data_old_split_multiseed_stability.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

from physicsgate.submitted.sse.target_adaptive import TARGET_COLUMNS, TARGET_ORDER, _fit_physres_prediction, _fit_residual_prediction, _split_group_indices, physics_prediction_for_target, residual_feature_names_for_target  # noqa: E402
from physicsgate.submitted.common.propagated_error import global_auxiliary_sigmas, propagated_sigma_finite_difference  # noqa: E402
from physicsgate.submitted.common.gate_fit import metric_dict, reliability_features, safe_logit, safe_sigmoid, series_arrays, split_positions  # noqa: E402
from physicsgate.submitted.sse.fixed_gate import DEFAULT_GATE_ALPHAS, GATE_FEATURES, GATE_METHOD, summarize, tune_fixed_gate_alpha  # noqa: E402
from physicsgate.submitted.sse.data import AUX_CODES, AUX_NAMES, DEFAULT_SCORING, DEFAULT_SEEDS, FEATURE_SELECTION, INNER_CV, MODEL_STRATEGY, N_ITER, N_JOBS, PHI, PRIMARY_METRIC_LEVEL, TARGET_MODE, fit_auxiliary_direct_branch, fit_primary_direct_branch, load_and_prepare_data  # noqa: E402

TABLE_DIR = REPO_ROOT / "results" / "submitted"

TEST_SIZE = 0.25  # Isolation cell: grouped split, matching the ORIGINAL pipeline's split settings.

SPLIT_TYPE = "group_aware_split_by_materials_GroupShuffleSplit"

R3_GATE_FEATURES = [
    "R2_direct_residual_disagreement",
    "R3_residual_physics_correction_size",
    "R5_physics_out_of_range_amount",
    "R6_global_propagated_uncertainty_proxy",
]


def _fit_uniform_gate_model(
    *,
    y_train: np.ndarray,
    direct_train: np.ndarray,
    direct_test: np.ndarray,
    residual_train: np.ndarray,
    residual_test: np.ndarray,
    reliability_train: pd.DataFrame,
    reliability_test: pd.DataFrame,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the same oracle-logit Ridge gate without delta-squared weighting."""
    truth = np.asarray(y_train, dtype=float).reshape(-1)
    direct = np.asarray(direct_train, dtype=float).reshape(-1)
    residual = np.asarray(residual_train, dtype=float).reshape(-1)
    delta = residual - direct
    usable = np.abs(delta) > np.finfo(float).eps
    w_star = np.zeros_like(truth)
    w_star[usable] = np.clip((truth[usable] - direct[usable]) / delta[usable], 0.0, 1.0)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(reliability_train[GATE_FEATURES].to_numpy(dtype=float), safe_logit(w_star))
    train_weight = safe_sigmoid(
        model.predict(reliability_train[GATE_FEATURES].to_numpy(dtype=float))
    )
    test_weight = safe_sigmoid(
        model.predict(reliability_test[GATE_FEATURES].to_numpy(dtype=float))
    )
    train_prediction = direct + train_weight * delta
    test_direct = np.asarray(direct_test, dtype=float).reshape(-1)
    test_residual = np.asarray(residual_test, dtype=float).reshape(-1)
    test_prediction = test_direct + test_weight * (test_residual - test_direct)
    return train_prediction, test_prediction, test_weight


def tune_uniform_gate_alpha(
    *,
    y_train: np.ndarray,
    direct_train: np.ndarray,
    direct_test: np.ndarray,
    residual_train: np.ndarray,
    residual_test: np.ndarray,
    reliability_train: pd.DataFrame,
    reliability_test: pd.DataFrame,
    groups: np.ndarray,
    seed: int,
    alpha_grid: list[float],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Select Ridge alpha by the same train-side grouped CV as the original gate."""
    truth = np.asarray(y_train, dtype=float).reshape(-1)
    direct = np.asarray(direct_train, dtype=float).reshape(-1)
    residual = np.asarray(residual_train, dtype=float).reshape(-1)
    folds = split_positions(groups, INNER_CV, seed)
    best_alpha = float(alpha_grid[0])
    best_score = np.inf
    for alpha in alpha_grid:
        validation_prediction = np.full_like(truth, np.nan, dtype=float)
        for fit_idx, valid_idx in folds:
            _, pred_valid, _ = _fit_uniform_gate_model(
                y_train=truth[fit_idx],
                direct_train=direct[fit_idx],
                direct_test=direct[valid_idx],
                residual_train=residual[fit_idx],
                residual_test=residual[valid_idx],
                reliability_train=reliability_train.iloc[fit_idx].reset_index(drop=True),
                reliability_test=reliability_train.iloc[valid_idx].reset_index(drop=True),
                alpha=float(alpha),
            )
            validation_prediction[valid_idx] = pred_valid
        score = metric_dict(truth, validation_prediction)["RMSE"]
        if score < best_score:
            best_score = float(score)
            best_alpha = float(alpha)
    _, test_prediction, test_weight = _fit_uniform_gate_model(
        y_train=truth,
        direct_train=direct,
        direct_test=direct_test,
        residual_train=residual,
        residual_test=residual_test,
        reliability_train=reliability_train,
        reliability_test=reliability_test,
        alpha=best_alpha,
    )
    return test_prediction, test_weight, best_alpha, best_score


def _fit_r3_gate_model(
    *,
    y_train: np.ndarray,
    direct_train: np.ndarray,
    direct_test: np.ndarray,
    residual_train: np.ndarray,
    residual_test: np.ndarray,
    reliability_train: pd.DataFrame,
    reliability_test: pd.DataFrame,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the original delta-squared weighted gate with R3 added."""
    truth = np.asarray(y_train, dtype=float).reshape(-1)
    direct = np.asarray(direct_train, dtype=float).reshape(-1)
    residual = np.asarray(residual_train, dtype=float).reshape(-1)
    delta = residual - direct
    usable = np.abs(delta) > np.finfo(float).eps
    w_star = np.zeros_like(truth)
    w_star[usable] = np.clip((truth[usable] - direct[usable]) / delta[usable], 0.0, 1.0)
    target = safe_logit(w_star)
    sample_weight = delta**2
    if float(np.sum(sample_weight)) <= np.finfo(float).eps:
        train_weight = np.zeros_like(truth)
        test_weight = np.zeros_like(np.asarray(direct_test, dtype=float).reshape(-1))
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(
            reliability_train[R3_GATE_FEATURES].to_numpy(dtype=float),
            target,
            ridge__sample_weight=sample_weight,
        )
        train_weight = safe_sigmoid(
            model.predict(reliability_train[R3_GATE_FEATURES].to_numpy(dtype=float))
        )
        test_weight = safe_sigmoid(
            model.predict(reliability_test[R3_GATE_FEATURES].to_numpy(dtype=float))
        )
    train_prediction = direct + train_weight * delta
    test_direct = np.asarray(direct_test, dtype=float).reshape(-1)
    test_residual = np.asarray(residual_test, dtype=float).reshape(-1)
    test_prediction = test_direct + test_weight * (test_residual - test_direct)
    return train_prediction, test_prediction, test_weight


def tune_r3_gate_alpha(
    *,
    y_train: np.ndarray,
    direct_train: np.ndarray,
    direct_test: np.ndarray,
    residual_train: np.ndarray,
    residual_test: np.ndarray,
    reliability_train: pd.DataFrame,
    reliability_test: pd.DataFrame,
    groups: np.ndarray,
    seed: int,
    alpha_grid: list[float],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Select the R2+R3+R5+R6 Ridge alpha by train-side grouped CV."""
    truth = np.asarray(y_train, dtype=float).reshape(-1)
    direct = np.asarray(direct_train, dtype=float).reshape(-1)
    residual = np.asarray(residual_train, dtype=float).reshape(-1)
    folds = split_positions(groups, INNER_CV, seed)
    best_alpha = float(alpha_grid[0])
    best_score = np.inf
    for alpha in alpha_grid:
        validation_prediction = np.full_like(truth, np.nan, dtype=float)
        for fit_idx, valid_idx in folds:
            _, pred_valid, _ = _fit_r3_gate_model(
                y_train=truth[fit_idx],
                direct_train=direct[fit_idx],
                direct_test=direct[valid_idx],
                residual_train=residual[fit_idx],
                residual_test=residual[valid_idx],
                reliability_train=reliability_train.iloc[fit_idx].reset_index(drop=True),
                reliability_test=reliability_train.iloc[valid_idx].reset_index(drop=True),
                alpha=float(alpha),
            )
            validation_prediction[valid_idx] = pred_valid
        score = metric_dict(truth, validation_prediction)["RMSE"]
        if score < best_score:
            best_score = float(score)
            best_alpha = float(alpha)
    _, test_prediction, test_weight = _fit_r3_gate_model(
        y_train=truth,
        direct_train=direct,
        direct_test=direct_test,
        residual_train=residual,
        residual_test=residual_test,
        reliability_train=reliability_train,
        reliability_test=reliability_test,
        alpha=best_alpha,
    )
    return test_prediction, test_weight, best_alpha, best_score


def run_seed_target(
    *,
    seed: int,
    target: str,
    data: pd.DataFrame,
    feature_columns: list[str],
    alpha_grid: list[float],
    test_size: float,
    gate_weighting: str,
    include_r3: bool,
    include_physresstack: bool,
) -> list[dict[str, Any]]:
    """Same as tmp_sse_improved_mnr_fixed_gate_multiseed_stability.run_seed_target,
    except the outer split is GroupShuffleSplit by Materials at test_size=0.25."""
    target_col = TARGET_COLUMNS[target]
    frame_valid = data.loc[data[target_col].notna()].reset_index(drop=True)

    groups_valid = frame_valid["Materials"].to_numpy()
    train_idx, test_idx = _split_group_indices(groups_valid, test_size=test_size, random_state=seed)
    train = frame_valid.iloc[train_idx].reset_index(drop=True)
    test = frame_valid.iloc[test_idx].reset_index(drop=True)

    y_train = train[target_col].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)
    T_train = train["T_K"].to_numpy(dtype=float)
    T_test = test["T_K"].to_numpy(dtype=float)

    direct_oof, direct_test, direct_model = fit_primary_direct_branch(
        target=target, train=train, test=test, feature_columns=feature_columns,
        seed=seed + TARGET_ORDER.index(target) * 31,
    )

    aux_codes = AUX_CODES[target]
    current_oof: dict[str, np.ndarray] = {target: direct_oof}
    current_test: dict[str, np.ndarray] = {target: direct_test}
    aux_models: dict[str, str] = {}
    aux_fillin_counts: dict[str, int] = {}
    for aux_code in aux_codes:
        aux_oof, aux_test, aux_model, n_fillin = fit_auxiliary_direct_branch(
            aux_code=aux_code, anchor_target=target, train=train, test=test,
            feature_columns=feature_columns,
            seed=seed + 500 + TARGET_ORDER.index(target) * 31 + TARGET_ORDER.index(aux_code) * 7,
        )
        current_oof[aux_code] = aux_oof
        current_test[aux_code] = aux_test
        aux_models[aux_code] = aux_model
        aux_fillin_counts[aux_code] = n_fillin

    physics_oof = physics_prediction_for_target(target, current_oof, T_train)
    physics_test = physics_prediction_for_target(target, current_test, T_test)

    residual_oof, residual_test, residual_model, _residual_params, _residual_feats = (
        _fit_residual_prediction(
            target=target,
            X_train_base=train[feature_columns].to_numpy(dtype=float),
            T_train=T_train,
            y_train=y_train,
            groups_train=train["Materials"].to_numpy(),
            physics_oof=physics_oof,
            current_oof=current_oof,
            X_test_base=test[feature_columns].to_numpy(dtype=float),
            T_test=T_test,
            physics_test=physics_test,
            current_test=current_test,
            model_strategy=MODEL_STRATEGY,
            inner_cv=INNER_CV,
            n_iter=N_ITER,
            scoring=DEFAULT_SCORING,
            random_state=seed + 101 + TARGET_ORDER.index(target) * 3,
            n_jobs=N_JOBS,
            feature_selection=FEATURE_SELECTION,
            feature_names=residual_feature_names_for_target(target, feature_columns),
        )
    )

    aux_pred_train = np.column_stack([current_oof[code] for code in aux_codes])
    aux_true_train = np.column_stack([train[TARGET_COLUMNS[code]].to_numpy(dtype=float) for code in aux_codes])
    aux_pred_test = np.column_stack([current_test[code] for code in aux_codes])
    aux_sigmas, _sigma_rows = global_auxiliary_sigmas(
        aux_pred_train=aux_pred_train, aux_true_train=aux_true_train, aux_names=AUX_NAMES[target],
    )
    known_train = {"T_K": T_train}
    known_test = {"T_K": T_test}
    sigma_train, _fd_train_audit = propagated_sigma_finite_difference(
        aux_pred=aux_pred_train, known=known_train, aux_sigmas=aux_sigmas, phi=PHI[target],
    )
    sigma_test, _fd_test_audit = propagated_sigma_finite_difference(
        aux_pred=aux_pred_test, known=known_test, aux_sigmas=aux_sigmas, phi=PHI[target],
    )

    rel_train_full = reliability_features(
        direct=direct_oof, physics=physics_oof, residual=residual_oof,
        y_train=y_train, r4=None, sigma_phys=sigma_train,
    )
    rel_test_full = reliability_features(
        direct=direct_test, physics=physics_test, residual=residual_test,
        y_train=y_train, r4=None, sigma_phys=sigma_test,
    )
    selected_gate_features = R3_GATE_FEATURES if include_r3 else GATE_FEATURES
    rel_train = rel_train_full[selected_gate_features].reset_index(drop=True)
    rel_test = rel_test_full[selected_gate_features].reset_index(drop=True)

    if include_r3:
        gate_tuner = tune_r3_gate_alpha
        gate_method = "FixedGate_R2_R3_R5_R6"
    elif gate_weighting == "uniform":
        gate_tuner = tune_uniform_gate_alpha
        gate_method = "UniformGate_R2_R5_R6"
    else:
        gate_tuner = tune_fixed_gate_alpha
        gate_method = GATE_METHOD
    gate_prediction, gate_weight, selected_alpha, selection_rmse = gate_tuner(
        y_train=y_train,
        direct_train=direct_oof,
        direct_test=direct_test,
        residual_train=residual_oof,
        residual_test=residual_test,
        reliability_train=rel_train,
        reliability_test=rel_test,
        groups=train["Materials"].astype(str).to_numpy(),
        seed=seed + 9001 + TARGET_ORDER.index(target) * 3,
        alpha_grid=alpha_grid,
    )

    physres_prediction = None
    physres_model = ""
    if include_physresstack:
        physres_prediction, physres_model, _physres_params, _physres_features = _fit_physres_prediction(
            y_train=y_train,
            groups_train=train["Materials"].astype(str).to_numpy(),
            direct_oof=direct_oof,
            physics_oof=physics_oof,
            residual_oof=residual_oof,
            direct_test=direct_test,
            physics_test=physics_test,
            residual_test=residual_test,
            model_strategy=MODEL_STRATEGY,
            inner_cv=INNER_CV,
            n_iter=N_ITER,
            scoring=DEFAULT_SCORING,
            random_state=seed + 202,
            n_jobs=N_JOBS,
            feature_selection=FEATURE_SELECTION,
            feature_names=["direct_prediction", "physics_prediction", "residual_prediction"],
        )

    target_mode = TARGET_MODE[target]
    primary_level = PRIMARY_METRIC_LEVEL[target_mode]
    rows: list[dict[str, Any]] = []
    method_predictions = [
        ("Direct", direct_test),
        ("Physics-only", physics_test),
        ("Ares_old_residual", residual_test),
        (gate_method, gate_prediction),
    ]
    if physres_prediction is not None:
        method_predictions.append(("PhysResStack", physres_prediction))
    for method, pred in method_predictions:
        for metric_level in ("row", "series"):
            if metric_level == "series":
                truth, predv = series_arrays(test, y_test, np.asarray(pred, dtype=float))
            else:
                truth, predv = y_test, np.asarray(pred, dtype=float)
            metrics = metric_dict(truth, predv)
            rows.append(
                {
                    "dataset": "SSE_MNR_data_old_split",
                    "target": target_mode,
                    "target_code": target,
                    "seed": int(seed),
                    "metric_level": metric_level,
                    "primary_metric_level": primary_level,
                    "method": method,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_train_materials": int(train["Materials"].nunique()),
                    "n_test_materials": int(test["Materials"].nunique()),
                    "test_size": test_size,
                    "split_type": SPLIT_TYPE,
                    "direct_model": direct_model,
                    "aux_model_1": f"{aux_codes[0]}:{aux_models[aux_codes[0]]}",
                    "aux_model_2": f"{aux_codes[1]}:{aux_models[aux_codes[1]]}",
                    "aux_1_fillin_train_rows": aux_fillin_counts[aux_codes[0]],
                    "aux_2_fillin_train_rows": aux_fillin_counts[aux_codes[1]],
                    "residual_model": residual_model,
                    "physres_meta_model": physres_model,
                    "gate_model_description": (
                        f"StandardScaler+Ridge(alpha={selected_alpha};weighting={gate_weighting};"
                        f"features={'+'.join(selected_gate_features)})"
                    ),
                    "selected_gate_alpha": selected_alpha,
                    "gate_selection_RMSE": selection_rmse,
                    "mean_gate_test_weight": float(np.mean(gate_weight)),
                    "uses_true_test_auxiliary_values": False,
                    "uses_test_target_for_gate_selection": False,
                    "reruns_base_models_independently_per_seed": True,
                    "reruns_aux_models_independently_per_target": True,
                    **metrics,
                }
            )
    return rows


def summarize_with_wincount(
    detail: pd.DataFrame, *, gate_method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The shared summarizer predates configurable gate names and expects the
    # original method label. Translate only for aggregation, then restore the
    # actual experimental label in the returned table.
    detail_for_summary = detail.copy()
    if gate_method != GATE_METHOD:
        detail_for_summary["method"] = detail_for_summary["method"].replace(
            {gate_method: GATE_METHOD}
        )
    grouped = summarize(detail_for_summary)
    if gate_method != GATE_METHOD:
        grouped["method"] = grouped["method"].replace({GATE_METHOD: gate_method})
    pivot = detail.pivot_table(
        index=["target", "target_code", "metric_level", "primary_metric_level", "seed"],
        columns="method",
        values="RMSE",
    ).reset_index()
    pivot["pct_change_gate_vs_direct"] = (
        (pivot[gate_method] - pivot["Direct"]) / pivot["Direct"] * 100.0
    )
    pivot["gate_beats_direct"] = pivot[gate_method] < pivot["Direct"]
    win_summary = pivot.groupby(
        ["target", "target_code", "metric_level", "primary_metric_level"], as_index=False
    ).agg(
        gate_win_count_of_n=("gate_beats_direct", "sum"),
        n_seeds_compared=("gate_beats_direct", "size"),
    )
    grouped = grouped.merge(
        win_summary, on=["target", "target_code", "metric_level", "primary_metric_level"], how="left"
    )
    return grouped, pivot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--alphas", default=DEFAULT_GATE_ALPHAS)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--gate-weighting", choices=("delta2", "uniform"), default="delta2")
    parser.add_argument("--include-r3", action="store_true")
    parser.add_argument("--include-physresstack", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    alpha_grid = [float(value) for value in args.alphas.split(",") if value.strip()]
    if args.include_r3 and args.gate_weighting != "delta2":
        raise ValueError("--include-r3 currently requires --gate-weighting delta2")
    if args.include_r3:
        gate_method = "FixedGate_R2_R3_R5_R6"
    elif args.gate_weighting == "uniform":
        gate_method = "UniformGate_R2_R5_R6"
    else:
        gate_method = GATE_METHOD
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1")
    if args.output_tag.strip():
        output_stem = f"tmp_sse_mnr_data_{args.output_tag.strip()}_multiseed_stability"
    elif np.isclose(args.test_size, TEST_SIZE):
        output_stem = "tmp_sse_mnr_data_old_split_multiseed_stability"
    else:
        train_pct = int(round((1.0 - args.test_size) * 100))
        test_pct = int(round(args.test_size * 100))
        output_stem = f"tmp_sse_mnr_data_group_{train_pct}_{test_pct}_multiseed_stability"
    detailed_output = TABLE_DIR / f"{output_stem}_all_seeds_detailed.csv"
    summary_output = TABLE_DIR / f"{output_stem}_stability_summary.csv"
    pivot_output = TABLE_DIR / f"{output_stem}_per_seed_pivot.csv"
    start = time.time()

    data, feature_columns = load_and_prepare_data()
    print(f"TEST_SIZE={args.test_size}, split={SPLIT_TYPE}")

    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for target in TARGET_ORDER:
            t0 = time.time()
            rows = run_seed_target(
                seed=seed, target=target, data=data, feature_columns=feature_columns,
                alpha_grid=alpha_grid, test_size=args.test_size,
                gate_weighting=args.gate_weighting,
                include_r3=args.include_r3,
                include_physresstack=args.include_physresstack,
            )
            all_rows.extend(rows)
            print(
                f"Completed seed={seed} target={TARGET_MODE[target]} "
                f"in {time.time() - t0:.1f}s ({len(rows)} rows)"
            )

    detail = pd.DataFrame(all_rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detailed_output, index=False)
    print(f"Wrote {detailed_output}")

    summary, pivot = summarize_with_wincount(detail, gate_method=gate_method)
    summary.to_csv(summary_output, index=False)
    print(f"Wrote {summary_output}")
    pivot.to_csv(pivot_output, index=False)
    print(f"Wrote {pivot_output}")
    print(f"Total runtime: {time.time() - start:.1f}s")

    primary = summary[summary["metric_level"].eq(summary["primary_metric_level"])].copy()
    display_cols = [
        "target", "metric_level", "method", "RMSE_mean", "RMSE_std", "R2_mean",
        "percent_RMSE_change_vs_direct_of_means",
        "gate_pct_change_vs_direct_mean_per_seed", "gate_pct_change_vs_direct_std_per_seed",
        "gate_win_count_of_n", "n_seeds_compared",
    ]
    print(primary[[c for c in display_cols if c in primary.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
