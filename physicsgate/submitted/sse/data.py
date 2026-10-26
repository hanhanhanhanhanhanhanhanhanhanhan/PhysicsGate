"""SSE data loading and base-model search.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/tmp_sse_improved_mnr_fixed_gate_multiseed_stability.py`` in the original analysis code; only imports and paths were adapted.
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
from sklearn.base import clone
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

from physicsgate.submitted.sse.target_adaptive import DEFAULT_SCORING, TARGET_COLUMNS, _fit_and_predict_test, _group_oof_predictions, _model_metadata, direct_features_for_target  # noqa: E402
from physicsgate.submitted.common.gate_fit import phi_e, phi_l, phi_p  # noqa: E402

RAW_INPUT_PATH = REPO_ROOT / "data" / "raw" / "SSE.xlsx"

DEFAULT_SEEDS = "4,42,99,178,2024"

MODEL_STRATEGY = "tuned_sklearn"

N_ITER = 20

INNER_CV = 3

N_JOBS = -1

FEATURE_SELECTION = "none"

TARGET_MODE = {"L": "ln_sigma_T", "P": "ln_sigma0", "E": "Ea"}

PRIMARY_METRIC_LEVEL = {"ln_sigma_T": "row", "ln_sigma0": "row", "Ea": "row"}

PHI = {"L": phi_l, "P": phi_p, "E": phi_e}

AUX_CODES = {"L": ["P", "E"], "P": ["L", "E"], "E": ["L", "P"]}

AUX_NAMES = {
    "L": ["ln_sigma0", "Ea"],
    "P": ["ln_sigma_T", "Ea"],
    "E": ["ln_sigma_T", "ln_sigma0"],
}


def load_and_prepare_data() -> tuple[pd.DataFrame, list[str]]:
    """Load SSE.xlsx, derive T_K, and rename the mangled target columns.

    Feature columns (columns 1-26, 26 total) are captured by POSITION before
    any renaming and are never referenced by their literal (possibly
    mangled) string content elsewhere in this script.
    """
    raw = pd.read_excel(RAW_INPUT_PATH)
    if raw.shape[1] != 30:
        raise ValueError(f"Expected 30 columns in SSE.xlsx, found {raw.shape[1]}")
    feature_columns = list(raw.columns[1:27])
    if len(feature_columns) != 26:
        raise ValueError(f"Expected 26 feature columns, found {len(feature_columns)}")

    data = raw.rename(
        columns={
            raw.columns[27]: "ln_sigma_T",
            raw.columns[28]: "Ea",
            raw.columns[29]: "ln_sigma0",
        }
    ).copy()
    data["Materials"] = data["Materials"].astype(str)
    # Alias for `Materials` under the name the reused SSE helpers/series_arrays
    # expect, so those functions can be imported and reused unmodified.
    data["arrhenius_series_id"] = data["Materials"]

    data["T_K"] = 1000.0 / data[feature_columns[-1]].astype(float)

    sanity_row0 = float(data["T_K"].iloc[0])
    print(f"Sanity check: row 0 T_K = {sanity_row0:.3f} K (expected ~383 K)")
    if not (300.0 <= sanity_row0 <= 470.0):
        raise ValueError(
            f"T_K derivation looks wrong: row0 T_K={sanity_row0}, expected ~383 K"
        )

    nan_ea = set(np.flatnonzero(data["Ea"].isna().to_numpy()))
    nan_p = set(np.flatnonzero(data["ln_sigma0"].isna().to_numpy()))
    print(
        f"NaN rows: Ea={len(nan_ea)}, ln_sigma0={len(nan_p)}, "
        f"identical row sets={nan_ea == nan_p}"
    )
    print(f"Loaded {len(data)} rows, {data['Materials'].nunique()} materials, "
          f"{len(feature_columns)} feature columns.")
    return data, feature_columns


def fit_primary_direct_branch(
    *,
    target: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Fit the anchor target's own Direct model on its own NaN-free train/test frames."""
    X_train_base = train[feature_columns].to_numpy(dtype=float)
    X_test_base = test[feature_columns].to_numpy(dtype=float)
    T_train = train["T_K"].to_numpy(dtype=float)
    T_test = test["T_K"].to_numpy(dtype=float)
    y_train = train[TARGET_COLUMNS[target]].to_numpy(dtype=float)
    groups_train = train["Materials"].to_numpy()

    X_train = direct_features_for_target(target, X_train_base, T_train)
    X_test = direct_features_for_target(target, X_test_base, T_test)

    estimator, selected_model, _params_json = _model_metadata(
        X=X_train,
        y=y_train,
        groups=groups_train,
        model_strategy=MODEL_STRATEGY,
        role=f"direct_{target}",
        n_iter=N_ITER,
        inner_cv=INNER_CV,
        scoring=DEFAULT_SCORING,
        random_state=seed,
        n_jobs=N_JOBS,
        feature_selection=FEATURE_SELECTION,
    )
    oof = _group_oof_predictions(estimator, X_train, y_train, groups_train, n_splits=INNER_CV)
    test_pred = _fit_and_predict_test(estimator, X_train, y_train, X_test)
    return oof, test_pred, selected_model


def fit_auxiliary_direct_branch(
    *,
    aux_code: str,
    anchor_target: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Fit an auxiliary Direct model for `aux_code`, used inside `anchor_target`'s pipeline.

    `train`/`test` are the ANCHOR target's own (already NaN-filtered-for-the-
    anchor) frames, which may still contain NaN values for `aux_code` (this
    happens for anchor_target="L", which has 0 NaN itself but whose train/test
    rows span the full dataset, including the ~21 rows where Ea/ln_sigma0 are
    NaN). Rows valid for `aux_code` get genuine group-CV OOF predictions;
    any remaining (aux-NaN) anchor-train rows get a refit-predict fill-in
    (no leakage: there is no true aux label at those rows to leak). All
    anchor-test rows get a plain refit-predict regardless of aux validity
    there (predicting never requires a true aux label).
    """
    aux_col = TARGET_COLUMNS[aux_code]
    valid_mask = train[aux_col].notna().to_numpy()
    n_fillin = int((~valid_mask).sum())

    train_valid = train.loc[valid_mask].reset_index(drop=True)
    X_train_valid_base = train_valid[feature_columns].to_numpy(dtype=float)
    T_train_valid = train_valid["T_K"].to_numpy(dtype=float)
    X_train_valid = direct_features_for_target(aux_code, X_train_valid_base, T_train_valid)
    y_train_valid = train_valid[aux_col].to_numpy(dtype=float)
    groups_train_valid = train_valid["Materials"].to_numpy()

    estimator, selected_model, _params_json = _model_metadata(
        X=X_train_valid,
        y=y_train_valid,
        groups=groups_train_valid,
        model_strategy=MODEL_STRATEGY,
        role=f"aux_{aux_code}_for_{anchor_target}",
        n_iter=N_ITER,
        inner_cv=INNER_CV,
        scoring=DEFAULT_SCORING,
        random_state=seed,
        n_jobs=N_JOBS,
        feature_selection=FEATURE_SELECTION,
    )
    oof_valid = _group_oof_predictions(
        estimator, X_train_valid, y_train_valid, groups_train_valid, n_splits=INNER_CV
    )

    oof_full = np.full(len(train), np.nan, dtype=float)
    oof_full[np.flatnonzero(valid_mask)] = oof_valid

    fitted = clone(estimator).fit(X_train_valid, y_train_valid)

    invalid_idx = np.flatnonzero(~valid_mask)
    if invalid_idx.size:
        X_invalid_base = train.iloc[invalid_idx][feature_columns].to_numpy(dtype=float)
        T_invalid = train.iloc[invalid_idx]["T_K"].to_numpy(dtype=float)
        X_invalid = direct_features_for_target(aux_code, X_invalid_base, T_invalid)
        oof_full[invalid_idx] = fitted.predict(X_invalid)

    assert np.all(np.isfinite(oof_full)), "auxiliary OOF array must be fully populated"

    X_test_base = test[feature_columns].to_numpy(dtype=float)
    T_test = test["T_K"].to_numpy(dtype=float)
    X_test_full = direct_features_for_target(aux_code, X_test_base, T_test)
    test_pred = fitted.predict(X_test_full)

    return oof_full, test_pred, selected_model, n_fillin
