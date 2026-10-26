"""ESTM gate tuning.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/run_estm_all_targets_physicsgate_r4_r6_tuned.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

from physicsgate.submitted.estm import physics as estm  # noqa: E402

TARGET_COLUMNS = estm.TARGET_COLUMNS


def _as_1d(values: object) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _positive(values: object, floor: float = 1e-300) -> np.ndarray:
    return np.maximum(_as_1d(values), floor)


def phi_for_target(target: str, y_train: np.ndarray) -> Any:
    """Return a target-specific ESTM phi aligned with the mainline branch."""

    def _phi(aux: np.ndarray, known: dict[str, np.ndarray]) -> np.ndarray:
        temperature = _positive(known["temperature_K"], floor=1e-12)
        aux = np.asarray(aux, dtype=float)
        aux_map = {name: aux[:, idx] for idx, name in enumerate([c for c in TARGET_COLUMNS if c != target])}
        direct_target = known.get("direct_target_prediction")
        if direct_target is None:
            direct_target = np.ones(aux.shape[0], dtype=float)
        raw = estm.compute_estm_physics_prediction(
            target,
            aux_map,
            temperature,
            direct_target_prediction=_as_1d(direct_target),
        )
        return estm.stabilize_inverse_physics_branch(target, raw, y_train)

    return _phi


def branch_frame(
    *,
    source: pd.DataFrame,
    indices: np.ndarray,
    target: str,
    fold_id: int,
    seed: int,
    y_true: np.ndarray,
    direct: np.ndarray,
    phys: np.ndarray,
    residual: np.ndarray,
    aux_predictions: dict[str, np.ndarray],
    train_or_test: str,
) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "row_index": source.iloc[indices]["core_v2_row_index"].to_numpy(dtype=int),
            "formula_id": source.iloc[indices]["formula_id"].astype(str).to_numpy(),
            "temperature_K": source.iloc[indices]["temperature_K"].to_numpy(dtype=float),
            "target_code": target,
            "fold_id": fold_id,
            "seed": seed,
            "train_or_test": train_or_test,
            "y_true": y_true,
            "y_direct": direct,
            "y_phys": phys,
            "y_residual": residual,
        }
    )
    for column, values in aux_predictions.items():
        rows[f"pred_{column}"] = values
    return rows
