"""LMB meta-learner helpers.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/run_lmb_all_targets_90_10_wide_meta_a2.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

ROOT = WORKDIR

from physicsgate.submitted.lmb.common import CAPACITY_TARGET, DISCHARGE_ENERGY_TARGET, ENERGY_DENSITY_TARGET, VOLTAGE_TARGET  # noqa: E402
from physicsgate.submitted.lmb.clean_methods import _select_prediction_model  # noqa: E402

DATASET = "LMB"

TARGET_DISPLAY = {
    ENERGY_DENSITY_TARGET: "Energy Density(Wh/kg)",
    DISCHARGE_ENERGY_TARGET: "Discharge Energy(Wh)",
    VOLTAGE_TARGET: "Normal Discharge Voltage(V)",
    CAPACITY_TARGET: "Discharge capacity(Ah)",
}


def select_prediction(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    *,
    role: str,
    args: argparse.Namespace,
    offset: int,
    selection_rows: list[dict[str, Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], str, float]:
    start = len(selection_rows)
    oof, test, selected, score = _select_prediction_model(
        X_train,
        y_train,
        X_test,
        random_state=args.random_state + offset,
        n_jobs=args.n_jobs,
        inner_cv=args.inner_cv,
        search_level=args.search_level,
        role=role,
        selection_rows=selection_rows,
    )
    for row in selection_rows[start:]:
        row["target"] = role.split("::", 1)[0]
        row["output_tag"] = args.output_tag
    return oof, test, selected, score
