"""LMB capacity-voltage route helpers.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/run_lmb_physicsgate_vc_mae.py`` in the original analysis code; only imports and paths were adapted.
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
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = WORKDIR

from physicsgate.submitted.common.gate_features import reliability_features  # noqa: E402
from physicsgate.submitted.lmb.meta import DATASET, TARGET_DISPLAY, select_prediction  # noqa: E402
from physicsgate.submitted.lmb.common import CAPACITY_TARGET, DISCHARGE_ENERGY_TARGET, ENERGY_DENSITY_TARGET, MASS_KG_COLUMN, VOLTAGE_TARGET, metrics_dict  # noqa: E402


def _as_array(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def _positive(values: Any, floor: float = 1e-12) -> NDArray[np.float64]:
    return np.maximum(_as_array(values), floor)


def vc_mass_physics_prediction(
    target: str,
    predictions: dict[str, NDArray[np.float64]],
    mass_kg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the LMB VC/mass physics branch from predicted auxiliaries."""
    mass = _positive(mass_kg)
    if target == ENERGY_DENSITY_TARGET:
        return _as_array(predictions[VOLTAGE_TARGET]) * _as_array(predictions[CAPACITY_TARGET]) / mass
    if target == DISCHARGE_ENERGY_TARGET:
        return _as_array(predictions[VOLTAGE_TARGET]) * _as_array(predictions[CAPACITY_TARGET])
    if target == VOLTAGE_TARGET:
        return _as_array(predictions[ENERGY_DENSITY_TARGET]) * mass / _positive(predictions[CAPACITY_TARGET])
    if target == CAPACITY_TARGET:
        return _as_array(predictions[ENERGY_DENSITY_TARGET]) * mass / _positive(predictions[VOLTAGE_TARGET])
    raise ValueError(f"Unknown LMB target: {target}")


def propagated_sigma_vc_mass(
    target: str,
    predictions: dict[str, NDArray[np.float64]],
    mass_kg: NDArray[np.float64],
    sigmas: dict[str, float],
) -> NDArray[np.float64]:
    """First-order propagated uncertainty for the VC/mass LMB equations."""
    mass = _positive(mass_kg)
    if target == ENERGY_DENSITY_TARGET:
        voltage = _as_array(predictions[VOLTAGE_TARGET])
        capacity = _as_array(predictions[CAPACITY_TARGET])
        return np.sqrt(
            ((capacity / mass) * sigmas.get(VOLTAGE_TARGET, 0.0)) ** 2
            + ((voltage / mass) * sigmas.get(CAPACITY_TARGET, 0.0)) ** 2
        )
    if target == DISCHARGE_ENERGY_TARGET:
        voltage = _as_array(predictions[VOLTAGE_TARGET])
        capacity = _as_array(predictions[CAPACITY_TARGET])
        return np.sqrt(
            (capacity * sigmas.get(VOLTAGE_TARGET, 0.0)) ** 2
            + (voltage * sigmas.get(CAPACITY_TARGET, 0.0)) ** 2
        )
    if target == VOLTAGE_TARGET:
        energy_density = _as_array(predictions[ENERGY_DENSITY_TARGET])
        capacity = _positive(predictions[CAPACITY_TARGET])
        return np.sqrt(
            ((mass / capacity) * sigmas.get(ENERGY_DENSITY_TARGET, 0.0)) ** 2
            + ((energy_density * mass / (capacity**2)) * sigmas.get(CAPACITY_TARGET, 0.0)) ** 2
        )
    if target == CAPACITY_TARGET:
        energy_density = _as_array(predictions[ENERGY_DENSITY_TARGET])
        voltage = _positive(predictions[VOLTAGE_TARGET])
        return np.sqrt(
            ((mass / voltage) * sigmas.get(ENERGY_DENSITY_TARGET, 0.0)) ** 2
            + ((energy_density * mass / (voltage**2)) * sigmas.get(VOLTAGE_TARGET, 0.0)) ** 2
        )
    raise ValueError(f"Unknown LMB target: {target}")
