"""PV physics routes.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/core_v2/run_pv_all_targets_composition_wide_meta_a2.py`` in the original analysis code; only imports and paths were adapted.
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

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = WORKDIR

TARGET_DISPLAY = {
    "PCE_percent": "PCE_percent",
    "Voc_V": "VOC (V)",
    "Jsc_mA_cm2": "JSC (mA/cm2)",
    "FF_fraction": "FF",
}


def _as_array(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def _positive(values: Any, floor: float = 1e-12) -> NDArray[np.float64]:
    return np.maximum(_as_array(values), floor)


def compute_pv_physics_prediction(
    target: str,
    predictions: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Compute target-specific PV physics prediction from predicted branches."""
    voc = predictions.get("Voc_V")
    jsc = predictions.get("Jsc_mA_cm2")
    ff = predictions.get("FF_fraction")
    pce = predictions.get("PCE_percent")
    if target == "PCE_percent":
        if voc is None or jsc is None or ff is None:
            raise ValueError("PCE physics requires predicted Voc, Jsc, and FF")
        return _as_array(voc) * _as_array(jsc) * _as_array(ff)
    if target == "Voc_V":
        if pce is None or jsc is None or ff is None:
            raise ValueError("Voc physics requires predicted PCE, Jsc, and FF")
        return _positive(pce) / (_positive(jsc) * _positive(ff))
    if target == "Jsc_mA_cm2":
        if pce is None or voc is None or ff is None:
            raise ValueError("Jsc physics requires predicted PCE, Voc, and FF")
        return _positive(pce) / (_positive(voc) * _positive(ff))
    if target == "FF_fraction":
        if pce is None or voc is None or jsc is None:
            raise ValueError("FF physics requires predicted PCE, Voc, and Jsc")
        return _positive(pce) / (_positive(voc) * _positive(jsc))
    raise ValueError(f"Unknown PV target: {target}")
