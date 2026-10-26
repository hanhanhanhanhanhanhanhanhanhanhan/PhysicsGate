"""LMB columns, paths and model helpers.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/lmb/common.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = WORKDIR

DATA_RAW_DIR = ROOT / "data" / "raw"

DATA_PROCESSED_DIR = ROOT / "data" / "processed"

TABLE_DIR = ROOT / "results" / "submitted"

FIGURE_DIR = ROOT / "results" / "submitted" / "figures" / "publication"

MAIN_SHEETS = ("Original data", "Firstly extended data", "Secondly extended data")

OVERALL_SHEET = "Overall data"

FEATURE_COLUMNS_RAW = (
    "Sb mole fraction",
    "Bi mole fraction",
    "Sn mole fraction",
    "Pb mole fraction",
    "Difference in Pauling electronegativities",
    " Averaged electro-negativity",
    "Average atomic radius",
    "Difference in atomic radii",
    "Covalent atomic radius",
    "Difference in covalent radii",
    "Averaged melting temperature",
    "Difference in melting temperatures",
    "Resulting lattice constant",
    "Difference in lattice constants",
    "Mixing entropy",
    "¦Ë",
    "¦Ë(COVALENT)",
    "Mixing enthalpy",
    "VEC",
    "Averaged ionization energy",
    "Averaged density",
    "Averaged density(liquid)",
    "Averaged thermal conductivity",
    "Averaged electrical conductivity",
    "Averaged resistivity",
    "Cathode melting point(¡æ)",
    "LiCl mole fraction",
    "LiF mole fraction",
    "LiBr mole fraction",
    "LiI mole fraction",
    "KI mole fraction",
    "KBr mole fraction",
    "Salt melting point(¡æ)",
    "Cathode mass(g)",
    "Anode mass(g)",
    "Opretaing temperature(¡æ)",
    "Area (cm^2)",
    "Current densities(mA/cm^2)",
    "Cut-off voltage(V)",
    "Theoretical capacity(Ah)",
    "Cycles",
)

TARGET_COLUMNS_RAW = (
    "Energy Density(Wh/kg)",
    "Normal Discharge Voltage(V)",
    "Discharge capacity(Ah)",
    "Discharge Energy(Wh)",
)

NAME_COLUMN = "name"

SOURCE_SHEET_COLUMN = "source_sheet"

ENERGY_DENSITY_TARGET = "energy_density_wh_kg"

VOLTAGE_TARGET = "normal_discharge_voltage_v"

CAPACITY_TARGET = "discharge_capacity_ah"

DISCHARGE_ENERGY_TARGET = "discharge_energy_wh"

TARGET_COLUMNS = (
    ENERGY_DENSITY_TARGET,
    VOLTAGE_TARGET,
    CAPACITY_TARGET,
    DISCHARGE_ENERGY_TARGET,
)

CATHODE_MASS_COLUMN = "cathode_mass_g"

ANODE_MASS_COLUMN = "anode_mass_g"

MASS_G_COLUMN = "mass_g"

MASS_KG_COLUMN = "mass_kg"

FEATURE_COLUMNS = tuple()


def safe_name(name: object) -> str:
    """Convert workbook labels to stable snake_case names."""
    text = str(name).strip()
    replacements = {
        "µ": "u",
        "μ": "u",
        "σ": "sigma",
        "Σ": "sigma",
        "λ": "lambda",
        "¦Ë": "lambda",
        "¡æ": "c",
        "℃": "c",
        "¦¸": "ohm",
        "Ω": "ohm",
        "%": "percent",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


FEATURE_COLUMNS = tuple(safe_name(col) for col in FEATURE_COLUMNS_RAW)


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with safe snake_case columns."""
    renamed = {column: safe_name(column) for column in frame.columns}
    return frame.rename(columns=renamed).copy()


def validate_feature_columns(frame: pd.DataFrame) -> None:
    """Ensure the expected 41 paper features are available."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing LMB feature columns: {missing}")
    if len(FEATURE_COLUMNS) != 41:
        raise ValueError(f"Expected 41 feature columns, found {len(FEATURE_COLUMNS)}")


def coerce_numeric_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values."""
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def compute_mass_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add total cell mass in grams and kilograms."""
    if CATHODE_MASS_COLUMN not in frame.columns or ANODE_MASS_COLUMN not in frame.columns:
        raise ValueError("Cathode and anode mass columns are required")
    result = frame.copy()
    result[MASS_G_COLUMN] = result[CATHODE_MASS_COLUMN] + result[ANODE_MASS_COLUMN]
    result[MASS_KG_COLUMN] = result[MASS_G_COLUMN] / 1000.0
    return result


def energy_density_from_energy(
    discharge_energy_wh: np.ndarray | pd.Series,
    mass_kg: np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute Wh/kg from discharge energy and cell mass."""
    mass = np.asarray(mass_kg, dtype=float)
    return np.asarray(discharge_energy_wh, dtype=float) / np.maximum(mass, 1e-12)


def energy_density_from_voltage_capacity(
    voltage_v: np.ndarray | pd.Series,
    capacity_ah: np.ndarray | pd.Series,
    mass_kg: np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute Wh/kg from voltage, capacity, and cell mass."""
    mass = np.asarray(mass_kg, dtype=float)
    return (
        np.asarray(voltage_v, dtype=float)
        * np.asarray(capacity_ah, dtype=float)
        / np.maximum(mass, 1e-12)
    )


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Return mean absolute percentage error with zero-safe denominator."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(true), 1e-12)
    return float(np.mean(np.abs((true - pred) / denominator)) * 100.0)


def nrmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Repository-standard normalized RMSE: RMSE divided by target std."""
    true = np.asarray(y_true, dtype=float)
    spread = float(np.std(true, ddof=0))
    if spread <= 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(true, y_pred)) / spread)


def metrics_dict(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Return the LMB benchmark metric bundle."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(true, pred))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(true, pred)),
        "MAPE": mape(true, pred),
        "nRMSE": nrmse(true, pred),
        "R2": float(r2_score(true, pred)),
    }


def build_rf_regressor(*, random_state: int, n_jobs: int = 1):
    """Build the paper-style random forest regressor used for auxiliaries."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=n_jobs,
            min_samples_leaf=1,
        ),
    )


def build_energy_density_regressor(*, random_state: int, n_jobs: int = 1):
    """Build the main direct energy-density model.

    XGBoost is used when available, matching the paper-facing comparison. The
    fallback keeps the repository runnable in minimal environments.
    """
    try:
        from xgboost import XGBRegressor  # type: ignore

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        )
    except Exception:
        return make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(random_state=random_state),
        )


def selected_model_name(model: object) -> str:
    """Return a compact estimator name for output tables."""
    if hasattr(model, "steps"):
        final = model.steps[-1][1]  # type: ignore[attr-defined]
        return final.__class__.__name__
    return model.__class__.__name__


def ensure_output_dirs() -> None:
    """Create standard output directories."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
