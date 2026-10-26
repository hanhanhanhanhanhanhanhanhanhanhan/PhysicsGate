"""LMB raw-data loader.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/lmb/load_lmb_dataset.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = WORKDIR

from physicsgate.submitted.lmb.common import DATA_PROCESSED_DIR, DATA_RAW_DIR, DISCHARGE_ENERGY_TARGET, ENERGY_DENSITY_TARGET, FEATURE_COLUMNS, MAIN_SHEETS, MASS_KG_COLUMN, NAME_COLUMN, OVERALL_SHEET, SOURCE_SHEET_COLUMN, TABLE_DIR, TARGET_COLUMNS, TARGET_COLUMNS_RAW, coerce_numeric_columns, compute_mass_columns, energy_density_from_energy, energy_density_from_voltage_capacity, ensure_output_dirs, safe_name, standardize_columns, validate_feature_columns, VOLTAGE_TARGET, CAPACITY_TARGET  # noqa: E402

DEFAULT_INPUT = DATA_RAW_DIR / "Energy density.xlsx"

DEFAULT_OUTPUT = DATA_PROCESSED_DIR / "lmb_clean.csv"

DEFAULT_PROFILE = TABLE_DIR / "lmb_dataset_profile.csv"

DEFAULT_MAPPING = TABLE_DIR / "lmb_column_mapping.csv"


def _read_main_sheets(input_path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    workbook = pd.ExcelFile(input_path)
    missing = [sheet for sheet in MAIN_SHEETS if sheet not in workbook.sheet_names]
    if missing:
        raise ValueError(f"Missing expected LMB sheets: {missing}")
    frames: list[pd.DataFrame] = []
    sheet_rows: dict[str, int] = {}
    for sheet in MAIN_SHEETS:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        sheet_rows[sheet] = len(frame)
        frame[SOURCE_SHEET_COLUMN] = sheet
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), sheet_rows


def build_column_mapping(raw_columns: list[str]) -> pd.DataFrame:
    """Create raw-to-standardized column mapping for auditability."""
    target_raw_safe = {safe_name(column) for column in TARGET_COLUMNS_RAW}
    feature_safe = set(FEATURE_COLUMNS)
    rows = []
    for column in raw_columns:
        safe = safe_name(column)
        if safe == "name":
            role = "identifier"
        elif safe in feature_safe:
            role = "feature"
        elif safe in target_raw_safe:
            role = "target"
        elif safe == SOURCE_SHEET_COLUMN:
            role = "source_sheet"
        else:
            role = "other"
        rows.append(
            {
                "raw_column": column,
                "safe_column": safe,
                "role": role,
                "included_main_feature": bool(safe in feature_safe),
            }
        )
    return pd.DataFrame(rows)


def load_lmb_dataset(input_path: str | Path = DEFAULT_INPUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the main 2158-row LMB electrochemical performance dataset."""
    raw_main, _ = _read_main_sheets(input_path)
    mapping = build_column_mapping(list(raw_main.columns))
    data = standardize_columns(raw_main)
    if "name" in data.columns:
        data = data.rename(columns={"name": NAME_COLUMN})
    numeric_columns = [*FEATURE_COLUMNS, *TARGET_COLUMNS]
    data = coerce_numeric_columns(data, numeric_columns)
    data = compute_mass_columns(data)
    validate_feature_columns(data)
    required = [*TARGET_COLUMNS, MASS_KG_COLUMN]
    clean = data.dropna(subset=required).copy()
    clean = clean.loc[np.isfinite(clean[MASS_KG_COLUMN]) & (clean[MASS_KG_COLUMN] > 0)].copy()
    clean["ed_from_true_discharge_energy"] = energy_density_from_energy(
        clean[DISCHARGE_ENERGY_TARGET], clean[MASS_KG_COLUMN]
    )
    clean["ed_from_true_voltage_capacity"] = energy_density_from_voltage_capacity(
        clean[VOLTAGE_TARGET], clean[CAPACITY_TARGET], clean[MASS_KG_COLUMN]
    )
    clean["ed_residual_true_energy_branch"] = (
        clean[ENERGY_DENSITY_TARGET] - clean["ed_from_true_discharge_energy"]
    )
    clean["ed_residual_true_vc_branch"] = (
        clean[ENERGY_DENSITY_TARGET] - clean["ed_from_true_voltage_capacity"]
    )
    return clean.reset_index(drop=True), mapping


def profile_lmb_dataset(
    input_path: str | Path,
    clean: pd.DataFrame,
) -> pd.DataFrame:
    """Build a one-row dataset profile table."""
    workbook = pd.ExcelFile(input_path)
    _, sheet_rows = _read_main_sheets(input_path)
    overall_rows = (
        len(pd.read_excel(workbook, sheet_name=OVERALL_SHEET))
        if OVERALL_SHEET in workbook.sheet_names
        else np.nan
    )
    profile = {
        "n_rows_original": sheet_rows.get("Original data", 0),
        "n_rows_firstly_extended": sheet_rows.get("Firstly extended data", 0),
        "n_rows_secondly_extended": sheet_rows.get("Secondly extended data", 0),
        "n_rows_main_raw": int(sum(sheet_rows.values())),
        "n_rows_clean": int(len(clean)),
        "n_rows_overall_sheet": overall_rows,
        "n_features": len(FEATURE_COLUMNS),
        "n_unique_names": int(clean[NAME_COLUMN].nunique()) if NAME_COLUMN in clean else np.nan,
        "min_mass_kg": float(clean[MASS_KG_COLUMN].min()),
        "max_mass_kg": float(clean[MASS_KG_COLUMN].max()),
        "min_energy_density_wh_kg": float(clean[ENERGY_DENSITY_TARGET].min()),
        "max_energy_density_wh_kg": float(clean[ENERGY_DENSITY_TARGET].max()),
        "min_discharge_energy_wh": float(clean[DISCHARGE_ENERGY_TARGET].min()),
        "max_discharge_energy_wh": float(clean[DISCHARGE_ENERGY_TARGET].max()),
        "mean_abs_ed_residual_true_energy_branch": float(
            clean["ed_residual_true_energy_branch"].abs().mean()
        ),
        "mean_abs_ed_residual_true_vc_branch": float(
            clean["ed_residual_true_vc_branch"].abs().mean()
        ),
    }
    return pd.DataFrame([profile])


def run_loader(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    profile_path: str | Path = DEFAULT_PROFILE,
    mapping_path: str | Path = DEFAULT_MAPPING,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, save, and profile the LMB dataset."""
    ensure_output_dirs()
    clean, mapping = load_lmb_dataset(input_path)
    profile = profile_lmb_dataset(input_path, clean)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mapping_path).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    profile.to_csv(profile_path, index=False)
    mapping.to_csv(mapping_path, index=False)
    return clean, profile, mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile-output", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean, profile, _ = run_loader(
        input_path=args.input,
        output_path=args.output,
        profile_path=args.profile_output,
        mapping_path=args.mapping_output,
    )
    print(f"Loaded LMB dataset: {len(clean)} retained rows")
    print(profile.to_string(index=False))


if __name__ == "__main__":
    main()
