"""ESTM raw-data loader.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/thermoelectric/load_estm_dataset.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
from pathlib import Path
import re
import sys
import unicodedata
import numpy as np
import pandas as pd

ROOT = WORKDIR

from physicsgate.submitted.common.formula import composition_to_fraction_vector, element_feature_names, parse_formula  # noqa: E402

DEFAULT_INPUT_PATH = ROOT / "data" / "raw" / "estm.csv"

DEFAULT_PROCESSED_PATH = ROOT / "data" / "processed" / "estm_clean.csv"

DEFAULT_PROFILE_PATH = ROOT / "results" / "submitted" / "estm_dataset_profile.csv"

DEFAULT_PARSE_FAILURE_PATH = (
    ROOT / "results" / "submitted" / "estm_formula_parse_failures.csv"
)

REQUIRED_OUTPUT_COLUMNS = (
    "formula",
    "temperature_K",
    "seebeck_uV_per_K",
    "seebeck_V_per_K",
    "electrical_conductivity_S_per_m",
    "log10_electrical_conductivity",
    "thermal_conductivity_W_per_mK",
    "log10_thermal_conductivity",
    "power_factor_W_per_mK2",
    "ZT",
    "reference",
    "formula_id",
)


def _safe_column_name(column: object) -> str:
    text = unicodedata.normalize("NFKC", str(column)).strip()
    text = text.replace("μ", "u").replace("µ", "u")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def drop_empty_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop fully empty spreadsheet artifact columns named Unnamed:*."""
    keep_columns = []
    for column in frame.columns:
        if str(column).startswith("Unnamed") and frame[column].isna().all():
            continue
        keep_columns.append(column)
    return frame.loc[:, keep_columns].copy()


def standardize_estm_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize raw ESTM CSV column names to safe snake_case names."""
    aliases = {
        "formula": "formula",
        "temperature_k": "temperature_K",
        "temperature": "temperature_K",
        "seebeck_coefficient_v_k": "seebeck_uV_per_K",
        "seebeck_coefficient_uv_k": "seebeck_uV_per_K",
        "seebeck_coefficient_v_per_k": "seebeck_uV_per_K",
        "seebeck_coefficient_uv_per_k": "seebeck_uV_per_K",
        "electrical_conductivity_s_m": "electrical_conductivity_S_per_m",
        "electrical_conductivity_s_per_m": "electrical_conductivity_S_per_m",
        "thermal_conductivity_w_mk": "thermal_conductivity_W_per_mK",
        "thermal_conductivity_w_per_mk": "thermal_conductivity_W_per_mK",
        "power_factor_w_mk2": "power_factor_input_W_per_mK2",
        "power_factor_w_per_mk2": "power_factor_input_W_per_mK2",
        "zt": "ZT_input",
        "reference": "reference",
    }
    renamed: dict[object, str] = {}
    for column in frame.columns:
        safe = _safe_column_name(column)
        renamed[column] = aliases.get(safe, safe)
    standardized = frame.rename(columns=renamed)
    duplicates = standardized.columns[standardized.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate standardized ESTM columns: {duplicates}")
    return standardized


def _validate_required_columns(frame: pd.DataFrame) -> None:
    required = (
        "formula",
        "temperature_K",
        "seebeck_uV_per_K",
        "electrical_conductivity_S_per_m",
        "thermal_conductivity_W_per_mK",
        "reference",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"ESTM CSV is missing required columns: {missing}")


def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column not in {"formula", "reference", "formula_id"}:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _positive_log10(values: pd.Series) -> pd.Series:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    logged = np.full(array.shape, np.nan, dtype=float)
    mask = np.isfinite(array) & (array > 0)
    logged[mask] = np.log10(array[mask])
    return pd.Series(logged, index=values.index)


def clean_estm_dataframe(
    raw_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw ESTM data and return cleaned rows plus parse failures."""
    raw = drop_empty_unnamed_columns(raw_frame)
    frame = standardize_estm_columns(raw)
    _validate_required_columns(frame)
    frame = _coerce_numeric(frame)
    frame = frame.dropna(subset=["formula", "temperature_K"]).copy()
    frame["formula"] = frame["formula"].astype(str).str.strip()
    if "reference" not in frame:
        frame["reference"] = ""
    frame["reference"] = frame["reference"].astype(str).str.strip()

    frame = frame[frame["formula"].str.len() > 0].copy()

    vectors: list[np.ndarray] = []
    parsed_rows: list[int] = []
    failures: list[dict[str, object]] = []
    for row_index, formula in frame["formula"].items():
        try:
            parse_formula(formula)
            vectors.append(composition_to_fraction_vector(formula, max_atomic_number=100))
            parsed_rows.append(row_index)
        except Exception as exc:  # pragma: no cover - exact parser message tested indirectly
            failures.append(
                {
                    "row_index": int(row_index),
                    "formula": formula,
                    "error": str(exc),
                }
            )

    parse_failures = pd.DataFrame(failures, columns=["row_index", "formula", "error"])
    cleaned = frame.loc[parsed_rows].copy()
    if cleaned.empty:
        elem_frame = pd.DataFrame(columns=element_feature_names(max_atomic_number=100))
    else:
        elem_frame = pd.DataFrame(
            np.vstack(vectors),
            columns=element_feature_names(max_atomic_number=100),
            index=cleaned.index,
        )

    cleaned["seebeck_V_per_K"] = cleaned["seebeck_uV_per_K"] * 1e-6
    cleaned["log10_electrical_conductivity"] = _positive_log10(
        cleaned["electrical_conductivity_S_per_m"]
    )
    cleaned["log10_thermal_conductivity"] = _positive_log10(
        cleaned["thermal_conductivity_W_per_mK"]
    )
    cleaned["power_factor_W_per_mK2"] = (
        np.square(cleaned["seebeck_V_per_K"])
        * cleaned["electrical_conductivity_S_per_m"]
    )
    cleaned["ZT"] = (
        cleaned["power_factor_W_per_mK2"]
        * cleaned["temperature_K"]
        / cleaned["thermal_conductivity_W_per_mK"]
    )
    cleaned.loc[~np.isfinite(cleaned["ZT"]), "ZT"] = np.nan
    cleaned["formula_id"] = cleaned["formula"]

    for column in REQUIRED_OUTPUT_COLUMNS:
        if column not in cleaned:
            cleaned[column] = np.nan
    cleaned = pd.concat(
        [cleaned.loc[:, list(REQUIRED_OUTPUT_COLUMNS)], elem_frame],
        axis=1,
    ).reset_index(drop=True)
    return cleaned, parse_failures


def build_estm_dataset_profile(
    raw_frame: pd.DataFrame,
    cleaned_frame: pd.DataFrame,
    parse_failures: pd.DataFrame,
) -> pd.DataFrame:
    """Build a one-row ESTM profile for reporting."""
    profile = {
        "n_rows_raw": int(len(raw_frame)),
        "n_rows_clean": int(len(cleaned_frame)),
        "n_rows_formula_parse_failures": int(len(parse_failures)),
        "n_unique_formula": int(cleaned_frame["formula_id"].nunique(dropna=True)),
        "min_temperature_K": float(cleaned_frame["temperature_K"].min()),
        "max_temperature_K": float(cleaned_frame["temperature_K"].max()),
        "min_seebeck_uV_per_K": float(cleaned_frame["seebeck_uV_per_K"].min()),
        "max_seebeck_uV_per_K": float(cleaned_frame["seebeck_uV_per_K"].max()),
        "min_log10_electrical_conductivity": float(
            cleaned_frame["log10_electrical_conductivity"].min()
        ),
        "max_log10_electrical_conductivity": float(
            cleaned_frame["log10_electrical_conductivity"].max()
        ),
        "min_log10_thermal_conductivity": float(
            cleaned_frame["log10_thermal_conductivity"].min()
        ),
        "max_log10_thermal_conductivity": float(
            cleaned_frame["log10_thermal_conductivity"].max()
        ),
        "min_ZT": float(cleaned_frame["ZT"].min()),
        "max_ZT": float(cleaned_frame["ZT"].max()),
        "mean_power_factor_W_per_mK2": float(
            cleaned_frame["power_factor_W_per_mK2"].mean()
        ),
    }
    return pd.DataFrame([profile])


def load_estm_dataset(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    processed_output_path: str | Path = DEFAULT_PROCESSED_PATH,
    profile_output_path: str | Path = DEFAULT_PROFILE_PATH,
    parse_failure_output_path: str | Path = DEFAULT_PARSE_FAILURE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read, clean, profile, and save the local ESTM CSV."""
    input_path = Path(input_path)
    raw_frame = pd.read_csv(input_path)
    cleaned, parse_failures = clean_estm_dataframe(raw_frame)
    profile = build_estm_dataset_profile(raw_frame, cleaned, parse_failures)

    processed_output_path = Path(processed_output_path)
    profile_output_path = Path(profile_output_path)
    parse_failure_output_path = Path(parse_failure_output_path)
    processed_output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_output_path.parent.mkdir(parents=True, exist_ok=True)
    parse_failure_output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed_output_path, index=False)
    profile.to_csv(profile_output_path, index=False)
    parse_failures.to_csv(parse_failure_output_path, index=False)
    return cleaned, profile, parse_failures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean and profile the local ESTM thermoelectric CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
    )
    parser.add_argument(
        "--parse-failure-output",
        type=Path,
        default=DEFAULT_PARSE_FAILURE_PATH,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cleaned, profile, parse_failures = load_estm_dataset(
        args.input,
        processed_output_path=args.processed_output,
        profile_output_path=args.profile_output,
        parse_failure_output_path=args.parse_failure_output,
    )
    profile_row = profile.iloc[0]
    print(f"Saved cleaned ESTM data to {args.processed_output}")
    print(f"Saved ESTM profile to {args.profile_output}")
    print(f"Saved formula parse failures to {args.parse_failure_output}")
    print(f"Rows retained: {int(profile_row['n_rows_clean'])}")
    print(f"Formula parse failures: {len(parse_failures)}")
    print(f"Unique formulas: {int(profile_row['n_unique_formula'])}")
    print(
        "Temperature range K: "
        f"{profile_row['min_temperature_K']:.3f} to "
        f"{profile_row['max_temperature_K']:.3f}"
    )
    print(f"ZT range: {profile_row['min_ZT']:.6g} to {profile_row['max_ZT']:.6g}")


if __name__ == "__main__":
    main()
