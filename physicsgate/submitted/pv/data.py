"""PV raw-data loader.

Submitted-protocol code. Definitions are copied verbatim from
``physstack/experiments/photovoltaic/load_pv_dataset.py`` in the original analysis code; only imports and paths were adapted.
"""

from __future__ import annotations

from physicsgate.submitted.paths import WORKDIR  # noqa: E402
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

ROOT = WORKDIR

from physicsgate.submitted.pv.common import BOOLEAN_FEATURES, BOOLEAN_FEATURES_RAW, FEATURE_COLUMNS, FEATURE_COLUMNS_RAW, NUMERIC_FEATURES, NUMERIC_FEATURES_RAW, TARGET_COLUMNS_RAW, metrics_dict, pce_from_terms, safe_name  # noqa: E402

UNKNOWN_PATTERN = r"Unknown|unkown|unknown|None|none"

DEFAULT_INPUT = ROOT / "data" / "raw" / "Perovskite_database_content_all_data.csv"

DEFAULT_OUTPUT = ROOT / "data" / "processed" / "pv_clean.csv"

PROFILE_OUTPUT = ROOT / "results" / "submitted" / "pv_dataset_profile.csv"

COLUMN_MAPPING_OUTPUT = ROOT / "results" / "submitted" / "pv_column_mapping.csv"

CONSISTENCY_OUTPUT = ROOT / "results" / "submitted" / "pv_pce_formula_consistency.csv"


def parse_numeric_list(value: object) -> float:
    """Convert scalar or pipe/semicolon-separated thickness values to a sum."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown", "unkown"}:
        return np.nan
    parts = [part.strip() for part in text.replace(";", "|").split("|")]
    values: list[float] = []
    for part in parts:
        if not part or part.lower() in {"nan", "none", "unknown", "unkown"}:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return np.nan
    total = float(sum(values))
    return np.nan if total == 0 else total


def _coerce_bool(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no"}:
        return 0.0
    return np.nan


def _drop_unknown_rows(frame: pd.DataFrame) -> pd.DataFrame:
    text = frame.astype(str)
    mask = text.apply(lambda column: column.str.contains(UNKNOWN_PATTERN, regex=True, na=False))
    return frame.loc[~mask.any(axis=1)].copy()


def clean_pv_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Clean raw PV data following the upstream GitHub protocol."""
    required = ["Ref_DOI_number", *FEATURE_COLUMNS_RAW, *TARGET_COLUMNS_RAW.values()]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Raw PV dataset is missing required columns: {missing}")

    selected = raw.loc[:, required].copy()
    selected.insert(0, "raw_row_index", raw.index.to_numpy())
    n_after_target_drop = int(
        selected.dropna(subset=list(TARGET_COLUMNS_RAW.values())).shape[0]
    )
    selected = selected.dropna(subset=list(TARGET_COLUMNS_RAW.values())).copy()
    selected = _drop_unknown_rows(selected)
    n_after_unknown_drop = int(selected.shape[0])

    selected = selected.loc[pd.to_numeric(selected["JV_default_PCE"], errors="coerce") <= 25]
    selected = selected.loc[pd.to_numeric(selected["JV_default_PCE"], errors="coerce") >= 0]
    selected = selected.loc[selected["HTL_thickness_list"].astype(str) != "TRUE"].copy()

    for column in NUMERIC_FEATURES_RAW:
        selected[column] = selected[column].map(parse_numeric_list)

    n_duplicates = int(selected.duplicated().sum())
    selected = selected.drop_duplicates().copy()
    n_after_duplicates = int(selected.shape[0])

    imputer = KNNImputer(n_neighbors=5)
    selected[NUMERIC_FEATURES_RAW] = imputer.fit_transform(selected[NUMERIC_FEATURES_RAW])

    output = pd.DataFrame(index=selected.index)
    output["raw_row_index"] = selected["raw_row_index"].astype(int)
    output["ref_doi_number"] = selected["Ref_DOI_number"].astype(str)
    for raw_name, safe in zip(FEATURE_COLUMNS_RAW, FEATURE_COLUMNS):
        output[safe] = selected[raw_name]
    for raw_name, safe in zip(BOOLEAN_FEATURES_RAW, BOOLEAN_FEATURES):
        output[safe] = selected[raw_name].map(_coerce_bool).astype(float)
    for raw_name, safe in zip(NUMERIC_FEATURES_RAW, NUMERIC_FEATURES):
        output[safe] = pd.to_numeric(selected[raw_name], errors="coerce")

    output["Voc_V"] = pd.to_numeric(selected[TARGET_COLUMNS_RAW["Voc_V"]], errors="coerce")
    output["Jsc_mA_cm2"] = pd.to_numeric(
        selected[TARGET_COLUMNS_RAW["Jsc_mA_cm2"]], errors="coerce"
    )
    ff_raw = pd.to_numeric(selected[TARGET_COLUMNS_RAW["FF_raw"]], errors="coerce")
    inferred_ff_unit = "percent" if float(ff_raw.dropna().median()) > 1.2 else "fraction"
    output["FF_fraction"] = ff_raw / 100.0 if inferred_ff_unit == "percent" else ff_raw
    output["PCE_percent"] = pd.to_numeric(
        selected[TARGET_COLUMNS_RAW["PCE_percent"]], errors="coerce"
    )
    output["PCE_formula_percent"] = pce_from_terms(
        output["Voc_V"], output["Jsc_mA_cm2"], output["FF_fraction"]
    )

    output = output.dropna(subset=["Voc_V", "Jsc_mA_cm2", "FF_fraction", "PCE_percent"])

    audit = {
        "n_rows_raw": int(raw.shape[0]),
        "n_rows_after_drop_missing_targets": n_after_target_drop,
        "n_rows_after_drop_unknown": n_after_unknown_drop,
        "n_duplicate_rows_removed": n_duplicates,
        "n_rows_after_drop_duplicates": n_after_duplicates,
        "n_rows_cleaned": int(output.shape[0]),
        "ff_unit_inferred": inferred_ff_unit,
        "knn_imputer_n_neighbors": 5,
    }
    return output.reset_index(drop=True), audit


def build_profile(cleaned: pd.DataFrame, audit: dict[str, object]) -> pd.DataFrame:
    """Build one-row dataset profile."""
    consistency = metrics_dict(cleaned["PCE_percent"], cleaned["PCE_formula_percent"])
    row = {
        **audit,
        "n_selected_features": len(FEATURE_COLUMNS),
        "n_unique_doi": int(cleaned["ref_doi_number"].nunique()),
        "min_PCE_percent": float(cleaned["PCE_percent"].min()),
        "max_PCE_percent": float(cleaned["PCE_percent"].max()),
        "min_Voc_V": float(cleaned["Voc_V"].min()),
        "max_Voc_V": float(cleaned["Voc_V"].max()),
        "min_Jsc_mA_cm2": float(cleaned["Jsc_mA_cm2"].min()),
        "max_Jsc_mA_cm2": float(cleaned["Jsc_mA_cm2"].max()),
        "min_FF_fraction": float(cleaned["FF_fraction"].min()),
        "max_FF_fraction": float(cleaned["FF_fraction"].max()),
        "pce_formula_mae": consistency["mae"],
        "pce_formula_rmse": consistency["rmse"],
        "pce_formula_r2": consistency["r2"],
        "expected_paper_cleaned_rows": 26457,
        "within_two_percent_of_expected": bool(
            abs(int(audit["n_rows_cleaned"]) - 26457) / 26457 <= 0.02
        ),
    }
    return pd.DataFrame([row])


def build_column_mapping() -> pd.DataFrame:
    """Create raw-to-standardized column mapping."""
    rows = [
        {
            "raw_column": "Ref_DOI_number",
            "standardized_column": "ref_doi_number",
            "role": "reference",
        }
    ]
    rows.extend(
        {
            "raw_column": raw_name,
            "standardized_column": safe_name(raw_name),
            "role": "feature",
        }
        for raw_name in FEATURE_COLUMNS_RAW
    )
    rows.extend(
        [
            {
                "raw_column": "JV_default_Voc",
                "standardized_column": "Voc_V",
                "role": "target",
            },
            {
                "raw_column": "JV_default_Jsc",
                "standardized_column": "Jsc_mA_cm2",
                "role": "target",
            },
            {
                "raw_column": "JV_default_FF",
                "standardized_column": "FF_fraction",
                "role": "target",
            },
            {
                "raw_column": "JV_default_PCE",
                "standardized_column": "PCE_percent",
                "role": "target",
            },
        ]
    )
    return pd.DataFrame(rows)


def build_consistency(cleaned: pd.DataFrame, ff_unit: str) -> pd.DataFrame:
    """Build PCE formula consistency audit table."""
    err = cleaned["PCE_percent"] - cleaned["PCE_formula_percent"]
    metrics = metrics_dict(cleaned["PCE_percent"], cleaned["PCE_formula_percent"])
    return pd.DataFrame(
        [
            {
                "n_rows_with_all_terms": int(cleaned.shape[0]),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "median_absolute_error": float(np.median(np.abs(err))),
                "fraction_within_0_1_pce_points": float((np.abs(err) <= 0.1).mean()),
                "fraction_within_0_5_pce_points": float((np.abs(err) <= 0.5).mean()),
                "inferred_FF_unit": ff_unit,
                "notes": (
                    "PCE_percent is compared with Voc_V * Jsc_mA_cm2 * FF_fraction."
                ),
            }
        ]
    )


def run_loader(input_path: Path, output_path: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Load, clean, and save PV data and audit tables."""
    raw = pd.read_csv(input_path, low_memory=False)
    cleaned, audit = clean_pv_dataset(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    build_profile(cleaned, audit).to_csv(PROFILE_OUTPUT, index=False)
    build_column_mapping().to_csv(COLUMN_MAPPING_OUTPUT, index=False)
    build_consistency(cleaned, str(audit["ff_unit_inferred"])).to_csv(
        CONSISTENCY_OUTPUT, index=False
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    audit = run_loader(args.input, args.output)
    print(f"Wrote {args.output}")
    print(f"Rows retained: {audit['n_rows_cleaned']}")
    print(f"FF unit inferred: {audit['ff_unit_inferred']}")
    print(f"Profile: {PROFILE_OUTPUT}")
    print(f"Consistency: {CONSISTENCY_OUTPUT}")


if __name__ == "__main__":
    main()
