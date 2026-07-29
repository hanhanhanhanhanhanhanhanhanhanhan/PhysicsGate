"""Dataset adapters for the four retained PhysicsGate benchmarks.

The adapters standardize columns and write a schema sidecar. They do not fit
predictive models or use held-out labels. Raw datasets are intentionally not
distributed with this repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from .equations import ARRHENIUS_KB_EV_PER_K
from .formula import composition_to_fraction_vector, element_feature_names, parse_formula


PV_FEATURES_RAW = (
    "Cell_architecture",
    "Cell_flexible",
    "Cell_semitransparent",
    "Substrate_stack_sequence",
    "ETL_stack_sequence",
    "ETL_thickness",
    "ETL_deposition_procedure",
    "Perovskite_dimension_2D",
    "Perovskite_dimension_2D3D_mixture",
    "Perovskite_dimension_3D",
    "Perovskite_dimension_3D_with_2D_capping_layer",
    "Perovskite_composition_perovskite_ABC3_structure",
    "Perovskite_composition_long_form",
    "Perovskite_thickness",
    "Perovskite_composition_inorganic",
    "Perovskite_band_gap",
    "Perovskite_band_gap_graded",
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_solvents",
    "Perovskite_deposition_quenching_induced_crystallisation",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Perovskite_deposition_solvent_annealing",
    "HTL_stack_sequence",
    "HTL_thickness_list",
    "HTL_deposition_procedure",
    "Backcontact_stack_sequence",
    "Backcontact_thickness_list",
    "Backcontact_deposition_procedure",
)
PV_NUMERIC_RAW = (
    "ETL_thickness",
    "Perovskite_thickness",
    "Perovskite_band_gap",
    "HTL_thickness_list",
    "Backcontact_thickness_list",
)
PV_TARGETS_RAW = {
    "Voc_V": "JV_default_Voc",
    "Jsc_mA_cm2": "JV_default_Jsc",
    "FF_fraction": "JV_default_FF",
    "PCE_percent": "JV_default_PCE",
}

LMB_FEATURES_RAW = (
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
    "娄脣",
    "娄脣(COVALENT)",
    "Mixing enthalpy",
    "VEC",
    "Averaged ionization energy",
    "Averaged density",
    "Averaged density(liquid)",
    "Averaged thermal conductivity",
    "Averaged electrical conductivity",
    "Averaged resistivity",
    "Cathode melting point(隆忙)",
    "LiCl mole fraction",
    "LiF mole fraction",
    "LiBr mole fraction",
    "LiI mole fraction",
    "KI mole fraction",
    "KBr mole fraction",
    "Salt melting point(隆忙)",
    "Cathode mass(g)",
    "Anode mass(g)",
    "Opretaing temperature(隆忙)",
    "Area (cm^2)",
    "Current densities(mA/cm^2)",
    "Cut-off voltage(V)",
    "Theoretical capacity(Ah)",
    "Cycles",
)
LMB_TARGETS_RAW = {
    "Energy Density(Wh/kg)": "energy_density_wh_kg",
    "Normal Discharge Voltage(V)": "normal_discharge_voltage_v",
    "Discharge capacity(Ah)": "discharge_capacity_ah",
    "Discharge Energy(Wh)": "discharge_energy_wh",
}
LMB_SHEETS = ("Original data", "Firstly extended data", "Secondly extended data")
LMB_REPLACEMENT_CHARACTER_ALIASES = {
    "\ufffd": "lambda",
    "\ufffd(COVALENT)": "lambda(COVALENT)",
    "Cathode melting point(\ufffd)": "Cathode melting point(c)",
    "Salt melting point(\ufffd)": "Salt melting point(c)",
    "Opretaing temperature(\ufffd)": "Opretaing temperature(c)",
}


def _safe_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    replacements = {
        "σ": "sigma",
        "娄脣": "lambda",
        "隆忙": "c",
        "λ": "lambda",
        "℃": "c",
        "%": "percent",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_").lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(
    frame: pd.DataFrame,
    *,
    input_path: Path,
    output_path: Path,
    dataset: str,
    feature_columns: list[str],
    target_columns: list[str],
    group_column: str | None,
    context_columns: list[str],
    notes: list[str],
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    schema_path = output_path.with_suffix(".schema.json")
    payload = {
        "dataset": dataset,
        "raw_input_name": input_path.name,
        "raw_input_sha256": _sha256(input_path),
        "n_rows": int(len(frame)),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "group_column": group_column,
        "context_columns": context_columns,
        "notes": notes,
    }
    schema_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path, schema_path


def preprocess_sse(input_path: str | Path, output_path: str | Path) -> tuple[Path, Path]:
    """Prepare the revised row-level SSE workbook.

    The first column is the material/group identifier, the final three columns
    are the targets, and the intervening columns are descriptors.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    raw = pd.read_excel(input_path)
    if raw.shape[1] < 5:
        raise ValueError("SSE workbook must contain an id, descriptors and three targets")
    id_column = raw.columns[0]
    feature_raw = list(raw.columns[1:-3])
    target_raw = list(raw.columns[-3:])
    if len(feature_raw) != 26:
        raise ValueError(f"Expected 26 SSE descriptors, found {len(feature_raw)}")

    frame = pd.DataFrame({"group_id": raw[id_column].astype(str).str.strip()})
    features: list[str] = []
    for index, column in enumerate(feature_raw, start=1):
        name = _safe_name(column) or f"feature_{index:02d}"
        if name in frame:
            name = f"{name}_{index:02d}"
        frame[name] = pd.to_numeric(raw[column], errors="coerce")
        features.append(name)

    target_aliases = {
        "lnsigmat": "ln_sigma_T",
        "mnrea": "Ea",
        "mnr_ea": "Ea",
        "ea": "Ea",
        "mnrlnsigma0": "ln_sigma0",
        "mnr_lnsigma0": "ln_sigma0",
        "lnsigma0": "ln_sigma0",
    }
    targets: list[str] = []
    for column in target_raw:
        compact = _safe_name(column).replace("_", "")
        name = target_aliases.get(compact, target_aliases.get(_safe_name(column)))
        if name is None:
            raise ValueError(f"Unrecognized SSE target column: {column!r}")
        frame[name] = pd.to_numeric(raw[column], errors="coerce")
        targets.append(name)

    inverse_candidates = [
        column
        for column in features
        if ("1000" in column and "t" in column) or column in {"inv_t", "inv_t_1000"}
    ]
    context: list[str] = []
    if inverse_candidates:
        inv = frame[inverse_candidates[0]].to_numpy(dtype=float)
        frame["T_K"] = np.where(inv > 0, 1000.0 / inv, np.nan)
        context.append("T_K")
    elif "temperature_k" in frame:
        frame["T_K"] = frame["temperature_k"]
        context.append("T_K")
    else:
        raise ValueError("SSE descriptors must contain temperature or 1000/T")

    complete = frame.dropna(subset=["ln_sigma_T", "Ea", "ln_sigma0", "T_K"])
    if not complete.empty:
        frame["arrhenius_residual_true"] = np.nan
        frame.loc[complete.index, "arrhenius_residual_true"] = (
            complete["ln_sigma_T"]
            - complete["ln_sigma0"]
            + complete["Ea"] / (ARRHENIUS_KB_EV_PER_K * complete["T_K"])
        )
    return _write(
        frame,
        input_path=input_path,
        output_path=output_path,
        dataset="SSE",
        feature_columns=features,
        target_columns=["ln_sigma_T", "Ea", "ln_sigma0"],
        group_column="group_id",
        context_columns=context,
        notes=["Target-wise missing rows are retained and removed inside each route."],
    )


def preprocess_estm(input_path: str | Path, output_path: str | Path) -> tuple[Path, Path]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    raw = pd.read_csv(input_path, low_memory=False)
    columns = {_safe_name(column): column for column in raw.columns}

    def take(*aliases: str) -> pd.Series:
        for alias in aliases:
            if alias in columns:
                return raw[columns[alias]]
        raise ValueError(f"Missing ESTM column; expected one of {aliases}")

    formula = take("formula").astype(str).str.strip()
    temperature = pd.to_numeric(take("temperature_k", "temperature"), errors="coerce")
    seebeck = pd.to_numeric(
        take(
            "seebeck_coefficient_v_k",
            "seebeck_coefficient_uv_k",
            "seebeck_coefficient_v_per_k",
            "seebeck_coefficient_uv_per_k",
        ),
        errors="coerce",
    )
    sigma = pd.to_numeric(
        take("electrical_conductivity_s_m", "electrical_conductivity_s_per_m"),
        errors="coerce",
    )
    kappa = pd.to_numeric(
        take("thermal_conductivity_w_mk", "thermal_conductivity_w_per_mk"),
        errors="coerce",
    )

    rows: list[int] = []
    vectors: list[np.ndarray] = []
    for index, value in formula.items():
        try:
            parse_formula(value)
            vectors.append(composition_to_fraction_vector(value, max_atomic_number=100))
            rows.append(index)
        except Exception:
            continue
    frame = pd.DataFrame(
        np.vstack(vectors),
        columns=element_feature_names(max_atomic_number=100),
        index=rows,
    )
    frame.insert(0, "formula_id", formula.loc[rows].to_numpy())
    frame["temperature_K"] = temperature.loc[rows].to_numpy()
    frame["seebeck_uV_per_K"] = seebeck.loc[rows].to_numpy()
    frame["log10_electrical_conductivity"] = np.log10(sigma.loc[rows].where(sigma.loc[rows] > 0))
    frame["log10_thermal_conductivity"] = np.log10(kappa.loc[rows].where(kappa.loc[rows] > 0))
    s_v = frame["seebeck_uV_per_K"] * 1e-6
    frame["ZT"] = (
        s_v.pow(2)
        * np.power(10.0, frame["log10_electrical_conductivity"])
        * frame["temperature_K"]
        / np.power(10.0, frame["log10_thermal_conductivity"])
    )
    targets = [
        "seebeck_uV_per_K",
        "ZT",
        "log10_electrical_conductivity",
        "log10_thermal_conductivity",
    ]
    features = [*element_feature_names(max_atomic_number=100), "temperature_K"]
    frame = frame.dropna(subset=[*targets, "temperature_K"]).reset_index(drop=True)
    return _write(
        frame,
        input_path=input_path,
        output_path=output_path,
        dataset="ESTM",
        feature_columns=features,
        target_columns=targets,
        group_column="formula_id",
        context_columns=["temperature_K"],
        notes=["Formula parse failures are excluded.", "Conductivities are modelled in log10 space."],
    )


def _parse_numeric_list(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    parts = str(value).replace(";", "|").split("|")
    try:
        values = [float(part.strip()) for part in parts if part.strip()]
    except ValueError:
        return np.nan
    return float(sum(values)) if values else np.nan


def preprocess_pv(input_path: str | Path, output_path: str | Path) -> tuple[Path, Path]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    raw = pd.read_csv(input_path, low_memory=False)
    required = ["Ref_DOI_number", *PV_FEATURES_RAW, *PV_TARGETS_RAW.values()]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"PV dataset is missing columns: {missing}")
    selected = raw.loc[:, required].copy()
    selected.insert(0, "raw_row_index", raw.index.to_numpy())
    selected = selected.dropna(subset=list(PV_TARGETS_RAW.values()))
    unknown = selected.astype(str).apply(
        lambda column: column.str.contains(
            r"Unknown|unkown|unknown|None|none", regex=True, na=False
        )
    )
    selected = selected.loc[~unknown.any(axis=1)].copy()
    pce = pd.to_numeric(selected["JV_default_PCE"], errors="coerce")
    selected = selected.loc[pce.between(0, 25)].copy()

    frame = pd.DataFrame(
        {
            "raw_row_index": selected["raw_row_index"].astype(int),
            "ref_doi_number": selected["Ref_DOI_number"].astype(str),
        },
        index=selected.index,
    )
    features: list[str] = []
    for column in PV_FEATURES_RAW:
        name = _safe_name(column)
        if column in PV_NUMERIC_RAW:
            frame[name] = selected[column].map(_parse_numeric_list)
        else:
            frame[name] = selected[column].astype(str)
        features.append(name)
    frame["Voc_V"] = pd.to_numeric(selected["JV_default_Voc"], errors="coerce")
    frame["Jsc_mA_cm2"] = pd.to_numeric(selected["JV_default_Jsc"], errors="coerce")
    ff = pd.to_numeric(selected["JV_default_FF"], errors="coerce")
    frame["FF_fraction"] = ff / 100.0 if float(ff.median()) > 1.2 else ff
    frame["PCE_percent"] = pd.to_numeric(selected["JV_default_PCE"], errors="coerce")
    targets = ["Voc_V", "Jsc_mA_cm2", "FF_fraction", "PCE_percent"]
    frame = frame.dropna(subset=targets).reset_index(drop=True)
    return _write(
        frame,
        input_path=input_path,
        output_path=output_path,
        dataset="PV",
        feature_columns=features,
        target_columns=targets,
        group_column=None,
        context_columns=[],
        notes=[
            "Feature imputation/encoding is fitted inside each outer-training split.",
            "The paper-matched outer protocol is a row holdout.",
        ],
    )


def preprocess_lmb(input_path: str | Path, output_path: str | Path) -> tuple[Path, Path]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    workbook = pd.ExcelFile(input_path)
    missing_sheets = [sheet for sheet in LMB_SHEETS if sheet not in workbook.sheet_names]
    if missing_sheets:
        raise ValueError(f"LMB workbook is missing sheets: {missing_sheets}")
    raw = pd.concat(
        [pd.read_excel(workbook, sheet_name=sheet) for sheet in LMB_SHEETS],
        ignore_index=True,
    )
    raw = raw.rename(
        columns={
            column: _safe_name(
                LMB_REPLACEMENT_CHARACTER_ALIASES.get(str(column), column)
            )
            for column in raw.columns
        }
    )
    features = [_safe_name(column) for column in LMB_FEATURES_RAW]
    targets = list(LMB_TARGETS_RAW.values())
    target_map = {_safe_name(raw_name): safe for raw_name, safe in LMB_TARGETS_RAW.items()}
    raw = raw.rename(columns=target_map)
    if "name" not in raw:
        raise ValueError("LMB workbook must contain a Name column")
    missing = sorted(set(features + targets) - set(raw.columns))
    if missing:
        raise ValueError(f"LMB workbook is missing columns: {missing}")
    frame = pd.DataFrame({"group_id": raw["name"].astype(str).str.strip()})
    for column in features + targets:
        frame[column] = pd.to_numeric(raw[column], errors="coerce")
    cathode = _safe_name("Cathode mass(g)")
    anode = _safe_name("Anode mass(g)")
    frame["mass_kg"] = (frame[cathode] + frame[anode]) / 1000.0
    frame = frame.dropna(subset=[*targets, "mass_kg"])
    frame = frame.loc[frame["mass_kg"] > 0].reset_index(drop=True)
    return _write(
        frame,
        input_path=input_path,
        output_path=output_path,
        dataset="LMB",
        feature_columns=features,
        target_columns=targets,
        group_column="group_id",
        context_columns=["mass_kg"],
        notes=["All 41 original descriptors are retained; no RFE is applied."],
    )


__all__ = [
    "preprocess_estm",
    "preprocess_lmb",
    "preprocess_pv",
    "preprocess_sse",
]
