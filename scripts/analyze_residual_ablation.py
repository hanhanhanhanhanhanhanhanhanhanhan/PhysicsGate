"""Residual branch with reduced inputs against the reported residual branch.

Compares the outputs of ``physicsgate.revised.residual_ablation`` (residual
model on the Physics-only prediction and the measurement condition, or on the
reported inputs without the Direct prediction) with the revised-protocol
Data S2, endpoint by endpoint. Columns with the suffix ``_minimal`` hold the
reduced-input results. For each method the paired gain
over Direct is G = log2(RMSE_Direct / RMSE_method), averaged over the five
evaluated outer splits; the equivalent RMSE reduction is 100 (1 - 2^-G).
Direct and Physics-only do not depend on the residual inputs and are checked
for identity. LMB PhysResStack is not recomputed.

    python scripts/analyze_residual_ablation.py [--results-dir DIR] [--name NAME] [--reference CSV] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_with_data_s2 import LMB_CODE, METHOD, frame, recomputed  # noqa: E402

RESULTS = ROOT / "results" / "residual_ablation" / "workdir" / "results" / "submitted"
REFERENCE = ROOT / "source_data" / "supplementary" / "Supplementary_Data_S2_All_Evaluation_Unit_Metrics.csv"
KEYS = ["Dataset", "Target code", "Route", "Outer seed"]
METHODS = ["Physics-only", "Residual-only", "PhysResStack", "PhysicsGate"]


def ablation(results: Path) -> pd.DataFrame:
    parts = [recomputed(results)]
    path = results / "submitted_lmb_detailed.csv"
    if path.exists():  # LMB: pipeline outputs (Direct, Physics-only, Residual-only, PhysicsGate)
        d = pd.read_csv(path)
        route = np.where(d["formula_family"].astype(str).str.startswith("VC_mass_only"), "VC", "DE")
        route = pd.Series(route, index=d.index) + "-" + d["target"].map(LMB_CODE)
        lmb = frame("LMB", d["raw_target_column"], route, d["seed"], d["method"], d)
        lmb["Method"] = lmb["method"].map(METHOD)
        parts.append(lmb.dropna(subset=["Method"]))
    out = pd.concat(parts, ignore_index=True)
    out = out.rename(columns={"code": "Target code", "seed": "Outer seed"})
    return out.drop_duplicates([*KEYS, "Method"], keep="last")[[*KEYS, "Method", "RMSE"]]


def gains(table: pd.DataFrame) -> pd.DataFrame:
    wide = table.pivot_table(index=[*KEYS], columns="Method", values="RMSE")
    return pd.DataFrame({m: np.log2(wide["Direct"] / wide[m]) for m in METHODS if m in wide})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--name", default="residual_ablation", help="prefix of the output files")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revised" / "analyses")
    args = parser.parse_args()
    reference = pd.read_csv(args.reference)
    reference["Route"] = reference["Route"].fillna("")
    labels = reference.drop_duplicates(KEYS[:3]).set_index(KEYS[:3])[["Target order", "Target"]]
    minimal = ablation(args.results_dir)

    check = reference.merge(minimal, on=[*KEYS, "Method"], suffixes=("", "_minimal"))
    fixed = check[check["Method"].isin(["Direct", "Physics-only"])]
    differs = fixed[(fixed["RMSE_minimal"] / fixed["RMSE"] - 1).abs() > 1e-9]
    print(f"Direct and Physics-only: {len(fixed) - len(differs)} of {len(fixed)} units identical")

    g_ref, g_min = gains(reference), gains(minimal)
    by_split = g_ref.join(g_min, rsuffix="_minimal", how="inner").reset_index()
    by_split = by_split.join(labels, on=KEYS[:3])
    rows = []
    for (order, target, route), unit in by_split.groupby(["Target order", "Target", "Route"]):
        row = {"Target order": order, "Dataset": unit["Dataset"].iloc[0], "Target": target, "Route": route,
               "splits": len(unit)}
        for m in METHODS:
            for suffix in ("", "_minimal"):
                column = m + suffix
                if column in unit and unit[column].notna().all():
                    row[f"G {column}"] = unit[column].mean()
                    row[f"positive {column}"] = int((unit[column] > 0).sum())
        if {"G Residual-only", "G Residual-only_minimal"} <= row.keys():
            over_physics = row["G Residual-only"] - row["G Physics-only"]
            row["residual gain over Physics-only retained"] = (
                (row["G Residual-only_minimal"] - row["G Physics-only"]) / over_physics if over_physics > 0 else np.nan)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("Target order")
    macro = {"Target": "Macro mean (16 endpoints)"}
    for column in summary.columns:
        if column.startswith("G "):
            values = summary[column].dropna()
            macro[column] = values.mean() if len(values) == len(summary) else np.nan
    summary = pd.concat([summary, pd.DataFrame([macro])], ignore_index=True)
    for column in [c for c in summary.columns if c.startswith("G ")]:
        summary[column.replace("G ", "reduction % ")] = 100 * (1 - 2.0 ** -summary[column])
    args.out.mkdir(parents=True, exist_ok=True)
    by_split.to_csv(args.out / f"{args.name}_by_split.csv", index=False)
    summary.to_csv(args.out / f"{args.name}_summary.csv", index=False)
    show = ["Target", "Route", "G Physics-only", "G Residual-only", "G Residual-only_minimal",
            "residual gain over Physics-only retained", "G PhysResStack", "G PhysResStack_minimal",
            "G PhysicsGate", "G PhysicsGate_minimal", "positive PhysicsGate", "positive PhysicsGate_minimal"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary[[c for c in show if c in summary]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
