"""Compare recomputed metrics with Supplementary Data S2, unit by unit.

Reads the outputs of ``scripts/reproduce_submitted.py`` from
``results/submitted/`` and writes ``results/submitted/data_s2_comparison.csv``.
A unit is one dataset, endpoint, route, outer split and method. When every unit
has been recomputed, the recomputed metrics are also written in the layout of
Data S2, as ``data_s2_recomputed.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "submitted"
S2_PATH = ROOT / "source_data" / "supplementary" / "Supplementary_Data_S2_All_Evaluation_Unit_Metrics.csv"

METHOD = {
    "Direct": "Direct", "Physics-only": "Physics-only", "Ares_old": "Residual-only",
    "Ares_old_residual": "Residual-only", "Ordinary stacking": "Ordinary stacking",
    "PhysResStack": "PhysResStack", "FixedGate_R2_R5_R6": "PhysicsGate",
    "DirectAnchored_PhysicsGate_R2_R5_R6_tuned": "PhysicsGate",
}
LMB_CODE = {"Energy Density(Wh/kg)": "ED", "Normal Discharge Voltage(V)": "V",
            "Discharge capacity(Ah)": "C", "Discharge Energy(Wh)": "DE"}


def frame(dataset, code, route, seed, method, d):
    return pd.DataFrame({"Dataset": dataset, "code": code, "Route": route, "seed": seed,
                         "method": method, "RMSE": d["RMSE"], "MAE": d["MAE"], "nRMSE": d["nRMSE"],
                         "R2": d["R2"]})


def recomputed(OUT: Path) -> pd.DataFrame:
    parts = []
    path = OUT / "tmp_sse_mnr_data_submitted_multiseed_stability_all_seeds_detailed.csv"
    if path.exists():
        d = pd.read_csv(path)
        d = d[d["metric_level"].eq("row")]
        parts.append(frame("SSE", d["target_code"], "", d["seed"], d["method"], d))
    path = OUT / "sse_ordinary_stacking.csv"
    if path.exists():
        d = pd.read_csv(path)
        parts.append(frame("SSE", d["target_code"], "", d["seed"], d["method"], d))
    path = OUT / "tmp_estm_fixed_gate_multiseed_with_physresstack_and_ordinary_per_seed_summary.csv"
    if path.exists():
        d = pd.read_csv(path).rename(columns={"RMSE_mean": "RMSE", "MAE_mean": "MAE", "nRMSE_mean": "nRMSE",
                                              "R2_mean": "R2"})
        parts.append(frame("ESTM", d["target_code"], "", d["outer_seed"], d["method"], d))
    path = OUT / "lmb_six_methods_detailed.csv"
    if path.exists():
        d = pd.read_csv(path)
        route = np.where(d["formula_family"].astype(str).str.startswith("VC_mass_only"), "VC", "DE")
        route = pd.Series(route, index=d.index) + "-" + d["target"].map(LMB_CODE)
        parts.append(frame("LMB", d["raw_target_column"], route, d["seed"], d["method"], d))
    path = OUT / "pv" / "outputs" / "pv_unified_train_range_50seed" / "PV_Unified_Frozen5_Detailed.csv"
    if path.exists():
        d = pd.read_csv(path)
        parts.append(frame("PV", d["target_code"], "", d["seed"], d["method"], d))
    if not parts:
        raise SystemExit("No recomputed outputs found in results/submitted/")
    out = pd.concat(parts, ignore_index=True)
    out["Method"] = out["method"].map(METHOD)
    return out.dropna(subset=["Method"]).drop_duplicates(["Dataset", "code", "Route", "seed", "Method"], keep="last")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=OUT)
    out = parser.parse_args().results_dir.resolve()
    s2 = pd.read_csv(S2_PATH)
    s2["Route"] = s2["Route"].fillna("")
    m = s2.merge(recomputed(out), left_on=["Dataset", "Target code", "Route", "Outer seed", "Method"],
                 right_on=["Dataset", "code", "Route", "seed", "Method"], how="left",
                 suffixes=("_S2", "_recomputed"))
    m["relative_RMSE_difference"] = (m["RMSE_recomputed"] / m["RMSE_S2"] - 1).abs()
    m["identical"] = m["relative_RMSE_difference"] < 1e-9
    summary = m.groupby(["Dataset", "Method"]).agg(
        units=("RMSE_S2", "size"),
        recomputed=("RMSE_recomputed", lambda s: int(s.notna().sum())),
        identical=("identical", "sum"),
        max_relative_difference=("relative_RMSE_difference", "max"),
    )
    pd.set_option("display.width", 200)
    print(summary.to_string())
    n_done = int(m["RMSE_recomputed"].notna().sum())
    print(f"\n{n_done} of {len(m)} units recomputed; {int(m['identical'].sum())} identical "
          f"(relative RMSE difference < 1e-9).")
    diff = m[m["RMSE_recomputed"].notna() & ~m["identical"]]
    if len(diff):
        print("\nUnits that differ:")
        print(diff[["Dataset", "Target", "Route", "Outer seed", "Method", "RMSE_S2", "RMSE_recomputed",
                    "relative_RMSE_difference"]].sort_values("relative_RMSE_difference", ascending=False)
              .to_string(index=False))
    keep = ["Target order", "Dataset", "Target", "Route", "Outer seed", "Method", "RMSE_S2", "RMSE_recomputed",
            "MAE_S2", "MAE_recomputed", "R2_S2", "R2_recomputed", "relative_RMSE_difference", "identical"]
    out.mkdir(parents=True, exist_ok=True)
    m[keep].to_csv(out / "data_s2_comparison.csv", index=False)
    print(f"\nWrote {out / 'data_s2_comparison.csv'}")
    if n_done == len(m):
        recomputed_s2 = m[[c for c in s2.columns if c not in ("RMSE", "MAE", "nRMSE", "R2")]].copy()
        for metric in ("RMSE", "MAE", "nRMSE", "R2"):
            recomputed_s2[metric] = m[f"{metric}_recomputed"]
        recomputed_s2["Route"] = recomputed_s2["Route"].replace("", np.nan)
        recomputed_s2[list(s2.columns)].to_csv(out / "data_s2_recomputed.csv", index=False)
        print(f"Wrote {out / 'data_s2_recomputed.csv'}")


if __name__ == "__main__":
    main()
