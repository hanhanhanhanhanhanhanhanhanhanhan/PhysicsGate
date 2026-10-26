"""Training and inference cost of the six methods from the component timings.

Reads the records written by ``physicsgate.revised.cost_audit`` and assembles,
for every endpoint and outer training set (an ESTM fold counts as one training
set), the cost of training each method from scratch:

- Direct: model selection and final fit of the target model;
- Physics-only: the same for the auxiliary models, plus the equation;
- Residual-only: the auxiliary models with out-of-fold (OOF) predictions, the
  Direct model with OOF predictions where it is a residual input (ESTM, LMB,
  PV), and the residual model;
- Ordinary stacking: its base-model pool with OOF predictions and the
  meta-learner (SSE and LMB use a separate pool; ESTM and PV stack the Direct,
  auxiliary and Physics-only predictions);
- PhysResStack: the Residual-only branch, the Direct model with OOF
  predictions and the meta-learner;
- PhysicsGate: the same branches, R3 and the gate.

The OOF fits are the inner-CV fits that precede the final fit of a model. Base
models that LMB and PV share across endpoints are charged in full to every
endpoint that uses them. Inference is the time the fitted models take to
predict the test set of the outer split (one batch); the equation, R3 and the
gate add vectorized array operations to it.

    python scripts/analyze_cost.py [--records DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "cost_audit" / "workdir" / "results" / "submitted" / "cost_audit"
STAGES = {"sse_pipeline": "SSE", "sse_ordinary": "SSE", "revised_estm_pipeline": "ESTM", "lmb_pipeline": "LMB",
          "lmb_complete": "LMB", "pv_selection": "PV", "revised_pv_pipeline": "PV"}
INNER_CV = {"SSE": 3, "ESTM": 5, "LMB": 5, "PV": 5}
LMB_CODE = {"energy_density_wh_kg": "ED", "normal_discharge_voltage_v": "V", "discharge_capacity_ah": "C",
            "discharge_energy_wh": "DE"}
LMB_AUX = {("VC", "ED"): ("V", "C"), ("VC", "V"): ("ED", "C"), ("VC", "C"): ("ED", "V"),
           ("DE", "ED"): ("DE",), ("DE", "DE"): ("ED",)}
PV_TARGETS = ("Voc_V", "Jsc_mA_cm2", "FF_fraction", "PCE_percent")
METHODS = ["Direct", "Physics-only", "Residual-only", "Ordinary stacking", "PhysResStack", "PhysicsGate"]


def invocation_costs(components: pd.DataFrame, calls: pd.DataFrame, k: int) -> pd.DataFrame:
    """Wall and CPU time of each component call, its OOF share and its test-set prediction."""
    rows = []
    for invocation, group in calls.groupby("invocation", sort=False):
        group = group.reset_index(drop=True)
        fits = group.index[group["kind"].eq("fit")]
        oof = test_predict = 0.0
        test_rows = np.nan
        if len(fits):
            final = fits[-1]
            if len(fits) > k:
                oof = float(group.loc[fits[-k - 1]:final - 1, "wall_s"].sum())
            after = group.loc[final + 1:]
            predicts = after[after["kind"].eq("predict")]
            if len(predicts):
                test_predict, test_rows = float(predicts["wall_s"].iloc[-1]), predicts["rows"].iloc[-1]
        rows.append({"invocation": invocation, "oof_s": oof, "test_predict_s": test_predict, "test_rows": test_rows})
    out = components.merge(pd.DataFrame(rows), on="invocation", how="left")
    return out.fillna({"oof_s": 0.0, "test_predict_s": 0.0})


def load(records: Path) -> pd.DataFrame:
    parts = []
    for tag, dataset in STAGES.items():
        path = records / f"{tag}_components.csv"
        if not path.exists():
            continue
        components = pd.read_csv(path)
        calls = pd.read_csv(records / f"{tag}_calls.csv")
        part = invocation_costs(components, calls, INNER_CV[dataset])
        part["dataset"], part["tag"] = dataset, tag
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summed(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = ["wall_s", "cpu_s", "oof_s", "test_predict_s"]
    return frame.groupby([*keys, "component"])[cols].sum().unstack("component").fillna(0.0)


def pick(table: pd.DataFrame, component: str, column: str = "wall_s") -> pd.Series:
    key = (column, component)
    return table[key] if key in table.columns else pd.Series(0.0, index=table.index)


def unit_parts(c: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset, outer training set and endpoint, with the cost of each part."""
    rows = []

    def add(dataset, seed, fold, endpoint, parts):
        rows.append({"dataset": dataset, "seed": seed, "fold": fold, "endpoint": endpoint, **parts})

    def parts_of(direct, aux, own, n_aux=None):
        out = {}
        for metric in ("wall_s", "cpu_s", "oof_s", "test_predict_s"):
            out[f"direct_{metric}"] = direct[metric]
            out[f"aux_{metric}"] = sum(a[metric] for a in aux)
            for name in ("physics", "residual", "R3", "gate", "ordinary", "physres"):
                out[f"{name}_{metric}"] = own.get((metric, name), 0.0)
        out["test_rows"] = direct["test_rows"]
        return out

    def model(table, key, labels):
        cols = {}
        for metric in ("wall_s", "cpu_s", "oof_s", "test_predict_s"):
            cols[metric] = sum(table.get((metric, label), pd.Series(dtype=float)).get(key, 0.0) for label in labels)
        return cols

    for dataset in ("SSE", "ESTM"):
        d = c[c["dataset"].eq(dataset)]
        if d.empty:
            continue
        own = summed(d, ["seed", "fold", "unit_target"])
        test_rows = d[d["component"].eq("direct")].groupby(["seed", "fold", "unit_target"])["test_rows"].last()
        for key in own.index:
            seed, fold, target = key
            row = own.loc[key]
            direct = {m: row.get((m, "direct"), 0.0) for m in ("wall_s", "cpu_s", "oof_s", "test_predict_s")}
            direct["test_rows"] = test_rows.get(key, np.nan)
            aux = [{m: row.get((m, "auxiliary"), 0.0) for m in ("wall_s", "cpu_s", "oof_s", "test_predict_s")}]
            add(dataset, seed, fold, target, parts_of(direct, aux, row.to_dict()))
    d = c[c["dataset"].eq("LMB")]
    if not d.empty:
        d = d.assign(code=d["model_target"].map(LMB_CODE), unit_code=d["unit_target"].map(LMB_CODE))
        base = summed(d[d["component"].isin(["base_selection", "base_oof"])].assign(component="base"), ["seed", "code"])
        own = summed(d[d["component"].isin(["physics", "residual", "R3", "gate", "physres"])], ["seed", "route", "unit_code"])
        ordinary = summed(d[d["component"].eq("ordinary")], ["seed", "code"])
        rows_test = d[d["component"].eq("base_oof")].groupby(["seed", "code"])["test_rows"].last()
        for (seed, route, code) in own.index:
            row = own.loc[(seed, route, code)].to_dict()
            for metric in ("wall_s", "cpu_s", "oof_s", "test_predict_s"):
                row[(metric, "ordinary")] = ordinary.get((metric, "ordinary"), pd.Series(dtype=float)).get((seed, code), 0.0)
            direct = {m: base[(m, "base")].get((seed, code), 0.0) for m in ("wall_s", "cpu_s", "oof_s", "test_predict_s")}
            direct["test_rows"] = rows_test.get((seed, code), np.nan)
            aux = [{m: base[(m, "base")].get((seed, a), 0.0) for m in ("wall_s", "cpu_s", "oof_s", "test_predict_s")}
                   for a in LMB_AUX[route, code]]
            add("LMB", seed, 0, f"{route}-{code}", parts_of(direct, aux, row))
    d = c[c["dataset"].eq("PV")]
    if not d.empty:
        base = summed(d[d["component"].eq("base")], ["seed", "model_target"])
        own = summed(d[d["tag"].eq("revised_pv_pipeline")], ["seed", "unit_target"])
        rows_test = d[d["component"].eq("base")].groupby(["seed", "model_target"])["test_rows"].last()
        for (seed, target) in own.index:
            row = own.loc[(seed, target)].to_dict()
            direct = {m: base[(m, "base")].get((seed, target), 0.0) for m in ("wall_s", "cpu_s", "oof_s", "test_predict_s")}
            direct["test_rows"] = rows_test.get((seed, target), np.nan)
            aux = [{m: base[(m, "base")].get((seed, a), 0.0) for m in ("wall_s", "cpu_s", "oof_s", "test_predict_s")}
                   for a in PV_TARGETS if a != target]
            add("PV", seed, 0, target, parts_of(direct, aux, row))
    return pd.DataFrame(rows)


def method_costs(u: pd.DataFrame) -> pd.DataFrame:
    """Training (wall and CPU seconds) and test-set prediction time (s) of each method."""
    out = u[["dataset", "seed", "fold", "endpoint"]].copy()
    # OOF fits are not needed by a model used on its own; the OOF share of CPU time is taken pro rata
    oof_share = {p: (u[f"{p}_oof_s"] / u[f"{p}_wall_s"].where(u[f"{p}_wall_s"] > 0)).fillna(0.0)
                 for p in ("direct", "aux")}
    for metric in ("wall_s", "cpu_s"):
        alone = {p: u[f"{p}_{metric}"] * (1 - oof_share[p]) for p in ("direct", "aux")}
        full = {p: u[f"{p}_{metric}"] for p in ("direct", "aux", "physics", "residual", "R3", "gate", "ordinary", "physres")}
        direct_in_residual = (u["dataset"] != "SSE").astype(float)
        residual_only = full["aux"] + full["physics"] + full["residual"] + direct_in_residual * full["direct"]
        stack_base = np.where(u["dataset"].isin(["ESTM", "PV"]), full["direct"] + full["aux"] + full["physics"], 0.0)
        out[f"Direct {metric}"] = alone["direct"]
        out[f"Physics-only {metric}"] = alone["aux"] + full["physics"]
        out[f"Residual-only {metric}"] = residual_only
        out[f"Ordinary stacking {metric}"] = stack_base + full["ordinary"]
        out[f"PhysResStack {metric}"] = full["direct"] + full["aux"] + full["physics"] + full["residual"] + full["physres"]
        out[f"PhysicsGate {metric}"] = (full["direct"] + full["aux"] + full["physics"] + full["residual"]
                                        + full["R3"] + full["gate"])
        out[f"gate head {metric}"] = full["R3"] + full["gate"]
        out[f"PhysResStack head {metric}"] = full["physres"]
    predict = {p: u[f"{p}_test_predict_s"] for p in ("direct", "aux", "residual", "ordinary", "physres", "gate")}
    out["test_rows"] = u["test_rows"]
    out["Direct inference_s"] = predict["direct"]
    out["Physics-only inference_s"] = predict["aux"]
    out["Residual-only inference_s"] = predict["aux"] + predict["residual"] + (u["dataset"] != "SSE") * predict["direct"]
    out["PhysResStack inference_s"] = predict["direct"] + predict["aux"] + predict["residual"] + predict["physres"]
    out["PhysicsGate inference_s"] = predict["direct"] + predict["aux"] + predict["residual"] + predict["gate"]
    return out


def dataset_totals(u: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """Training wall time of all endpoints of a dataset on one outer training set.

    In ESTM, LMB and PV the auxiliary models of an endpoint are the Direct models
    of the other endpoints and are trained once; SSE refits its auxiliary models
    for every anchor target, so its endpoints are summed.
    """
    u = u.assign(target=u["endpoint"].str.split("-").str[-1])
    rows = []
    for (dataset, seed, fold), group in u.groupby(["dataset", "seed", "fold"], sort=False):
        c = costs.loc[group.index]
        row = {"dataset": dataset, "seed": seed, "fold": fold, "endpoints": len(group),
               "Direct": c["Direct wall_s"].groupby(group["target"]).first().sum()}
        if dataset == "SSE":
            row["PhysResStack"], row["PhysicsGate"] = c["PhysResStack wall_s"].sum(), c["PhysicsGate wall_s"].sum()
        else:
            models = group.groupby("target")["direct_wall_s"].first().sum()
            branch = group["physics_wall_s"] + group["residual_wall_s"]
            row["PhysResStack"] = models + (branch + group["physres_wall_s"]).sum()
            row["PhysicsGate"] = models + (branch + group["R3_wall_s"] + group["gate_wall_s"]).sum()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revised" / "analyses")
    args = parser.parse_args()
    components = load(args.records)
    units = unit_parts(components)
    costs = method_costs(units)
    numeric = [c for c in costs.columns if c not in ("dataset", "seed", "fold", "endpoint")]
    by_dataset = costs.groupby("dataset", sort=False)[numeric].median()
    totals = dataset_totals(units, costs)
    totals_median = totals.groupby("dataset", sort=False)[["endpoints", "Direct", "PhysResStack", "PhysicsGate"]].median()
    for m in ("PhysResStack", "PhysicsGate"):
        totals_median[f"{m} / Direct"] = totals_median[m] / totals_median["Direct"]
    for metric in ("wall_s", "cpu_s"):
        for m in METHODS:
            by_dataset[f"{m} {metric} / Direct"] = by_dataset[f"{m} {metric}"] / by_dataset[f"Direct {metric}"]
    process = pd.concat([pd.read_csv(p) for p in sorted(args.records.glob("*_process.csv"))], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.out / "cost_parts_by_unit.csv", index=False)
    costs.to_csv(args.out / "cost_methods_by_unit.csv", index=False)
    by_dataset.to_csv(args.out / "cost_methods_by_dataset.csv")
    totals.to_csv(args.out / "cost_dataset_totals.csv", index=False)
    process.to_csv(args.out / "cost_environment.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("Training wall time per endpoint and outer training set (median, s):")
        print(by_dataset[[f"{m} wall_s" for m in METHODS] + ["gate head wall_s", "PhysResStack head wall_s"]].round(2).to_string())
        print("\nTraining CPU time (median, s):")
        print(by_dataset[[f"{m} cpu_s" for m in METHODS]].round(1).to_string())
        print("\nRatio to Direct (wall):")
        print(by_dataset[[f"{m} wall_s / Direct" for m in METHODS]].round(2).to_string())
        print("\nAll endpoints of a dataset on one outer training set (median wall time, s):")
        print(totals_median.round(2).to_string())
        print("\nTest-set prediction (median, s):")
        print(by_dataset[["test_rows"] + [c for c in numeric if c.endswith("inference_s")]].round(3).to_string())
        print()
        print(process[["stage", "wall_s", "peak_memory_MB", "logical_cpus", "processor", "sklearn", "xgboost"]].to_string(index=False))


if __name__ == "__main__":
    main()
