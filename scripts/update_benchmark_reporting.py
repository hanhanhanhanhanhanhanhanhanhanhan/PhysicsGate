"""Rebuild the five-split benchmark tables and figure worksheets from Data S2.

Writes Tables S4 and S5 and the source data of Figures 2 and 3 for a table in
the layout of Supplementary Data S2. The published files serve as templates:
their rows, labels, order and Origin clipping settings are kept and every
quantity computed from Data S2 is recomputed.

    python scripts/update_benchmark_reporting.py --check
        # recompute from the published Data S2 and compare with the published files

    python scripts/update_benchmark_reporting.py \
        --data-s2 results/revised/workdir/results/submitted/data_s2_recomputed.csv \
        --out results/revised

``--ci exact`` (default) computes exact bootstrap intervals for every endpoint,
as in the revised tables; ``--ci published`` reproduces the intervals of the
submitted tables (tag ``v1.0-submitted``).
"""

from __future__ import annotations

import argparse
import io
import math
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_S2 = ROOT / "source_data" / "supplementary" / "Supplementary_Data_S2_All_Evaluation_Unit_Metrics.csv"
METHODS = ["Direct", "Physics-only", "Residual-only", "Ordinary stacking", "PhysResStack", "PhysicsGate"]
FIGURE3 = {"PhysicsOnly": "Physics-only", "ResidualOnly": "Residual-only", "OrdinaryStacking": "Ordinary stacking",
           "PhysResStack": "PhysResStack", "PhysicsGate": "PhysicsGate"}
# Endpoint labels of the published tables and worksheets -> "Target order" of Data S2
TABLE_S5_ORDER = {"Ea": 1, "ln_sigma0": 2, "ln_sigma_T": 3, "Seebeck Coefficient": 4, "ZT": 5, "lg_sigma_elec": 6,
                  "lg_kappa_thermal": 7, "Energy Density [DE-ED]": 8, "Discharge Energy": 9,
                  "Energy Density [VC-ED]": 10, "Discharge Capacity": 11, "Discharge Voltage": 12,
                  "FF": 13, "JSC": 14, "PCE": 15, "VOC": 16}
DATASET_OF = {**{o: "SSE" for o in (1, 2, 3)}, **{o: "ESTM" for o in (4, 5, 6, 7)},
              **{o: "LMB" for o in range(8, 13)}, **{o: "PV" for o in range(13, 17)}}
FIGURE3_ORDER = {"E_a": 1, "ln sigma_0": 2, "ln sigma T": 3, "Seebeck Coefficient": 4, "ZT": 5, "lg sigma_elec": 6,
                 "lg kappa_thermal": 7, "Energy Density (DE/m route)": 8, "Discharge Energy": 9,
                 "Energy Density (VC/m route)": 10, "Discharge Capacity": 11, "Discharge Voltage": 12,
                 "FF": 13, "J_SC": 14, "PCE": 15, "V_OC": 16}
FIGURE3_AXIS = (-1.2, 1.5)  # x range of the Figure 3 main panels; error bars are cut at its ends
# Figure 2 lists the endpoints in a different order from Data S2.
FIGURE2_ORDER = {**{i: i for i in range(1, 8)}, 13: 8, 14: 9, 15: 10, 16: 11, 9: 12, 11: 13, 10: 14, 8: 15, 12: 16}
FILES = {
    "TableS4": Path("results", "tables", "TableS4.csv"),
    "TableS5": Path("results", "tables", "TableS5.csv"),
    "Figure2_RankMatrix": Path("source_data", "figure2", "Figure2_RankMatrix.csv"),
    "Figure2_MethodSummary": Path("source_data", "figure2", "Figure2_MethodSummary.csv"),
    "Figure3_PhysicsInformed": Path("source_data", "figure3", "Figure3_PhysicsInformed.csv"),
    "Figure3_Ensemble": Path("source_data", "figure3", "Figure3_Ensemble.csv"),
}


def format_sig(value: float, digits: int = 3) -> str:
    """Three significant digits, keeping trailing zeros (as in the published tables)."""
    if value == 0:
        return "0"
    decimals = max(0, digits - math.floor(math.log10(abs(value))) - 1)
    return f"{value:.{decimals}f}"


def paired(direct: np.ndarray, method: np.ndarray) -> dict:
    """Paired gain over the five outer splits with an exact bootstrap interval."""
    gains = np.log2(direct / method)
    n = len(gains)
    resamples = gains[np.array(list(product(range(n), repeat=n)))].mean(axis=1)
    low, high = np.quantile(resamples, [0.025, 0.975])
    mean = float(gains.mean())
    return {"mean": mean, "low": float(low), "high": float(high), "direct": float(direct.mean()),
            "method": float(method.mean()), "wins": int((method < direct).sum()), "n": n}


def published_intervals(s2: pd.DataFrame) -> dict:
    """Intervals of the published tables for SSE, LMB and PV."""
    rng = np.random.default_rng(0)
    out = {}
    for dataset, key, method in sorted((d, k, m) for d in ("ESTM", "LMB", "PV", "SSE")
                                       for k, o in TABLE_S5_ORDER.items() if DATASET_OF[o] == d
                                       for m in METHODS[1:]):
        order = TABLE_S5_ORDER[key]
        size = 15 if dataset == "ESTM" else 5
        draws = rng.integers(0, size, size=(2000, size))
        if dataset == "ESTM":
            continue
        pivot = s2[s2["Target order"].eq(order)].pivot(index="Outer seed", columns="Method", values="RMSE").sort_index()
        gains = np.log2(pivot["Direct"].to_numpy() / pivot[method].to_numpy())
        means = gains[draws].mean(axis=1)
        out[order, method] = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    return out


def endpoint_tables(s2: pd.DataFrame, ci: str = "exact") -> tuple[pd.DataFrame, dict]:
    s2 = s2.copy()
    s2["Route"] = s2["Route"].fillna("")
    keys = ["Target order", "Dataset", "Target", "Target code", "Route"]
    summary = s2.groupby(keys + ["Method"]).agg(
        n=("Outer seed", "nunique"), RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"), R2_mean=("R2", "mean"), R2_sd=("R2", "std")).reset_index()
    summary["Rank"] = summary.groupby(keys)["RMSE_mean"].rank(method="min").astype(int)
    stats = {}
    for order, endpoint in s2.groupby("Target order"):
        pivot = endpoint.pivot(index="Outer seed", columns="Method", values="RMSE").sort_index()
        for method in METHODS[1:]:
            stats[int(order), method] = paired(pivot["Direct"].to_numpy(), pivot[method].to_numpy())
    if ci == "published":
        for key, (low, high) in published_intervals(s2).items():
            stats[key].update(low=low, high=high)
    return summary, stats


def table_s4(template: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    out = template.copy()
    lookup = summary.set_index(["Target order", "Method"])
    for i, row in out.iterrows():
        s = lookup.loc[(int(row["Target order"]), row["Method"])]
        out.loc[i, "N outer seeds"] = int(s["n"])
        for metric in ("RMSE", "MAE", "R2"):
            out.loc[i, f"{metric} mean"] = s[f"{metric}_mean"]
            out.loc[i, f"{metric} s.d."] = s[f"{metric}_sd"]
            out.loc[i, f"{metric} (mean +/- s.d.)"] = (f"{format_sig(s[f'{metric}_mean'])} +/- "
                                                         f"{format_sig(s[f'{metric}_sd'])}")
        out.loc[i, "Rank (RMSE mean)"] = int(s["Rank"])
    # rows are listed by rank within each endpoint
    out = out.sort_values(["Target order", "Rank (RMSE mean)", "Method"], kind="stable").reset_index(drop=True)
    return out


def table_s5(template: pd.DataFrame, stats: dict, ci: str = "exact") -> pd.DataFrame:
    out = template.copy()
    for i, row in out.iterrows():
        st = stats[TABLE_S5_ORDER[row["Target key"]], row["Method"]]
        out.loc[i, "Mean G vs Direct"] = st["mean"]
        out.loc[i, "95% CI low"] = st["low"]
        out.loc[i, "95% CI high"] = st["high"]
        out.loc[i, "Paired evaluation units"] = st["n"]
        out.loc[i, "Mean Direct RMSE"] = st["direct"]
        out.loc[i, "Mean method RMSE"] = st["method"]
        out.loc[i, "Wins vs Direct"] = st["wins"]
        out.loc[i, "Win rate"] = st["wins"] / st["n"]
        out.loc[i, "Equivalent RMSE reduction (%)"] = 100 * (1 - 2 ** (-st["mean"]))
        out.loc[i, "95% CI crosses zero"] = bool(st["low"] <= 0 <= st["high"])
        if ci == "published" and row["Dataset"] == "ESTM":
            out.loc[i, "G (mean, 95% CI)"] = f"{st['mean']:.3g} ({st['low']:.3g}, {st['high']:.3g})"
        else:
            out.loc[i, "G (mean, 95% CI)"] = (f"{format_sig(st['mean'])} ({format_sig(st['low'])}, "
                                             f"{format_sig(st['high'])})")
    return out


def figure2(rank_template: pd.DataFrame, summary_template: pd.DataFrame, summary: pd.DataFrame):
    ranks = summary.assign(fo=summary["Target order"].map(FIGURE2_ORDER)).pivot(
        index="fo", columns="Method", values="Rank")
    matrix = rank_template.copy()
    for i, row in matrix.iterrows():
        for method in METHODS:
            matrix.loc[i, method] = int(ranks.loc[int(row["target_order"]), method])
    counts = pd.DataFrame({
        "method": METHODS,
        "macro_average_rank": [float(ranks[m].mean()) for m in METHODS],
        "rank_1_count": [int((ranks[m] == 1).sum()) for m in METHODS],
        "top_2_count": [int((ranks[m] <= 2).sum()) for m in METHODS],
        "target_count": [int(ranks[m].notna().sum()) for m in METHODS],
    }).set_index("method")
    methods = counts.sort_values(["macro_average_rank", "rank_1_count"], ascending=[True, False]).index
    method_summary = summary_template.set_index("method").reindex(methods).reset_index()
    for column in counts.columns:
        method_summary[column] = counts.loc[methods, column].to_numpy()
    return matrix, method_summary


def figure3(template: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = template.copy()
    for i, row in out.iterrows():
        for prefix, method in FIGURE3.items():
            if f"{prefix}_X" not in out:
                continue
            st = stats[FIGURE3_ORDER[row["Target_Label"]], method]
            values = {"": st["mean"], "ErrMinus": st["mean"] - st["low"], "ErrPlus": st["high"] - st["mean"]}
            for suffix, value in values.items():
                out.loc[i, f"{prefix}_X{suffix}"] = value
            flag = row[f"{prefix}_ClipFlag"]
            if isinstance(flag, str) and flag:
                # Keep the Origin arrow position; the label carries the underlying gain.
                if not (flag == "low" and st["mean"] < float(row[f"{prefix}_MainPlotX"])):
                    raise ValueError(f"{row['Target_Label']} {method}: clipping needs a new Origin setting")
                out[f"{prefix}_ClipLabel"] = out[f"{prefix}_ClipLabel"].astype(object)
                out.loc[i, f"{prefix}_ClipLabel"] = f"{st['mean']:.2f}"
            else:
                low_end, high_end = FIGURE3_AXIS
                out.loc[i, f"{prefix}_MainPlotX"] = values[""]
                out.loc[i, f"{prefix}_MainPlotXErrMinus"] = min(values["ErrMinus"], values[""] - low_end)
                out.loc[i, f"{prefix}_MainPlotXErrPlus"] = min(values["ErrPlus"], high_end - values[""])
    return out


def build(data_s2: Path, ci: str = "exact") -> dict[str, pd.DataFrame]:
    s2 = pd.read_csv(data_s2)
    templates = {name: pd.read_csv(ROOT / path, encoding="utf-8-sig", keep_default_na=False,
                                   na_values=[""]) for name, path in FILES.items()}
    summary, stats = endpoint_tables(s2, ci)
    matrix, method_summary = figure2(templates["Figure2_RankMatrix"], templates["Figure2_MethodSummary"], summary)
    return {
        "TableS4": table_s4(templates["TableS4"], summary),
        "TableS5": table_s5(templates["TableS5"], stats, ci),
        "Figure2_RankMatrix": matrix,
        "Figure2_MethodSummary": method_summary,
        "Figure3_PhysicsInformed": figure3(templates["Figure3_PhysicsInformed"], stats),
        "Figure3_Ensemble": figure3(templates["Figure3_Ensemble"], stats),
    }


def compare(name: str, published: pd.DataFrame, rebuilt: pd.DataFrame) -> str:
    if published.shape != rebuilt.shape or list(published.columns) != list(rebuilt.columns):
        return f"{name}: layout {published.shape} vs {rebuilt.shape}"
    worst, text = 0.0, []
    for c in published.columns:
        a, b = published[c], rebuilt[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(float), b.to_numpy(float)
            d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y) / np.maximum(1.0, np.abs(x)))
            worst = max(worst, float(np.nanmax(d, initial=0.0)))
        elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
            text.append(c)
    status = "identical" if worst < 1e-9 and not text else "differs"
    return f"{name}: {status} (max relative numeric difference {worst:.2e}; text columns differing: {text})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-s2", type=Path, default=DATA_S2)
    parser.add_argument("--out", type=Path, help="directory for the rebuilt files (layout as in the repository)")
    parser.add_argument("--check", action="store_true", help="compare the rebuilt files with the published ones")
    parser.add_argument("--ci", choices=["exact", "published"], default="exact")
    args = parser.parse_args()
    rebuilt = build(args.data_s2, args.ci)
    if args.check:  # compare as written to and read back from CSV, like the published files
        for name, path in FILES.items():
            published = pd.read_csv(ROOT / path, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
            buffer = io.StringIO()
            rebuilt[name].to_csv(buffer, index=False)
            buffer.seek(0)
            print(compare(name, published, pd.read_csv(buffer, keep_default_na=False, na_values=[""])))
    if args.out:
        out = args.out.resolve()
        for name, path in FILES.items():
            target = out / path
            target.parent.mkdir(parents=True, exist_ok=True)
            quote = 1 if path.parent.name in ("figure2", "figure3") else 0  # csv.QUOTE_ALL for Origin worksheets
            rebuilt[name].to_csv(target, index=False, encoding="utf-8-sig", quoting=quote)
        target = out / "source_data" / "supplementary" / DATA_S2.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.data_s2, target)
        print(f"Wrote the rebuilt tables and worksheets under {out}")


if __name__ == "__main__":
    main()
