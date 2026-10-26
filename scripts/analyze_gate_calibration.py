"""Calibration of the PhysicsGate weights over the repeated outer splits.

For every target and outer split, the held-out samples are grouped into ten
equal-count bins of the predicted weight w. Within a bin, the bin-optimal
weight is the single interpolation coefficient between Direct (D) and the
residual-corrected branch (R) that minimizes the squared error of
D + w (R - D) on those samples, restricted to [0, 1]. A calibrated weight
lies on the diagonal of bin-optimal against mean predicted weight.

The squared error of the blended prediction is a proper scoring rule for the
weight. It is decomposed, as for the Brier score, into the loss of the best
constant weight, the resolution gained by ranking samples with w, and the
reliability (miscalibration) cost:

    loss(gate) = loss(best constant) - resolution + reliability,

where resolution = loss(best constant) - loss(bin-optimal weights) and
reliability = loss(gate) - loss(bin-optimal weights). All three use held-out
targets and serve only as diagnostics. Losses are reported relative to the
mean squared error of Direct.

    python scripts/analyze_gate_calibration.py [--samples DATASET=PATH ...] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURE4 = ROOT / "results" / "revised" / "workdir" / "results" / "submitted" / "pv" / "outputs" / "figure4_gate_mechanism" / "50seed"
SAMPLES = {d: FIGURE4 / d.lower() / f"Figure4_{d}_PerSample_GateDiagnostics.csv" for d in ("ESTM", "PV", "SSE", "LMB")}
BINS = 10
BOOT = 2000


def best_weight(direct, residual, y) -> float:
    delta = residual - direct
    denominator = float(np.sum(delta ** 2))
    return float(np.clip(np.sum(delta * (y - direct)) / denominator, 0.0, 1.0)) if denominator > 0 else 0.0


def mse(y, prediction) -> float:
    return float(np.mean((y - prediction) ** 2))


def unit_statistics(unit: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y, d, r, w = (unit[c].to_numpy(dtype=float) for c in ("y_true", "A_direct", "A_residual", "w"))
    bins = pd.qcut(pd.Series(w).rank(method="first"), BINS, labels=False).to_numpy()
    recalibrated = np.empty_like(w)
    curve = []
    for b in range(BINS):
        mask = bins == b
        wb = best_weight(d[mask], r[mask], y[mask])
        recalibrated[mask] = wb
        curve.append({"bin": b + 1, "n": int(mask.sum()), "mean_w": float(w[mask].mean()), "bin_optimal_w": wb})
    curve = pd.DataFrame(curve)
    w0 = best_weight(d, r, y)
    scale = mse(y, d)
    loss_const = mse(y, d + w0 * (r - d)) / scale
    loss_gate = mse(y, d + w * (r - d)) / scale
    loss_recal = mse(y, d + recalibrated * (r - d)) / scale
    stats = {
        "best_constant_w": w0, "mean_w": float(w.mean()), "sd_w": float(w.std()),
        "loss_best_constant": loss_const, "loss_gate": loss_gate, "loss_bin_optimal": loss_recal,
        "resolution": loss_const - loss_recal, "reliability": loss_gate - loss_recal,
        "calibration_error": float(np.average(np.abs(curve["bin_optimal_w"] - curve["mean_w"]), weights=curve["n"])),
    }
    return stats, curve


def load(dataset: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda c: c in {"target", "target_code", "route_target", "seed", "y_true",
                                                       "A_direct", "A_residual", "w"})
    frame["dataset"] = dataset
    frame["endpoint"] = frame["target"].astype(str) + (" " + frame["route_target"].astype(str) if "route_target" in frame else "")
    return frame


def bootstrap(values: np.ndarray, rng) -> tuple[float, float]:
    means = values[rng.integers(0, len(values), size=(BOOT, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", nargs="*", default=[], help="DATASET=PATH overrides of the per-sample files")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revised" / "analyses")
    args = parser.parse_args()
    paths = dict(SAMPLES)
    paths.update({k: Path(v) for k, v in (item.split("=", 1) for item in args.samples)})
    units, curves = [], []
    for dataset, path in paths.items():
        frame = load(dataset, path)
        for (endpoint, seed), unit in frame.groupby(["endpoint", "seed"]):
            stats, curve = unit_statistics(unit)
            units.append({"dataset": dataset, "endpoint": endpoint, "seed": seed, **stats})
            curves.append(curve.assign(dataset=dataset, endpoint=endpoint, seed=seed))
    units, curves = pd.DataFrame(units), pd.concat(curves, ignore_index=True)
    rng = np.random.default_rng(0)
    rows = []
    for (dataset, endpoint), group in units.groupby(["dataset", "endpoint"], sort=False):
        row = {"dataset": dataset, "endpoint": endpoint, "splits": len(group)}
        for column in ("mean_w", "sd_w", "calibration_error", "resolution", "reliability",
                       "loss_best_constant", "loss_gate"):
            low, high = bootstrap(group[column].to_numpy(), rng)
            row.update({column: float(group[column].mean()), f"{column}_CI_low": low, f"{column}_CI_high": high,
                        f"{column}_median": float(group[column].median())})
        bins = curves[curves["dataset"].eq(dataset) & curves["endpoint"].eq(endpoint)]
        slope, intercept = np.polyfit(bins["mean_w"], bins["bin_optimal_w"], 1, w=np.sqrt(bins["n"]))
        row.update({"calibration_slope_pooled": float(slope), "calibration_intercept_pooled": float(intercept)})
        row["splits_gate_better_than_best_constant"] = int((group["loss_gate"] < group["loss_best_constant"]).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    curve_summary = curves.groupby(["dataset", "endpoint", "bin"], sort=False)[["mean_w", "bin_optimal_w"]].mean().reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.out / "gate_calibration_by_split.csv", index=False)
    summary.to_csv(args.out / "gate_calibration_summary.csv", index=False)
    curve_summary.to_csv(args.out / "gate_calibration_curves.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary[["dataset", "endpoint", "mean_w", "sd_w", "calibration_slope_pooled", "calibration_error",
                       "resolution_median", "reliability_median", "loss_best_constant_median", "loss_gate_median",
                       "splits_gate_better_than_best_constant"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
