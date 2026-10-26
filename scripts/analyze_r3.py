"""Empirical assessment of the propagated auxiliary-error descriptor R3.

Reads the per-sample records written by ``physicsgate.revised.r3_audit`` for
the five evaluated outer splits and reports, for every endpoint:

- how well R3 ranks the observed Physics-only error |y - P| of individual test
  samples (Spearman correlation, computed within each split and averaged), and
  the ratio of the RMS error in the top and bottom R3 deciles, over all samples
  and over the samples with R3 > 0 (R3 is zero where the physics prediction is
  held at the training-range bound, so that its derivative vanishes);
- how closely the first-order propagation in R3 agrees with Monte Carlo
  propagation of independent Gaussian auxiliary errors through the full
  equation (nonlinearity);
- how the propagated spread changes when auxiliary errors are resampled from
  the training-side out-of-fold residual vectors (correlated, non-Gaussian
  errors), and whether that spread ranks the observed errors better.

    python scripts/analyze_r3.py [--records DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "r3_audit" / "workdir" / "results" / "submitted" / "r3_audit"
LABELS = {
    ("SSE", "E"): "SSE Ea", ("SSE", "P"): "SSE ln sigma0", ("SSE", "L"): "SSE ln(sigma T)",
    ("ESTM", "seebeck_uV_per_K"): "ESTM S", ("ESTM", "ZT"): "ESTM ZT",
    ("ESTM", "log10_electrical_conductivity"): "ESTM log10 sigma", ("ESTM", "log10_thermal_conductivity"): "ESTM log10 kappa",
    ("LMB-DE", "discharge_energy_wh"): "LMB DE (Ed m)", ("LMB-DE", "energy_density_wh_kg"): "LMB Ed (DE/m)",
    ("LMB-VC", "energy_density_wh_kg"): "LMB Ed (QV/m)", ("LMB-VC", "discharge_capacity_ah"): "LMB Q (Ed m/V)",
    ("LMB-VC", "normal_discharge_voltage_v"): "LMB V (Ed m/Q)",
    ("PV", "FF_fraction"): "PV FF", ("PV", "Jsc_mA_cm2"): "PV JSC", ("PV", "PCE_percent"): "PV PCE", ("PV", "Voc_V"): "PV VOC",
}


def rho(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(spearmanr(a[ok], b[ok]).statistic)


def decile_ratio(r3: np.ndarray, error: np.ndarray) -> float:
    ok = np.isfinite(r3) & np.isfinite(error)
    r3, error = r3[ok], error[ok]
    if np.ptp(r3) == 0:
        return np.nan
    low, high = np.quantile(r3, [0.1, 0.9])
    bottom, top = error[r3 <= low], error[r3 >= high]
    return float(np.sqrt(np.mean(top ** 2)) / np.sqrt(np.mean(bottom ** 2)))


def split_statistics(unit: pd.DataFrame) -> dict:
    r3 = unit["R3"].to_numpy()
    error = np.abs(unit["y_true"].to_numpy() - unit["physics"].to_numpy())
    error_raw = np.abs(unit["y_true"].to_numpy() - unit["physics_raw"].to_numpy())
    mc_ind, mc_res = unit["sd_mc_independent"].to_numpy(), unit["sd_mc_residual"].to_numpy()
    positive = r3 > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_ind, ratio_res = mc_ind[positive] / r3[positive], mc_res[positive] / r3[positive]
    return {
        "n": len(unit), "fraction_R3_zero": float(np.mean(~positive)),
        "rho_R3_error": rho(r3, error), "rho_R3_error_unclipped": rho(r3, error_raw),
        "rms_error_top_bottom_decile": decile_ratio(r3, error),
        "rho_R3_error_R3_positive": rho(r3[positive], error[positive]),
        "rms_error_top_bottom_decile_R3_positive": decile_ratio(r3[positive], error[positive]),
        "rms_error_R3_zero_over_positive": (float(np.sqrt(np.mean(error[~positive] ** 2)) / np.sqrt(np.mean(error[positive] ** 2)))
                                            if (~positive).sum() >= 5 else np.nan),
        "rho_R3_mc_independent": rho(r3, mc_ind), "median_mc_independent_over_R3": float(np.nanmedian(ratio_ind)),
        "fraction_within_25pct": float(np.nanmean((ratio_ind > 0.8) & (ratio_ind < 1.25))),
        "rho_R3_mc_residual": rho(r3, mc_res), "median_mc_residual_over_R3": float(np.nanmedian(ratio_res)),
        "rho_mc_residual_error": rho(mc_res, error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=RECORDS)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "revised" / "analyses")
    args = parser.parse_args()
    records = pd.concat([pd.read_csv(p) for p in sorted(args.records.glob("*.csv.gz"))], ignore_index=True)
    rows = []
    for (dataset, target, seed), unit in records.groupby(["dataset", "target", "seed"]):  # ESTM folds pooled per split
        rows.append({"endpoint": LABELS[dataset, target], "dataset": dataset.split("-")[0], "seed": seed,
                     **split_statistics(unit)})
    by_split = pd.DataFrame(rows)
    order = list(LABELS.values())
    numeric = [c for c in by_split.columns if c not in ("endpoint", "dataset", "seed")]
    summary = by_split.groupby("endpoint")[numeric].mean().reindex(order)
    summary["rho_R3_error_min"] = by_split.groupby("endpoint")["rho_R3_error"].min().reindex(order)
    summary["rho_R3_error_max"] = by_split.groupby("endpoint")["rho_R3_error"].max().reindex(order)
    args.out.mkdir(parents=True, exist_ok=True)
    by_split.to_csv(args.out / "r3_audit_by_split.csv", index=False)
    summary.to_csv(args.out / "r3_audit_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary[["fraction_R3_zero", "rho_R3_error", "rho_R3_error_min", "rho_R3_error_max",
                       "rms_error_top_bottom_decile", "rho_R3_error_R3_positive",
                       "rms_error_top_bottom_decile_R3_positive", "rms_error_R3_zero_over_positive",
                       "rho_R3_mc_independent", "median_mc_independent_over_R3",
                       "fraction_within_25pct", "rho_R3_mc_residual", "median_mc_residual_over_R3",
                       "rho_mc_residual_error"]].round(3).to_string())


if __name__ == "__main__":
    main()
