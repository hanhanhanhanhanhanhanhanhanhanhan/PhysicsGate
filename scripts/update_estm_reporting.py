"""Align ESTM Table S5 and Figure 3 worksheets with five-seed Data S2."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET_ORDERS = {
    "Seebeck Coefficient": "4",
    "ZT": "5",
    "lg_sigma_elec": "6",
    "lg_kappa_thermal": "7",
}
FIGURE_METHODS = {
    "PhysicsOnly": "Physics-only",
    "ResidualOnly": "Residual-only",
    "OrdinaryStacking": "Ordinary stacking",
    "PhysResStack": "PhysResStack",
    "PhysicsGate": "PhysicsGate",
}


def paired_statistics(direct: dict[str, float], method: dict[str, float]) -> dict:
    if len(direct) != 5 or direct.keys() != method.keys():
        raise ValueError("Exactly five matching outer seeds are required")
    seeds = sorted(direct)
    d = np.array([direct[seed] for seed in seeds], dtype=float)
    m = np.array([method[seed] for seed in seeds], dtype=float)
    if not (np.isfinite(d).all() and np.isfinite(m).all()
            and (d > 0).all() and (m > 0).all()):
        raise ValueError("RMSE values must be finite and positive")
    gains = np.log2(d / m)
    # Enumerate all 5**5 equally likely paired bootstrap resamples.
    indices = np.array(list(product(range(5), repeat=5)))
    low, high = np.quantile(gains[indices].mean(axis=1), [0.025, 0.975])
    mean = float(gains.mean())
    return {
        "Mean G vs Direct": mean,
        "95% CI low": float(low),
        "95% CI high": float(high),
        "Paired evaluation units": 5,
        "Mean Direct RMSE": float(d.mean()),
        "Mean method RMSE": float(m.mean()),
        "Wins vs Direct": int((m < d).sum()),
        "Win rate": float((m < d).mean()),
        "Equivalent RMSE reduction (%)": float(100 * (1 - 2**(-mean))),
        "95% CI crosses zero": bool(low <= 0 <= high),
        "G (mean, 95% CI)": f"{mean:.3g} ({low:.3g}, {high:.3g})",
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rebuild(*, root: Path = ROOT, check: bool = False) -> None:
    source = root / "source_data/supplementary/Supplementary_Data_S2_All_Evaluation_Unit_Metrics.csv"
    bank: dict[tuple[str, str], dict[str, float]] = {}
    for row in read_rows(source):
        if row["Dataset"] != "ESTM":
            continue
        key = (row["Target order"], row["Method"])
        unit = bank.setdefault(key, {})
        seed = row["Outer seed"]
        if seed in unit:
            raise ValueError(f"Duplicate seed-level metric: {key}, {seed}")
        unit[seed] = float(row["RMSE"])
    if len(bank) != 24:
        raise ValueError("Expected four ESTM targets and six methods in Data S2")
    stats = {
        (order, method): paired_statistics(bank[order, "Direct"], values)
        for (order, method), values in bank.items()
        if method != "Direct"
    }
    paths = [root / "results/tables/TableS5.csv"] + [
        root / f"source_data/figure3/Figure3_{name}.csv"
        for name in ("PhysicsInformed", "Ensemble")
    ]
    prepared = []
    for path in paths:
        original = read_rows(path)
        rows = [row.copy() for row in original]
        for row in rows:
            if row["Dataset"] != "ESTM":
                continue
            if path.name == "TableS5.csv":
                result = stats[TARGET_ORDERS[row["Target key"]], row["Method"]]
                row.update({key: str(value) for key, value in result.items()})
            else:
                for prefix, method in FIGURE_METHODS.items():
                    if f"{prefix}_X" not in row:
                        continue
                    result = stats[row["Global_Order"], method]
                    mean = result["Mean G vs Direct"]
                    values = {
                        "": mean,
                        "ErrMinus": mean - result["95% CI low"],
                        "ErrPlus": result["95% CI high"] - mean,
                    }
                    for suffix, value in values.items():
                        row[f"{prefix}_X{suffix}"] = str(value)
                    flag = row[f"{prefix}_ClipFlag"]
                    if flag:
                        # Preserve the existing Origin arrow position, not the old gain.
                        position = float(row[f"{prefix}_MainPlotX"])
                        if not (flag == "low" and mean < position):
                            raise ValueError("Updated point needs a revised Origin clipping policy")
                        row[f"{prefix}_ClipLabel"] = f"{mean:.2f}"
                    else:
                        for suffix, value in values.items():
                            row[f"{prefix}_MainPlotX{suffix}"] = str(value)
        prepared.append((path, original, rows))
    for path, original, rows in prepared:
        if original == rows:
            continue
        if check:
            raise ValueError(f"Five-seed statistics are stale: {path}")
        encoding = "utf-8-sig" if path.read_bytes().startswith(b"\xef\xbb\xbf") else "utf-8"
        quoting = csv.QUOTE_ALL if path.parent.name == "figure3" else csv.QUOTE_MINIMAL
        with path.open("w", encoding=encoding, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=quoting)
            writer.writeheader()
            writer.writerows(rows)
    print("ESTM Table S5 and Figure 3 use five paired outer-seed units.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    rebuild(check=parser.parse_args().check)
