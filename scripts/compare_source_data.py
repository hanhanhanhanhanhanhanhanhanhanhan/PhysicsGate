"""Compare regenerated figure source data with ``source_data/``, file by file.

    python scripts/compare_source_data.py results/revised/source_data

For every published table, reports the largest relative difference over its
numeric columns (relative to max(1, |published value|)) and any differing text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("figure4", "figure5", "figureS1", "figureS2", "figureS3")


def compare(published: Path, regenerated: Path) -> dict:
    if not regenerated.exists():
        return {"status": "missing"}
    a = pd.read_csv(published, encoding="utf-8-sig")
    b = pd.read_csv(regenerated, encoding="utf-8-sig")
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return {"status": f"layout {a.shape} vs {b.shape}"}
    worst, column, text = 0.0, "", []
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            d = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y) / np.maximum(1.0, np.abs(x)))
            d = np.where(np.isnan(x) ^ np.isnan(y), np.inf, d)
            if np.nanmax(d, initial=0.0) > worst:
                worst, column = float(np.nanmax(d)), c
        elif not a[c].astype(str).equals(b[c].astype(str)):
            text.append(c)
    status = "identical" if worst < 1e-9 and not text else "differs"
    return {"status": status, "max_rel_diff": worst, "column": column, "text_columns": ", ".join(text)}


def main() -> None:
    regenerated = Path(sys.argv[1]).resolve()
    rows = []
    for folder in FOLDERS:
        for published in sorted((ROOT / "source_data" / folder).glob("*.csv")):
            rows.append({"figure": folder, "file": published.name,
                         **compare(published, regenerated / folder / published.name)})
    table = pd.DataFrame(rows)
    table.to_csv(regenerated / "comparison_with_source_data.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
