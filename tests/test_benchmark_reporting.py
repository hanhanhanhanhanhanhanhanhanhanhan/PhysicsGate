from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_reporting", ROOT / "scripts/update_benchmark_reporting.py"
)
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def read(source) -> pd.DataFrame:
    return pd.read_csv(source, encoding="utf-8-sig", keep_default_na=False, na_values=[""])


def test_published_tables_and_worksheets_follow_from_data_s2():
    rebuilt = reporting.build(reporting.DATA_S2)
    for name, path in reporting.FILES.items():
        buffer = io.StringIO()
        rebuilt[name].to_csv(buffer, index=False)
        buffer.seek(0)
        result = reporting.compare(name, read(ROOT / path), read(buffer))
        assert ": identical" in result, result
