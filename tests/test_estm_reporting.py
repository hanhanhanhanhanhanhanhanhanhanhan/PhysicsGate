from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "estm_reporting", ROOT / "scripts/update_estm_reporting.py"
)
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def test_statistics_pair_by_seed_and_bootstrap_seed_gains():
    direct = dict(zip("abcde", [2, 4, 8, 16, 32]))
    method = dict(zip("edcba", [16, 8, 4, 2, 1]))
    result = reporting.paired_statistics(direct, method)
    assert result["Paired evaluation units"] == 5
    assert result["Mean G vs Direct"] == 1
    assert result["95% CI low"] == result["95% CI high"] == 1
    assert result["Wins vs Direct"] == 5
    assert result["Equivalent RMSE reduction (%)"] == 50


def test_mixed_seed_gains_not_ratio_of_mean_rmse():
    direct = dict(zip("abcde", [1, 2, 3, 4, 5]))
    method = dict(zip("abcde", [2, 2, 1.5, 1, 2.5]))
    result = reporting.paired_statistics(direct, method)
    assert result["Mean G vs Direct"] == pytest.approx(0.6)
    assert result["Wins vs Direct"] == 3
    assert result["Win rate"] == 0.6
    assert not np.isclose(result["Mean G vs Direct"], np.log2(3 / 1.8))
    assert result["95% CI low"] <= 0 <= result["95% CI high"]


def test_reject_mismatched_or_fold_level_units():
    with pytest.raises(ValueError):
        reporting.paired_statistics({str(i): 1 for i in range(15)}, {str(i): 2 for i in range(15)})
    with pytest.raises(ValueError):
        reporting.paired_statistics(dict.fromkeys("abcde", 1), dict.fromkeys("abcdf", 1))


def test_published_estm_statistics_match_data_s2():
    reporting.rebuild(check=True)
