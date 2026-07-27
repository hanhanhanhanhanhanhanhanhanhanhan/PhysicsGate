from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import physicsgate.benchmark as benchmark


def test_sse_unified_runner_emits_six_methods(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    n = 60
    temperature = rng.uniform(280, 420, n)
    ea = 0.15 + 0.03 * rng.normal(size=n)
    ln_sigma0 = 2.0 + 0.2 * rng.normal(size=n)
    ln_sigma_t = (
        ln_sigma0
        - ea / (benchmark.ARRHENIUS_KB_EV_PER_K * temperature)
        + 0.03 * rng.normal(size=n)
    )
    frame = pd.DataFrame(
        {
            "group_id": [f"g{i // 3:02d}" for i in range(n)],
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "T_K": temperature,
            "ln_sigma_T": ln_sigma_t,
            "Ea": ea,
            "ln_sigma0": ln_sigma0,
        }
    )
    path = tmp_path / "sse.csv"
    frame.to_csv(path, index=False)
    path.with_suffix(".schema.json").write_text(
        json.dumps(
            {
                "dataset": "SSE",
                "feature_columns": ["x1", "x2", "T_K"],
                "target_columns": ["ln_sigma_T", "Ea", "ln_sigma0"],
                "group_column": "group_id",
                "context_columns": ["T_K"],
            }
        ),
        encoding="utf-8",
    )
    simple = lambda *_args, **_kwargs: make_pipeline(  # noqa: E731
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    monkeypatch.setattr(benchmark, "_direct_model", simple)
    monkeypatch.setattr(
        benchmark,
        "_ordinary_models",
        lambda *_args, **_kwargs: [simple(), simple(), simple()],
    )
    monkeypatch.setitem(
        benchmark.ENDPOINTS,
        "SSE",
        (benchmark.ENDPOINTS["SSE"][0],),
    )
    result = benchmark.run_benchmark(
        "SSE",
        path,
        split_ids=[14],
        output_path=tmp_path / "metrics.csv",
        n_jobs=1,
    )
    assert set(result["method"]) == set(benchmark.METHODS)
    assert len(result) == 6
    assert not result["uses_true_test_auxiliary_values"].any()
    assert result["uses_oof_second_stage_predictions"].all()
