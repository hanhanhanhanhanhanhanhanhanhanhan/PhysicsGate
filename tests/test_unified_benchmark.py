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


def test_propagated_proxy_uses_local_derivative_times_oof_rmse(monkeypatch):
    endpoint = benchmark.Endpoint("toy", "toy", ("aux",), "toy")
    predictions = {"aux": np.array([2.0])}

    def cubic_physics(_dataset, _endpoint, values, _context, **_kwargs):
        return np.asarray(values["aux"], dtype=float) ** 3

    monkeypatch.setattr(benchmark, "_physics", cubic_physics)
    proxy = benchmark._propagated_auxiliary_error_proxy(
        "TOY",
        endpoint,
        predictions,
        pd.DataFrame(index=[0]),
        direct_target=np.array([0.0]),
        target_range=(-np.inf, np.inf),
        auxiliary_rmse={"aux": 0.5},
    )
    step = max(
        benchmark.FINITE_DIFFERENCE_ABSOLUTE_STEP,
        benchmark.FINITE_DIFFERENCE_RELATIVE_STEP * 2.0,
    )
    expected_derivative = 3.0 * 2.0**2 + step**2
    np.testing.assert_allclose(proxy, [expected_derivative * 0.5])


def test_pv_inverse_floors_numerator_and_each_denominator_factor():
    endpoint = benchmark.ENDPOINTS["PV"][0]
    result = benchmark._physics(
        "PV",
        endpoint,
        {
            "PCE_percent": np.array([-1.0]),
            "Voc_V": np.array([-1.0]),
            "Jsc_mA_cm2": np.array([-1.0]),
            "FF_fraction": np.array([0.5]),
        },
        pd.DataFrame(index=[0]),
        direct_target=np.array([0.0]),
        target_range=(-1e20, 1e20),
    )
    np.testing.assert_allclose(result, [1e12])


def test_estm_clips_log_conductivity_inputs_before_exponentiation():
    endpoint = benchmark.ENDPOINTS["ESTM"][1]
    result = benchmark._physics(
        "ESTM",
        endpoint,
        {
            "seebeck_uV_per_K": np.array([1e6]),
            "log10_electrical_conductivity": np.array([40.0]),
            "log10_thermal_conductivity": np.array([-40.0]),
        },
        pd.DataFrame({"temperature_K": [1.0]}),
        direct_target=np.array([0.0]),
        target_range=(-np.inf, np.inf),
    )
    np.testing.assert_allclose(result, [1e60], rtol=1e-12)


def test_lmb_uses_analytic_first_order_propagation():
    endpoint = benchmark.ENDPOINTS["LMB"][0]
    result = benchmark._propagated_auxiliary_error_proxy(
        "LMB",
        endpoint,
        {"discharge_energy_wh": np.array([1.0, 1.0])},
        pd.DataFrame({"mass_kg": [0.5, 2.0]}),
        direct_target=np.zeros(2),
        target_range=(-np.inf, np.inf),
        auxiliary_rmse={"discharge_energy_wh": 1.0},
    )
    np.testing.assert_allclose(result, [2.0, 0.5])
