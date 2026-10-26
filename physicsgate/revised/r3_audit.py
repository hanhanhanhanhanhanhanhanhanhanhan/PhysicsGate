"""Per-sample audit of the propagated auxiliary-error descriptor R3.

Runs a pipeline module unchanged and records, for the test rows of every
endpoint and outer split, the observed Physics-only error, R3 and two Monte
Carlo propagations of auxiliary errors through the same physics function:

- ``sd_mc_independent``: independent Gaussian auxiliary errors with the scales
  used by R3 (tests the first-order approximation);
- ``sd_mc_residual``: auxiliary error vectors resampled from the training-side
  out-of-fold residuals (adds correlation between auxiliary errors and
  non-Gaussian error distributions).

The Monte Carlo draws use their own random generator; the pipeline results are
unchanged. Records are written to ``<WORKDIR>/results/submitted/r3_audit/``.

    python -m physicsgate.revised.r3_audit <module> [module arguments]
"""

from __future__ import annotations

import atexit
import os
import runpy
import sys
import zlib

import numpy as np
import pandas as pd

from physicsgate.submitted.common import propagated_error
from physicsgate.submitted.lmb import gate_tuned
from physicsgate.submitted.paths import WORKDIR

DRAWS = 300
RECORDS: list[pd.DataFrame] = []
_TRAIN_RESIDUALS: dict[str, np.ndarray] = {}
TEST_ARRAYS = ("aux_test_arr", "aux_test_matrix", "aux_pred_test")  # ESTM, PV, SSE
RAW_PHYSICS = ("phys_test_raw", "raw_phys_test", "physics_test")


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


def _monte_carlo(evaluate, aux: np.ndarray, sigmas: np.ndarray, residuals: np.ndarray, rng) -> tuple:
    """Standard deviation of the physics output under independent and resampled auxiliary errors."""
    independent = np.empty((aux.shape[0], DRAWS))
    resampled = np.empty((aux.shape[0], DRAWS))
    finite = residuals[np.all(np.isfinite(residuals), axis=1)]
    for k in range(DRAWS):
        with np.errstate(all="ignore"):
            independent[:, k] = evaluate(aux + rng.normal(0.0, 1.0, aux.shape) * np.nan_to_num(sigmas))
            resampled[:, k] = evaluate(aux + finite[rng.integers(0, len(finite), aux.shape[0])])
    return np.nanstd(independent, axis=1), np.nanstd(resampled, axis=1)


def _record(dataset, target, seed, fold, y, physics, physics_raw, r3, sd_ind, sd_res) -> None:
    RECORDS.append(pd.DataFrame({
        "dataset": dataset, "target": str(target), "seed": int(seed), "fold": int(fold),
        "y_true": np.asarray(y, dtype=float), "physics": np.asarray(physics, dtype=float),
        "physics_raw": np.asarray(physics_raw, dtype=float), "R3": np.asarray(r3, dtype=float),
        "sd_mc_independent": sd_ind, "sd_mc_residual": sd_res,
    }))


def _write() -> None:
    if not RECORDS:
        return
    out = WORKDIR / "results" / "submitted" / "r3_audit"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{os.environ.get('R3_AUDIT_TAG', 'audit')}_{os.getpid()}.csv.gz"
    pd.concat(RECORDS, ignore_index=True).to_csv(out / name, index=False)


# -------- finite-difference route (ESTM, PV, SSE)
_original_sigmas = propagated_error.global_auxiliary_sigmas
_original_fd = propagated_error.propagated_sigma_finite_difference


def _sigmas(*, aux_pred_train, aux_true_train, aux_names):
    _TRAIN_RESIDUALS["last"] = np.asarray(aux_true_train, dtype=float) - np.asarray(aux_pred_train, dtype=float)
    return _original_sigmas(aux_pred_train=aux_pred_train, aux_true_train=aux_true_train, aux_names=aux_names)


def _finite_difference(*, aux_pred, known, aux_sigmas, phi):
    result = _original_fd(aux_pred=aux_pred, known=known, aux_sigmas=aux_sigmas, phi=phi)
    caller = sys._getframe(1).f_locals
    if any(caller.get(name) is aux_pred for name in TEST_ARRAYS):
        target = caller["target"]
        seed = caller.get("outer_seed", caller.get("seed"))
        fold = caller.get("fold_id", 0)
        aux = np.asarray(aux_pred, dtype=float)
        raw = next(caller[n] for n in RAW_PHYSICS if n in caller)
        sd_ind, sd_res = _monte_carlo(lambda a: phi(a, known), aux, np.asarray(aux_sigmas, dtype=float),
                                      _TRAIN_RESIDUALS["last"], _rng(target, seed, fold))
        _record(DATASET, target, seed, fold, caller["y_test"], phi(aux, known), raw, result[0], sd_ind, sd_res)
    return result


# -------- analytic route (LMB)
_original_evaluate = gate_tuned.evaluate_target


def _evaluate_target(*args, **kwargs):
    route_sigma = gate_tuned.propagated_sigma

    def sigma(target, predictions, mass_kg, sigmas):
        out = route_sigma(target, predictions, mass_kg, sigmas)
        caller = sys._getframe(1).f_locals
        if mass_kg is caller.get("mass_test"):
            names = list(predictions)
            aux = np.column_stack([np.asarray(predictions[n], dtype=float) for n in names])
            train = caller["train"]
            residuals = np.column_stack([train[n].to_numpy(dtype=float) - np.asarray(caller["aux_oof"][n], dtype=float)
                                         for n in names])
            physics = gate_tuned.physics_prediction
            evaluate = lambda a: physics(target, {n: a[:, i] for i, n in enumerate(names)}, mass_kg)  # noqa: E731
            scales = np.array([sigmas.get(n, 0.0) for n in names])
            route = gate_tuned.FORMULA_FAMILY.split("_")[0]
            sd_ind, sd_res = _monte_carlo(evaluate, aux, scales, residuals, _rng(route, target, caller["seed"]))
            _record(f"LMB-{route}", target, caller["seed"], 0, caller["y_test"], caller["physics_test"],
                    caller["physics_test"], out, sd_ind, sd_res)
        return out

    gate_tuned.propagated_sigma = sigma
    try:
        return _original_evaluate(*args, **kwargs)
    finally:
        gate_tuned.propagated_sigma = route_sigma


DATASET = ""


def main() -> None:
    global DATASET
    module = sys.argv[1]
    DATASET = next((d for d in ("estm", "pv", "sse", "lmb") if f".{d}" in module or f"{d}_" in module), "").upper()
    propagated_error.global_auxiliary_sigmas = _sigmas
    propagated_error.propagated_sigma_finite_difference = _finite_difference
    gate_tuned.evaluate_target = _evaluate_target
    atexit.register(_write)
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
