"""Direct with a bagged ensemble of the selected model, at the training time of PhysicsGate.

Runs a pipeline module of the revised protocol unchanged and, wherever the
final Direct model of an endpoint is fitted, also fits ``m`` copies of the same
selected model on bootstrap resamples of the training rows and averages their
test predictions. The model selection of Direct is not repeated, so the bagged
ensemble costs the selection plus ``m`` final fits; ``m`` is chosen per dataset
so that this matches the training time of PhysicsGate measured by
``physicsgate.revised.cost_audit``.

The test predictions of the pipeline's Direct model and of the running average
of the bagged ensemble at ``m`` = 1, 2, 5, 10, ... are written with the test
targets to ``<WORKDIR>/results/submitted/bagged_direct/``, one file per
endpoint and outer training set as soon as it is done, so that an interrupted
run keeps the finished units.

    python -m physicsgate.revised.bagged_direct --m M <module> [module arguments]
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
from sklearn.base import clone

from physicsgate.submitted.estm import base_models as estm_base
from physicsgate.submitted.lmb import pipeline as lmb_pipeline
from physicsgate.submitted.paths import WORKDIR
from physicsgate.submitted.pv import stabilization as pv_stabilization
from physicsgate.submitted.sse import data as sse_data

CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
M = 1
DATASET = ""


def _rows(X, index):
    return X.iloc[index] if hasattr(X, "iloc") else np.asarray(X)[index]


def _bag(make_model, X_train, y_train, X_test, key: tuple) -> tuple[dict[int, np.ndarray], float]:
    """Running averages of ``M`` bootstrap fits of the model at the checkpoints, and the time taken."""
    rng = np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))
    y_train = np.asarray(y_train, dtype=float)
    n = len(y_train)
    total = np.zeros(np.shape(X_test)[0])
    averages = {}
    start = time.perf_counter()
    for k in range(1, M + 1):
        index = rng.integers(0, n, n)
        model = make_model()
        try:
            model.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
        except (AttributeError, ValueError):
            pass
        model.fit(_rows(X_train, index), y_train[index])
        total += np.asarray(model.predict(X_test), dtype=float).reshape(-1)
        if k in CHECKPOINTS or k == M:
            averages[k] = total / k
    return averages, time.perf_counter() - start


def _path(seed, fold, target):
    return WORKDIR / "results" / "submitted" / "bagged_direct" / f"{DATASET}_{int(seed)}_{int(fold)}_{target}.csv.gz"


def _record(seed, fold, target, y_test, plain, averages, seconds) -> None:
    frame = pd.DataFrame({"dataset": DATASET, "seed": int(seed), "fold": int(fold), "target": str(target),
                          "y_true": np.asarray(y_test, dtype=float), "direct": np.asarray(plain, dtype=float)})
    for k, prediction in averages.items():
        frame[f"bagged_{k}"] = prediction
    frame["m"], frame["bagging_seconds"] = M, seconds
    path = _path(seed, fold, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# ESTM: every base model of an endpoint is fitted by oof_and_test_predictions; Direct is column == target
_estm_oof_and_test = estm_base.oof_and_test_predictions


def _estm(*args, **kwargs):
    oof, test = _estm_oof_and_test(*args, **kwargs)
    caller = sys._getframe(1).f_locals
    if caller.get("column") == caller.get("target") and args[1] is caller.get("X_train"):
        key = ("ESTM", caller["outer_seed"], caller["fold_id"], caller["target"])
        averages, seconds = _bag(lambda: clone(args[0]), args[1], args[2], args[3], key)
        _record(caller["outer_seed"], caller["fold_id"], caller["target"], caller["y_test"], test, averages, seconds)
    return oof, test


# SSE: the final Direct fit is _fit_and_predict_test called from fit_primary_direct_branch
_sse_fit_and_predict = sse_data._fit_and_predict_test


def _sse(estimator, X_train, y_train, X_test):
    test = _sse_fit_and_predict(estimator, X_train, y_train, X_test)
    if sys._getframe(1).f_code.co_name == "fit_primary_direct_branch":
        unit = sys._getframe(2).f_locals  # run_seed_target
        key = ("SSE", unit["seed"], unit["target"])
        averages, seconds = _bag(lambda: clone(estimator), X_train, y_train, X_test, key)
        _record(unit["seed"], 0, unit["target"], unit["y_test"], test, averages, seconds)
    return test


# LMB: base models are fitted once per split by group_oof_and_test in main
_lmb_oof_and_test = lmb_pipeline.group_oof_and_test


def _lmb(model, X_train, y_train, X_test, groups, n_splits):
    oof, test = _lmb_oof_and_test(model, X_train, y_train, X_test, groups, n_splits)
    caller = sys._getframe(1).f_locals
    if "base_oof" in caller and "model" in caller:
        key = ("LMB", caller["seed"], caller["target"])
        averages, seconds = _bag(lambda: clone(model), X_train, y_train, X_test, key)
        y_test = caller["test"][caller["target"]].to_numpy(dtype=float)
        _record(caller["seed"], 0, caller["target"], y_test, test, averages, seconds)
    return oof, test


# PV: the selected base model of each target is refitted by fit_pv_base_once
_pv_fit_once = pv_stabilization.fit_pv_base_once


def _pv(**kwargs):
    oof, test = _pv_fit_once(**kwargs)
    seed = sys._getframe(1).f_locals["seed"]
    if _path(seed, 0, kwargs["target"]).exists():  # finished before an interruption
        return oof, test
    pvgated = pv_stabilization.pv_exp.pvgated
    candidates = pvgated.base_candidate_factories(kwargs["target"], random_state=kwargs["random_state"],
                                                  n_jobs=kwargs["n_jobs"], rf_n_estimators=None)
    factory = next(f for name, f, _ in candidates if name == kwargs["selected_name"])
    data, train_idx, test_idx = kwargs["data"], kwargs["train_idx"], kwargs["test_idx"]
    X_raw, y = data[pv_stabilization.PV_FEATURE_COLUMNS], data[kwargs["target"]].astype(float)
    X_train, X_test, _, _ = pvgated.encode_features_for_target(X_raw.iloc[train_idx], X_raw.iloc[test_idx],
                                                               y.iloc[train_idx], protocol="leakage_safe")
    averages, seconds = _bag(factory, X_train, y.iloc[train_idx].to_numpy(), X_test, ("PV", seed, kwargs["target"]))
    _record(seed, 0, kwargs["target"], y.iloc[test_idx].to_numpy(), test, averages, seconds)
    return oof, test


def main() -> None:
    global M, DATASET
    argv = sys.argv[1:]
    if argv[0] != "--m":
        raise SystemExit("usage: python -m physicsgate.revised.bagged_direct --m M <module> [module arguments]")
    M, argv = int(argv[1]), argv[2:]
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())  # all cores, as in the timings from which m is derived
    name = argv[0]
    DATASET = next(d for d in ("estm", "sse", "lmb", "pv") if f".{d}" in name or f"{d}_" in name).upper()
    # patched before the pipeline module is imported, which binds them by name
    estm_base.oof_and_test_predictions = _estm
    sse_data._fit_and_predict_test = _sse
    pv_stabilization.fit_pv_base_once = _pv
    module = importlib.import_module(name)
    if module is lmb_pipeline:
        module.group_oof_and_test = _lmb
    sys.argv = [name, *argv[1:]]
    module.main()


if __name__ == "__main__":
    main()
