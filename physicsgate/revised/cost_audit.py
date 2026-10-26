"""Wall-clock and CPU cost of the method components.

Runs a pipeline module unchanged and records the time spent in each component
of every outer split: model selection and out-of-fold fitting of the Direct and
auxiliary base models, the physics evaluation, the residual branch, the
propagated-error descriptor R3, the gate, and the meta-learners of Ordinary
stacking and PhysResStack. Every outermost estimator ``fit`` and ``predict``
call inside a component is also recorded with its number of rows, which
separates the out-of-fold fits from the final fit and the test-set prediction.

Steps that recompute an earlier result (the PV refit of the selected base
models and the LMB refits of the retained base and residual models) are
labelled ``refit_*`` and do not belong to any method. CPU time covers the
process and its worker processes.

Records are written to ``<WORKDIR>/results/submitted/cost_audit/``.

    python -m physicsgate.revised.cost_audit <module> [module arguments]
"""

from __future__ import annotations

import atexit
import functools
import importlib
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import psutil

from physicsgate.submitted.paths import WORKDIR

PROCESS = psutil.Process()
COMPONENTS: list[dict] = []
CALLS: list[dict] = []
UNITS: list[dict] = []
STACK: list[dict] = []
DEPTH = 0
STAGE = ""
STARTED = time.perf_counter()


def _cpu() -> dict[int, float]:
    """User + system CPU seconds of the process and of each worker process."""
    times = {PROCESS.pid: sum(PROCESS.cpu_times()[:2])}
    for child in PROCESS.children(recursive=True):
        try:
            times[child.pid] = sum(child.cpu_times()[:2])
        except psutil.Error:
            pass
    return times


def _cpu_since(start: dict[int, float]) -> float:
    return float(sum(t - start.get(pid, 0.0) for pid, t in _cpu().items()))


def _context(caller: dict) -> dict:
    seed = caller.get("outer_seed", caller.get("seed"))
    route = caller.get("route_name")
    if route is None and STAGE.endswith("lmb.pipeline"):
        route = sys.modules["physicsgate.submitted.lmb.gate_tuned"].FORMULA_FAMILY.split("_")[0]
    return {"stage": STAGE, "seed": None if seed is None else int(seed), "fold": int(caller.get("fold_id", 0)),
            "unit_target": caller.get("target"), "route": route}


def _component(function, labeler):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if STACK:  # nested calls belong to the enclosing component
            return function(*args, **kwargs)
        caller = sys._getframe(1).f_locals
        label = labeler(args, kwargs, caller) if callable(labeler) else labeler
        if label is None:
            return function(*args, **kwargs)
        component, model_target = label if isinstance(label, tuple) else (label, None)
        frame = {"invocation": len(COMPONENTS) + len(STACK), **_context(caller), "component": component,
                 "model_target": model_target, "function": function.__name__}
        STACK.append(frame)
        cpu = _cpu()
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            wall = time.perf_counter() - start
            STACK.pop()
            COMPONENTS.append({**frame, "wall_s": wall, "cpu_s": _cpu_since(cpu)})
    return wrapper


def _unit(function):
    """Total time of one evaluation unit, including steps outside the labelled components."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            seed = kwargs.get("outer_seed", kwargs.get("seed", args[0] if args else None))
            UNITS.append({"stage": STAGE, "function": function.__name__, "seed": seed,
                          "fold": kwargs.get("fold_id", 0), "unit_target": kwargs.get("target"),
                          "wall_s": time.perf_counter() - start})
    return wrapper


def _estimator_method(method, kind: str):
    @functools.wraps(method)  # scikit-learn scorers check the method name
    def wrapper(self, *args, **kwargs):
        global DEPTH
        if DEPTH or not STACK:
            return method(self, *args, **kwargs)
        X = args[0] if args else kwargs.get("X")
        DEPTH = 1
        start = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            DEPTH = 0
            name = type(self).__name__
            CALLS.append({"invocation": STACK[-1]["invocation"],
                          "kind": "search" if kind == "fit" and name.endswith("SearchCV") else kind,
                          "estimator": name, "rows": int(np.shape(X)[0]) if X is not None else -1,
                          "wall_s": time.perf_counter() - start})
    return wrapper


def _patch_estimators() -> None:
    import sklearn.ensemble, sklearn.linear_model, sklearn.model_selection, sklearn.neighbors  # noqa: E401,F401
    import sklearn.pipeline, sklearn.svm, sklearn.tree  # noqa: E401,F401
    from sklearn.base import BaseEstimator
    try:
        import xgboost  # noqa: F401
    except ImportError:
        pass
    seen, pending = set(), [BaseEstimator]
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        pending.extend(cls.__subclasses__())
        for name in ("fit", "predict"):
            method = cls.__dict__.get(name)
            if callable(method) and type(method).__name__ == "function":
                setattr(cls, name, _estimator_method(method, name))


# -------- component labels


def _estm_base(args, kwargs, caller):
    if args[1] is caller.get("residual_features_train"):
        return "residual"
    return ("direct" if caller["column"] == caller["target"] else "auxiliary"), caller["column"]


def _by_role(**labels):
    def labeler(args, kwargs, caller):
        role = kwargs.get("role", "")
        return next((label for key, label in labels.items() if key in role), "diagnostic")
    return labeler


def _lmb_oof(args, kwargs, caller):
    return ("base_oof", caller["target"]) if "base_oof" in caller else None


R3 = (("global_auxiliary_sigmas", "R3"), ("propagated_sigma_finite_difference", "R3"))
RULES = {
    "physicsgate.revised.estm_pipeline": [
        (None, "evaluate_fold", "unit"),
        (None, "oof_and_test_predictions", _estm_base),
        ("physicsgate.submitted.estm.physics", "compute_estm_physics_prediction", "physics"),
        ("physicsgate.submitted.estm.physics", "stabilize_inverse_physics_branch", "physics"),
        *[(None, name, label) for name, label in R3],
        (None, "reliability_features", "gate"),
        (None, "tune_r2_r5_r6_gate", "gate"),
        ("physicsgate.submitted.common.meta_selection", "select_prediction_model", "ordinary"),
        (None, "select_meta_model", "physres"),
    ],
    "physicsgate.submitted.sse.pipeline": [
        (None, "run_seed_target", "unit"),
        (None, "fit_primary_direct_branch", lambda a, k, c: ("direct", k["target"])),
        (None, "fit_auxiliary_direct_branch", lambda a, k, c: ("auxiliary", k["aux_code"])),
        (None, "physics_prediction_for_target", "physics"),
        (None, "_fit_residual_prediction", "residual"),
        *[(None, name, label) for name, label in R3],
        (None, "reliability_features", "gate"),
        (None, "tune_r3_gate_alpha", "gate"),
        (None, "tune_fixed_gate_alpha", "gate"),
        (None, "tune_uniform_gate_alpha", "gate"),
        (None, "_fit_physres_prediction", "physres"),
    ],
    "physicsgate.submitted.sse.ordinary": [
        (None, "_ordinary_stacking_oof_and_test", "ordinary"),
    ],
    "physicsgate.submitted.lmb.pipeline": [
        ("physicsgate.submitted.lmb.gate_tuned", "evaluate_target", "unit"),
        (None, "_base_search", lambda a, k, c: ("base_selection", k["target"])),
        (None, "group_oof_and_test", _lmb_oof),
        (None, "group_select_prediction", _by_role(FixedGate_Ares_old_h="residual", PhysResStack="physres")),
        ("physicsgate.submitted.lmb.gate_vc", "vc_mass_physics_prediction", "physics"),
        (None, "de_physics_prediction", "physics"),
        ("physicsgate.submitted.lmb.gate_vc", "propagated_sigma_vc_mass", "R3"),
        (None, "de_propagated_sigma", "R3"),
        ("physicsgate.submitted.lmb.gate_vc", "reliability_features", "gate"),
        (None, "group_gate_tuner", "gate"),
    ],
    "physicsgate.submitted.lmb.complete": [
        ("physicsgate.submitted.lmb.five_methods", "ordinary_prediction", lambda a, k, c: ("ordinary", k["target"])),
        ("physicsgate.submitted.lmb.pipeline", "group_oof_and_test", "refit_retained"),
        ("physicsgate.submitted.lmb.pipeline", "group_select_prediction", _by_role(PhysResStack="physres")),
    ],
    "physicsgate.submitted.pv.selection": [
        (None, "run_seed", "unit"),
        ("physicsgate.submitted.pv.base_models", "tune_base_model", lambda a, k, c: ("base", k["target"])),
    ],
    "physicsgate.revised.pv_pipeline": [
        (None, "run_seed", "unit"),
        (None, "fit_pv_base_once", lambda a, k, c: ("refit_selected", k["target"])),
        ("physicsgate.submitted.pv.physics", "compute_pv_physics_prediction", "physics"),
        (None, "train_range_clip", "physics"),
        ("physicsgate.submitted.pv.meta", "select_meta",
         _by_role(_Ares_old_h="residual", _Ordinary_stacking="ordinary", _PhysResStack="physres")),
        *[(None, name, label) for name, label in R3],
        (None, "reliability_features", "gate"),
        ("physicsgate.submitted.pv.selection", "tune_gate", "gate"),
        (None, "fit_gate_model", "gate_replay"),
        (None, "tune_subset", "diagnostic"),
    ],
}


def _write() -> None:
    out = WORKDIR / "results" / "submitted" / "cost_audit"
    out.mkdir(parents=True, exist_ok=True)
    tag = STAGE.rsplit(".", 2)[-2] + "_" + STAGE.rsplit(".", 1)[-1]
    pd.DataFrame(COMPONENTS).to_csv(out / f"{tag}_components.csv", index=False)
    pd.DataFrame(CALLS).to_csv(out / f"{tag}_calls.csv", index=False)
    pd.DataFrame(UNITS).to_csv(out / f"{tag}_units.csv", index=False)
    memory = PROCESS.memory_info()
    import sklearn
    try:
        import xgboost
        xgb_version = xgboost.__version__
    except ImportError:
        xgb_version = ""
    pd.DataFrame([{
        "stage": STAGE, "wall_s": time.perf_counter() - STARTED, "peak_memory_MB": getattr(memory, "peak_wset", memory.rss) / 2**20,
        "logical_cpus": os.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False), "processor": platform.processor(),
        "loky_max_cpu_count": os.environ.get("LOKY_MAX_CPU_COUNT", ""),
        "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "xgboost": xgb_version,
        "argv": " ".join(sys.argv),
    }]).to_csv(out / f"{tag}_process.csv", index=False)


def main() -> None:
    global STAGE
    # every stage is timed with all logical cores; the SSE and ESTM modules otherwise limit
    # scikit-learn to one core on import, which the reported SSE results depend on
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
    STAGE = sys.argv[1]
    sys.argv = [STAGE, *sys.argv[2:]]
    module = importlib.import_module(STAGE)
    for owner, name, labeler in RULES[STAGE]:
        target = module if owner is None else importlib.import_module(owner)
        function = getattr(target, name)
        setattr(target, name, _unit(function) if labeler == "unit" else _component(function, labeler))
    _patch_estimators()
    atexit.register(_write)
    module.main()


if __name__ == "__main__":
    main()
