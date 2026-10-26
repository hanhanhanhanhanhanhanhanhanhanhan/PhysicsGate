"""Residual branch with other inputs or another target.

Ablation of the residual branch R = P + h. Runs a pipeline module of the
revised protocol with the model h changed. The reported inputs of h are:

- SSE: descriptors, temperature, physics and auxiliary predictions;
- ESTM: descriptors, Direct, physics and auxiliary predictions, temperature;
- LMB: Direct, physics and auxiliary predictions, cell mass;
- PV: Direct, physics and auxiliary predictions.

``--inputs`` selects the variant:

- ``minimal`` (default): h on the Physics-only prediction and the measurement
  condition (temperature for SSE and ESTM, cell mass for LMB, none for PV);
- ``no_direct``: the reported inputs without the Direct prediction (the SSE
  residual has no Direct input, so it applies to ESTM, LMB and PV);
- ``direct_target``: the reported inputs, with h fitted to y instead of y - P,
  so that the branch is h itself rather than P + h;
- ``no_physics``: as ``direct_target`` without the Physics-only prediction
  among the inputs.

Model selection and everything else are unchanged, so the residual branch and
the methods built on it are recomputed. For the last two variants the selection
error is that of h against y, which equals the error of P + (h - P).

    python -m physicsgate.revised.residual_ablation [--inputs VARIANT] <module> [module arguments]
"""

from __future__ import annotations

import runpy
import sys

import numpy as np

from physicsgate.submitted.estm import base_models as estm_base
from physicsgate.submitted.lmb import gate_tuned
from physicsgate.submitted.pv import selection as pv_selection
from physicsgate.submitted.sse import target_adaptive

VARIANTS = ("minimal", "no_direct", "direct_target", "no_physics")
INPUTS = "minimal"


def _without(matrix, column) -> np.ndarray:
    """The feature matrix without the one column equal to ``column``."""
    matrix = np.asarray(matrix, dtype=float)
    matches = [j for j in range(matrix.shape[1]) if np.array_equal(matrix[:, j], np.asarray(column, dtype=float))]
    if len(matches) != 1:
        raise RuntimeError(f"Input found in {len(matches)} residual columns")
    return np.delete(matrix, matches[0], axis=1)


def _inputs(X_train, X_test, caller, *, direct, physics, minimal):
    """Residual inputs of the selected variant; ``direct``, ``physics`` and ``minimal`` name caller locals."""
    if INPUTS == "minimal":
        return minimal(caller)
    if INPUTS == "no_direct":
        return _without(X_train, caller[f"{direct}_oof"]), _without(X_test, caller[f"{direct}_test"])
    if INPUTS == "no_physics":
        return _without(X_train, caller[f"{physics}_oof"]), _without(X_test, caller[f"{physics}_test"])
    return X_train, X_test


def _to_y() -> bool:
    return INPUTS in ("direct_target", "no_physics")


# ESTM: the residual features are passed to oof_and_test_predictions
_estm_oof_and_test = estm_base.oof_and_test_predictions


def _estm(*args, **kwargs):
    args = list(args)
    caller = sys._getframe(1).f_locals
    if len(args) > 3 and args[1] is caller.get("residual_features_train"):
        args[1], args[3] = _inputs(
            args[1], args[3], caller, direct="direct", physics="phys",
            minimal=lambda c: (np.column_stack([c["phys_oof"], c["t_train"]]),
                               np.column_stack([c["phys_test"], c["t_test"]])))
        if _to_y():
            args[2] = caller["y_train"]
            oof, test = _estm_oof_and_test(*args, **kwargs)
            return oof - caller["phys_oof"], test - caller["phys_test"]
    return _estm_oof_and_test(*args, **kwargs)


# SSE: residual features are built by target_adaptive._residual_features
_sse_fit_residual = target_adaptive._fit_residual_prediction
_sse_features = target_adaptive._residual_features


def _sse_minimal_features(X_base, T, *, target, physics_prediction, predictions):
    return np.column_stack([np.asarray(physics_prediction, dtype=float), np.asarray(T, dtype=float)])


def _sse_no_physics_features(X_base, T, *, target, physics_prediction, predictions):
    full = _sse_features(X_base, T, target=target, physics_prediction=physics_prediction, predictions=predictions)
    return _without(full, physics_prediction)


def _sse(**kwargs):
    names = list(kwargs["feature_names"])
    if INPUTS == "minimal":
        target_adaptive._residual_features, names = _sse_minimal_features, ["physics_prediction", "T_K"]
    elif INPUTS == "no_physics":
        target_adaptive._residual_features = _sse_no_physics_features
        names = [n for n in names if not n.endswith("_physics_prediction")]
    if _to_y():  # the residual target inside is y_train - physics_oof
        kwargs["y_train"] = kwargs["y_train"] + kwargs["physics_oof"]
    try:
        oof, test, *rest = _sse_fit_residual(**{**kwargs, "feature_names": names})
    finally:
        target_adaptive._residual_features = _sse_features
    if _to_y():  # P + h is returned; the branch is h
        oof, test = oof - kwargs["physics_oof"], test - kwargs["physics_test"]
    return (oof, test, *rest)


# LMB: the residual model is selected by base.select_prediction with role "<target>::FixedGate_Ares_old_h"
_lmb_evaluate = gate_tuned.evaluate_target


def _lmb(*args, **kwargs):
    select = gate_tuned.base.select_prediction

    def changed(X_train, y_train, X_test, *rest, role="", **options):
        if not role.endswith("FixedGate_Ares_old_h"):
            return select(X_train, y_train, X_test, *rest, role=role, **options)
        caller = sys._getframe(1).f_locals
        X_train, X_test = _inputs(
            X_train, X_test, caller, direct="direct", physics="physics",
            minimal=lambda c: (np.column_stack([c["physics_oof"], c["mass_train"]]),
                               np.column_stack([c["physics_test"], c["mass_test"]])))
        if not _to_y():
            return select(X_train, y_train, X_test, *rest, role=role, **options)
        oof, test, *other = select(X_train, caller["y_train"], X_test, *rest, role=role, **options)
        return (np.asarray(oof, dtype=float) - caller["physics_oof"],
                np.asarray(test, dtype=float) - caller["physics_test"], *other)

    gate_tuned.base.select_prediction = changed
    try:
        return _lmb_evaluate(*args, **kwargs)
    finally:
        gate_tuned.base.select_prediction = select


# PV: the residual model is selected by select_meta with role "<target>_Ares_old_h_..."
_pv_select_meta = pv_selection.paper.select_meta


def _pv(X_train, y_train, X_test, *rest, role="", **options):
    if "_Ares_old_h" not in role:
        return _pv_select_meta(X_train, y_train, X_test, *rest, role=role, **options)
    caller = sys._getframe(1).f_locals
    X_train, X_test = _inputs(
        X_train, X_test, caller, direct="direct", physics="phys",
        minimal=lambda c: (np.asarray(c["phys_oof"], dtype=float).reshape(-1, 1),
                           np.asarray(c["phys_test"], dtype=float).reshape(-1, 1)))
    if not _to_y():
        return _pv_select_meta(X_train, y_train, X_test, *rest, role=role, **options)
    options["y_test"] = caller["y_test"]
    oof, test, *other = _pv_select_meta(X_train, caller["y_train"], X_test, *rest, role=role, **options)
    return (np.asarray(oof, dtype=float) - caller["phys_oof"],
            np.asarray(test, dtype=float) - caller["phys_test"], *other)


def main() -> None:
    global INPUTS
    argv = sys.argv[1:]
    if argv[0] == "--inputs":
        INPUTS, argv = argv[1], argv[2:]
    if INPUTS not in VARIANTS:
        raise SystemExit(f"--inputs must be one of {VARIANTS}")
    estm_base.oof_and_test_predictions = _estm
    if INPUTS != "no_direct":
        target_adaptive._fit_residual_prediction = _sse
    gate_tuned.evaluate_target = _lmb
    pv_selection.paper.select_meta = _pv
    module = argv[0]
    sys.argv = [module, *argv[1:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
