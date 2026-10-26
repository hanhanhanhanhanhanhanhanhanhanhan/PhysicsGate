"""SSE residual model with the Direct prediction added (Supplementary Table S16).

The submitted SSE code gives the residual model h the descriptors, the
temperature and the Physics-only and auxiliary predictions, but not the Direct
prediction of the target, which the residual models of the other datasets
receive. ``install`` adds it after the descriptors and the temperature, for the
comparison in Table S16; the reported SSE results use the submitted inputs.
"""

from __future__ import annotations

import numpy as np

from physicsgate.submitted.sse import target_adaptive as ta

SUBMITTED_FEATURES = ta._residual_features
SUBMITTED_NAMES = ta.residual_feature_names_for_target


def residual_features(X_base, T, *, target, physics_prediction, predictions):
    auxiliaries = [ta._as_1d(predictions[name], name=f"{name}_prediction") for name in ta.TARGET_ORDER if name != target]
    return np.column_stack([ta._direct_features(X_base, T), ta._as_1d(predictions[target], name=f"{target}_prediction"),
                            ta._as_1d(physics_prediction, name="physics_prediction"), *auxiliaries])


def residual_feature_names(target, base_feature_names):
    names = SUBMITTED_NAMES(target, base_feature_names)
    position = len(base_feature_names) + 1  # after the descriptors and T_K
    return [*names[:position], f"{target}_direct_prediction", *names[position:]]


def install() -> None:
    """Add the Direct prediction unless another wrapper has already replaced the inputs."""
    if ta._residual_features is SUBMITTED_FEATURES:
        ta._residual_features = residual_features
    if ta.residual_feature_names_for_target is SUBMITTED_NAMES:
        ta.residual_feature_names_for_target = residual_feature_names
