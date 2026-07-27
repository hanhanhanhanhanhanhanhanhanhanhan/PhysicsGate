"""Equation configuration contracts and physics-prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

ARRHENIUS_KB_EV_PER_K = 8.617333262e-5


@dataclass(frozen=True)
class EquationConfig:
    """Base metadata for a scientific equation linking coupled targets."""

    name: str
    target: str
    auxiliary_targets: tuple[str, ...]


@dataclass(frozen=True)
class LinearEquationConfig(EquationConfig):
    """Configuration for equations such as ``A = a * B + b * C + c``."""

    coefficients: Mapping[str, float]
    intercept: float = 0.0


@dataclass(frozen=True)
class ArrheniusEquationConfig(EquationConfig):
    """Configuration for ``ln_sigma = ln_sigma0 - Ea / (kB * T)``.

    ``Ea`` is assumed to be in eV and ``T`` in kelvin. The legacy
    ``gas_constant`` field is kept only for compatibility with earlier API
    contract tests; Arrhenius conductivity helpers use ``kB``.
    """

    temperature_name: str
    activation_energy_name: str
    pre_exponential_name: str = "ln_sigma0"
    kB: float = ARRHENIUS_KB_EV_PER_K
    gas_constant: float = 8.31446261815324


@dataclass(frozen=True)
class ProductRatioEquationConfig(EquationConfig):
    """Configuration placeholder for product-over-ratio equations."""

    numerator_targets: tuple[str, ...]
    denominator_target: str
    scale: float = 1.0


def validate_equation_config(config: EquationConfig) -> None:
    """Validate equation metadata without evaluating the equation."""
    if not config.name:
        raise ValueError("EquationConfig.name must be non-empty")
    if not config.target:
        raise ValueError("EquationConfig.target must be non-empty")
    if not config.auxiliary_targets:
        raise ValueError("EquationConfig.auxiliary_targets must be non-empty")
    if config.target in config.auxiliary_targets:
        raise ValueError("target must not also be an auxiliary target")
    if isinstance(config, ArrheniusEquationConfig):
        if not config.temperature_name:
            raise ValueError("ArrheniusEquationConfig.temperature_name must be non-empty")
        if not config.activation_energy_name:
            raise ValueError(
                "ArrheniusEquationConfig.activation_energy_name must be non-empty"
            )
        if not config.pre_exponential_name:
            raise ValueError(
                "ArrheniusEquationConfig.pre_exponential_name must be non-empty"
            )
        if config.kB <= 0:
            raise ValueError("ArrheniusEquationConfig.kB must be positive")


def _as_1d_float(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def arrhenius_ln_sigma(
    ln_sigma0: ArrayLike,
    Ea: ArrayLike,
    T: ArrayLike,
    *,
    kB: float = ARRHENIUS_KB_EV_PER_K,
) -> NDArray[np.float64]:
    """Compute Arrhenius log-conductivity from predicted auxiliaries.

    Parameters use scientific units expected by the benchmark: ``Ea`` in eV,
    ``T`` in kelvin, and ``kB`` in eV/K.
    """
    if kB <= 0:
        raise ValueError("kB must be positive")
    prefactor = _as_1d_float(ln_sigma0, name="ln_sigma0")
    activation = _as_1d_float(Ea, name="Ea")
    temperature = _as_1d_float(T, name="T")
    if not (prefactor.shape == activation.shape == temperature.shape):
        raise ValueError("ln_sigma0, Ea, and T must have matching shapes")
    if np.any(temperature <= 0):
        raise ValueError("Temperature values must be positive kelvin")
    return prefactor - activation / (kB * temperature)


def arrhenius_residual(
    ln_sigma_pred: ArrayLike,
    ln_sigma0_pred: ArrayLike,
    Ea_pred: ArrayLike,
    T: ArrayLike,
    *,
    kB: float = ARRHENIUS_KB_EV_PER_K,
) -> NDArray[np.float64]:
    """Return signed residuals against the Arrhenius equation."""
    prediction = _as_1d_float(ln_sigma_pred, name="ln_sigma_pred")
    physics = arrhenius_ln_sigma(ln_sigma0_pred, Ea_pred, T, kB=kB)
    if prediction.shape != physics.shape:
        raise ValueError("ln_sigma_pred and Arrhenius prediction must match")
    return prediction - physics


def _extract_arrhenius_temperature(
    config: ArrheniusEquationConfig,
    *,
    auxiliary_predictions: Mapping[str, ArrayLike],
    features: ArrayLike | Mapping[str, ArrayLike] | None,
) -> ArrayLike:
    if features is not None:
        if isinstance(features, Mapping):
            if config.temperature_name not in features:
                raise ValueError(
                    f"Missing temperature feature: {config.temperature_name}"
                )
            return features[config.temperature_name]
        return features
    if config.temperature_name in auxiliary_predictions:
        return auxiliary_predictions[config.temperature_name]
    raise ValueError(
        "Arrhenius physics prediction requires temperature via features "
        f"or auxiliary_predictions['{config.temperature_name}']"
    )


def compute_physics_prediction(
    config: EquationConfig,
    *,
    auxiliary_predictions: Mapping[str, ArrayLike],
    features: ArrayLike | Mapping[str, ArrayLike] | None = None,
) -> NDArray[np.float64]:
    """Compute equation-derived predictions from predicted auxiliaries only."""
    validate_equation_config(config)
    if isinstance(config, ArrheniusEquationConfig):
        required = {config.pre_exponential_name, config.activation_energy_name}
        missing = required - set(auxiliary_predictions)
        if missing:
            raise ValueError(f"Missing auxiliary predictions: {sorted(missing)}")
        temperature = _extract_arrhenius_temperature(
            config,
            auxiliary_predictions=auxiliary_predictions,
            features=features,
        )
        return arrhenius_ln_sigma(
            auxiliary_predictions[config.pre_exponential_name],
            auxiliary_predictions[config.activation_energy_name],
            temperature,
            kB=config.kB,
        )

    if not isinstance(config, LinearEquationConfig):
        raise NotImplementedError("Only LinearEquationConfig is implemented in Milestone 1")

    missing = set(config.auxiliary_targets) - set(auxiliary_predictions)
    if missing:
        raise ValueError(f"Missing auxiliary predictions: {sorted(missing)}")
    coefficient_names = set(config.coefficients)
    if coefficient_names != set(config.auxiliary_targets):
        raise ValueError("LinearEquationConfig coefficients must match auxiliary_targets")

    arrays = {
        name: np.asarray(auxiliary_predictions[name], dtype=float).reshape(-1)
        for name in config.auxiliary_targets
    }
    lengths = {array.shape[0] for array in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Auxiliary prediction arrays must have equal lengths")

    result = np.full(next(iter(lengths)), config.intercept, dtype=float)
    for name, coefficient in config.coefficients.items():
        result += coefficient * arrays[name]
    return result


def compute_physics_residual(
    y_pred: ArrayLike,
    physics_pred: ArrayLike,
) -> NDArray[np.float64]:
    """Return signed residuals against equation-derived predictions."""
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    physics = np.asarray(physics_pred, dtype=float).reshape(-1)
    if prediction.shape != physics.shape:
        raise ValueError("y_pred and physics_pred must have matching shapes")
    return prediction - physics


__all__ = [
    "ARRHENIUS_KB_EV_PER_K",
    "ArrheniusEquationConfig",
    "EquationConfig",
    "LinearEquationConfig",
    "ProductRatioEquationConfig",
    "arrhenius_ln_sigma",
    "arrhenius_residual",
    "compute_physics_prediction",
    "compute_physics_residual",
    "validate_equation_config",
]
