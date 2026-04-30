"""Unit conversion utilities for radar precipitation data.

Provides functions to convert between rainfall rate (mm/h), rainfall flux (kg m-2 s-1),
reflectivity (dBZ), and a normalized [-1, 1] representation used internally by the models.
"""

import warnings

import numpy as np


def rainfall_rate_to_reflectivity(rainfall_rate: np.ndarray) -> np.ndarray:
    """Convert rainfall rate to reflectivity using the Marshall-Palmer relationship.

    Applies Z = 200 * R^1.6 and converts to dBZ. Values below ~0.037 mm/h
    are clipped to 0 dBZ; values above 60 dBZ are clipped to 60.

    Parameters
    ----------
    rainfall_rate : np.ndarray
        Rainfall rate in mm/h. Can be any shape.

    Returns
    -------
    reflectivity : np.ndarray
        Reflectivity in dBZ, clipped to [0, 60]. Same shape as input.
    """
    epsilon = 1e-16
    return (10 * np.log10(200 * rainfall_rate**1.6 + epsilon)).clip(0, 60)


def rainfall_flux_to_reflectivity(rainfall_flux: np.ndarray) -> np.ndarray:
    """Convert rainfall flux to reflectivity.

    Wraps :func:`rainfall_rate_to_reflectivity` and emits a warning that it
    assumes the same scaling applies.

    Parameters
    ----------
    rainfall_flux : np.ndarray
        Rainfall flux in kg m-2 s-1. Can be any shape.

    Returns
    -------
    reflectivity : np.ndarray
        Reflectivity in dBZ, clipped to [0, 60]. Same shape as input.
    """
    warnings.warn(
        "Assuming we can use the same function (rainfall_rate_to_reflectivity) "
        "for scaling rainfall_flux to reflectivity.",
        UserWarning,
        stacklevel=2,
    )
    return rainfall_rate_to_reflectivity(rainfall_flux)


def normalize_reflectivity(reflectivity: np.ndarray) -> np.ndarray:
    """Normalize reflectivity from [0, 60] dBZ to [-1, 1].

    Parameters
    ----------
    reflectivity : np.ndarray
        Reflectivity in dBZ, expected in [0, 60]. Can be any shape.

    Returns
    -------
    normalized : np.ndarray
        Normalized reflectivity in [-1, 1]. Same shape as input.
    """
    return (reflectivity / 30.0) - 1.0


def denormalize_reflectivity(normalized: np.ndarray) -> np.ndarray:
    """Denormalize from [-1, 1] back to [0, 60] dBZ reflectivity.

    Parameters
    ----------
    normalized : np.ndarray
        Normalized reflectivity in [-1, 1]. Can be any shape.

    Returns
    -------
    reflectivity : np.ndarray
        Reflectivity in dBZ, in [0, 60]. Same shape as input.
    """
    return (normalized + 1.0) * 30.0


def reflectivity_to_rainfall_rate(reflectivity: np.ndarray) -> np.ndarray:
    """Convert reflectivity back to rainfall rate using the inverse Marshall-Palmer relationship.

    Applies R = (Z_linear / 200)^(1/1.6) where Z_linear = 10^(dBZ/10).

    Parameters
    ----------
    reflectivity : np.ndarray
        Reflectivity in dBZ. Can be any shape.

    Returns
    -------
    rainfall_rate : np.ndarray
        Rainfall rate in mm/h. Same shape as input.
    """
    z_linear = 10 ** (reflectivity / 10.0)
    return (z_linear / 200.0) ** (1.0 / 1.6)


def reflectivity_to_rainfall_flux(reflectivity: np.ndarray) -> np.ndarray:
    """Convert reflectivity back to rainfall flux.

    Wraps :func:`reflectivity_to_rainfall_rate` and emits a warning that it
    assumes the same scaling applies.

    Parameters
    ----------
    reflectivity : np.ndarray
        Reflectivity in dBZ. Can be any shape.

    Returns
    -------
    rainfall_flux : np.ndarray
        Rainfall flux in kg m-2 s-1. Same shape as input.
    """
    warnings.warn(
        "Assuming we can use the same function (reflectivity_to_rainfall_rate) "
        "for scaling reflectivity to rainfall_flux.",
        UserWarning,
        stacklevel=2,
    )
    return reflectivity_to_rainfall_rate(reflectivity)


def rainfall_rate_to_normalized(rainfall_rate: np.ndarray) -> np.ndarray:
    """Convert rainfall rate directly to normalized reflectivity.

    Composes :func:`rainfall_rate_to_reflectivity` and
    :func:`normalize_reflectivity`.

    Parameters
    ----------
    rainfall_rate : np.ndarray
        Rainfall rate in mm/h. Can be any shape.

    Returns
    -------
    normalized : np.ndarray
        Normalized reflectivity in [-1, 1]. Same shape as input.
    """
    reflectivity = rainfall_rate_to_reflectivity(rainfall_rate)
    return normalize_reflectivity(reflectivity)


def rainfall_flux_to_normalized(rainfall_flux: np.ndarray) -> np.ndarray:
    """Convert rainfall flux directly to normalized reflectivity.

    Composes :func:`rainfall_flux_to_reflectivity` and
    :func:`normalize_reflectivity`.

    Parameters
    ----------
    rainfall_flux : np.ndarray
        Rainfall flux in kg m-2 s-1. Can be any shape.

    Returns
    -------
    normalized : np.ndarray
        Normalized reflectivity in [-1, 1]. Same shape as input.
    """
    reflectivity = rainfall_flux_to_reflectivity(rainfall_flux)
    return normalize_reflectivity(reflectivity)


def normalized_to_rainfall_rate(normalized: np.ndarray) -> np.ndarray:
    """Convert normalized reflectivity back to rainfall rate.

    Composes :func:`denormalize_reflectivity` and
    :func:`reflectivity_to_rainfall_rate`.

    Parameters
    ----------
    normalized : np.ndarray
        Normalized reflectivity in [-1, 1]. Can be any shape.

    Returns
    -------
    rainfall_rate : np.ndarray
        Rainfall rate in mm/h. Same shape as input.
    """
    reflectivity = denormalize_reflectivity(normalized)
    return reflectivity_to_rainfall_rate(reflectivity)


def normalized_to_rainfall_flux(normalized: np.ndarray) -> np.ndarray:
    """Convert normalized reflectivity back to rainfall flux.

    Composes :func:`denormalize_reflectivity` and
    :func:`reflectivity_to_rainfall_flux`.

    Parameters
    ----------
    normalized : np.ndarray
        Normalized reflectivity in [-1, 1]. Can be any shape.

    Returns
    -------
    rainfall_flux : np.ndarray
        Rainfall flux in kg m-2 s-1. Same shape as input.
    """
    reflectivity = denormalize_reflectivity(normalized)
    return reflectivity_to_rainfall_flux(reflectivity)


_SECONDS_PER_HOUR = 3600.0
_5MIN_ACCUMULATION_HOURS = 5 * 60 / _SECONDS_PER_HOUR  # = 1/12


def rainfall_amount_5min_to_normalized(rainfall_amount: np.ndarray) -> np.ndarray:
    """Convert 5-minute rainfall amount (kg m-2 = mm) to normalized reflectivity.

    Converts to an equivalent rainfall rate (mm/h) by dividing by the 5-minute
    accumulation period (1/12 h), then applies the Marshall-Palmer Z-R relationship
    and normalizes to [-1, 1].

    Parameters
    ----------
    rainfall_amount : np.ndarray
        5-minute accumulated rainfall in kg m-2 (= mm). Can be any shape.

    Returns
    -------
    normalized : np.ndarray
        Normalized reflectivity in [-1, 1]. Same shape as input.
    """
    rainfall_rate = rainfall_amount / _5MIN_ACCUMULATION_HOURS
    return rainfall_rate_to_normalized(rainfall_rate)


def normalized_to_rainfall_amount_5min(normalized: np.ndarray) -> np.ndarray:
    """Convert normalized reflectivity back to 5-minute rainfall amount (kg m-2).

    Inverts :func:`rainfall_amount_5min_to_normalized`: denormalizes to reflectivity,
    applies the inverse Marshall-Palmer relationship to get rainfall rate (mm/h),
    then multiplies by the 5-minute accumulation period (1/12 h).

    Parameters
    ----------
    normalized : np.ndarray
        Normalized reflectivity in [-1, 1]. Can be any shape.

    Returns
    -------
    rainfall_amount : np.ndarray
        5-minute accumulated rainfall in kg m-2 (= mm). Same shape as input.
    """
    rainfall_rate = normalized_to_rainfall_rate(normalized)
    return rainfall_rate * _5MIN_ACCUMULATION_HOURS


NORMALIZATION_REGISTRY = {
    "rainfall_rate": rainfall_rate_to_normalized,
    "rainfall_flux": rainfall_flux_to_normalized,
    "rainfall_amount": rainfall_amount_5min_to_normalized,
}

DENORMALIZATION_REGISTRY = {
    "rainfall_rate": normalized_to_rainfall_rate,
    "rainfall_flux": normalized_to_rainfall_flux,
    "rainfall_amount": normalized_to_rainfall_amount_5min,
}
