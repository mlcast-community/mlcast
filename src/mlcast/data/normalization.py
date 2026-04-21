"""Unit conversion utilities for radar precipitation data.

Provides functions to convert between rain rate (mm/h), reflectivity (dBZ),
and a normalized [-1, 1] representation used internally by the models.
"""

import numpy as np


def rainrate_to_reflectivity(rainrate: np.ndarray) -> np.ndarray:
    """Convert rain rate to reflectivity using the Marshall-Palmer relationship.

    Applies Z = 200 * R^1.6 and converts to dBZ. Values below ~0.037 mm/h
    are clipped to 0 dBZ; values above 60 dBZ are clipped to 60.

    Parameters
    ----------
    rainrate : np.ndarray
        Rain rate in mm/h. Can be any shape.

    Returns
    -------
    reflectivity : np.ndarray
        Reflectivity in dBZ, clipped to [0, 60]. Same shape as input.
    """
    epsilon = 1e-16
    return (10 * np.log10(200 * rainrate**1.6 + epsilon)).clip(0, 60)


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


def reflectivity_to_rainrate(reflectivity: np.ndarray) -> np.ndarray:
    """Convert reflectivity back to rain rate using the inverse Marshall-Palmer relationship.

    Applies R = (Z_linear / 200)^(1/1.6) where Z_linear = 10^(dBZ/10).

    Parameters
    ----------
    reflectivity : np.ndarray
        Reflectivity in dBZ. Can be any shape.

    Returns
    -------
    rainrate : np.ndarray
        Rain rate in mm/h. Same shape as input.
    """
    z_linear = 10 ** (reflectivity / 10.0)
    return (z_linear / 200.0) ** (1.0 / 1.6)


def rainrate_to_normalized(rainrate: np.ndarray) -> np.ndarray:
    """Convert rain rate directly to normalized reflectivity.

    Composes :func:`rainrate_to_reflectivity` and
    :func:`normalize_reflectivity`.

    Parameters
    ----------
    rainrate : np.ndarray
        Rain rate in mm/h. Can be any shape.

    Returns
    -------
    normalized : np.ndarray
        Normalized reflectivity in [-1, 1]. Same shape as input.
    """
    reflectivity = rainrate_to_reflectivity(rainrate)
    return normalize_reflectivity(reflectivity)


def normalized_to_rainrate(normalized: np.ndarray) -> np.ndarray:
    """Convert normalized reflectivity back to rain rate.

    Composes :func:`denormalize_reflectivity` and
    :func:`reflectivity_to_rainrate`.

    Parameters
    ----------
    normalized : np.ndarray
        Normalized reflectivity in [-1, 1]. Can be any shape.

    Returns
    -------
    rainrate : np.ndarray
        Rain rate in mm/h. Same shape as input.
    """
    reflectivity = denormalize_reflectivity(normalized)
    return reflectivity_to_rainrate(reflectivity)


NORMALIZATION_REGISTRY = {
    "rainfall_rate": rainrate_to_normalized,
}
