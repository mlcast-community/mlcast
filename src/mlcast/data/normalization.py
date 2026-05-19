"""Unit conversion utilities for radar precipitation data.

Provides :class:`ReflectivityScaling`, a configurable container for the
Marshall-Palmer Z-R relationship and the dBZ normalization range, plus a
module-level default instance and backwards-compatible shim functions and
registries.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_SECONDS_PER_HOUR = 3600.0
_5MIN_ACCUMULATION_HOURS = 5 * 60 / _SECONDS_PER_HOUR  # = 1/12


@dataclass(frozen=True)
class ReflectivityScaling:
    """Configurable Marshall-Palmer Z-R conversion and dBZ normalization range.

    All forward/inverse conversions between rainfall rate (mm/h), rainfall flux
    (kg m-2 s-1), 5-minute rainfall amount (kg m-2), reflectivity (dBZ), and the
    normalized representation used by the model share a single set of
    constants stored on the instance.

    The Marshall-Palmer relationship is Z = a * R^b. Reflectivity output is
    clipped to ``[dbz_floor, dbz_ceiling]``; the floor anchors the zero point in
    both directions — R = 0 ↔ dbz_floor — so any value at or below ``dbz_floor``
    on the inverse direction maps back to exactly 0 mm/h, and R = 0 forward maps
    to exactly ``dbz_floor``. The ceiling caps the forward conversion but is not
    enforced on the inverse direction. The normalized representation is a linear
    map of ``[dbz_floor, dbz_ceiling]`` onto ``[norm_min, norm_max]``.

    Parameters
    ----------
    a : float, default=200.0
        Marshall-Palmer Z-R coefficient.
    b : float, default=1.6
        Marshall-Palmer Z-R exponent.
    dbz_floor : float, default=0.0
        Lower bound of the valid dBZ range and zero-rainfall anchor.
    dbz_ceiling : float, default=60.0
        Upper bound of the valid dBZ range.
    norm_min : float, default=-1.0
        Lower end of the normalized range; corresponds to ``dbz_floor``.
    norm_max : float, default=1.0
        Upper end of the normalized range; corresponds to ``dbz_ceiling``.
    """

    a: float = 200.0
    b: float = 1.6
    dbz_floor: float = 0.0
    dbz_ceiling: float = 60.0
    norm_min: float = -1.0
    norm_max: float = 1.0

    def _typed_constants(self, dtype: np.dtype) -> tuple[np.generic, np.generic, np.generic, np.generic]:
        """Return ``(a, b, dbz_floor, dbz_ceiling)`` cast to ``dtype``.

        For non-floating dtypes (e.g. int), falls back to ``float64`` so the
        Z-R math has somewhere meaningful to live. Used by the Z-R conversion
        methods to keep the per-call arithmetic in the caller's precision and
        avoid the implicit fp64 upcast from mixing the dataclass's Python-float
        fields with the input array.
        """
        dt = dtype.type if np.issubdtype(dtype, np.floating) else np.float64
        return dt(self.a), dt(self.b), dt(self.dbz_floor), dt(self.dbz_ceiling)

    def rainfall_rate_to_reflectivity(self, rainfall_rate: np.ndarray) -> np.ndarray:
        """Convert rainfall rate (mm/h) to reflectivity (dBZ).

        Applies ``dBZ = 10*log10(a) + 10*b*log10(R)`` and clips to
        ``[dbz_floor, dbz_ceiling]``. NaN inputs propagate untouched. The output
        dtype matches the input dtype for floating-point inputs.
        """
        a, b, floor, ceiling = self._typed_constants(rainfall_rate.dtype)
        with np.errstate(divide="ignore", invalid="ignore"):
            dbz = 10 * np.log10(a) + 10 * b * np.log10(rainfall_rate)
        return np.clip(dbz, floor, ceiling)

    def reflectivity_to_rainfall_rate(self, reflectivity: np.ndarray) -> np.ndarray:
        """Convert reflectivity (dBZ) back to rainfall rate (mm/h).

        Applies ``R = 10^((dBZ - 10*log10(a)) / (10*b))``. Values at or below
        ``dbz_floor`` are set to exactly 0 mm/h, so ``dbz_floor`` ↔ 0 mm/h holds
        in both directions. NaN inputs propagate untouched. The output dtype
        matches the input dtype for floating-point inputs.
        """
        a, b, floor, _ = self._typed_constants(reflectivity.dtype)
        rr = np.power(10, (reflectivity - 10 * np.log10(a)) / (10 * b))
        rr[reflectivity <= floor] = 0
        return rr

    def rainfall_flux_to_reflectivity(self, rainfall_flux: np.ndarray) -> np.ndarray:
        """Convert rainfall flux (kg m-2 s-1) to reflectivity.

        Assumes the same Z-R scaling as rainfall rate and emits a ``UserWarning``.
        """
        warnings.warn(
            "Assuming we can use the same function (rainfall_rate_to_reflectivity) "
            "for scaling rainfall_flux to reflectivity.",
            UserWarning,
            stacklevel=2,
        )
        return self.rainfall_rate_to_reflectivity(rainfall_flux)

    def reflectivity_to_rainfall_flux(self, reflectivity: np.ndarray) -> np.ndarray:
        """Convert reflectivity back to rainfall flux (kg m-2 s-1).

        Assumes the same Z-R scaling as rainfall rate and emits a ``UserWarning``.
        """
        warnings.warn(
            "Assuming we can use the same function (reflectivity_to_rainfall_rate) "
            "for scaling reflectivity to rainfall_flux.",
            UserWarning,
            stacklevel=2,
        )
        return self.reflectivity_to_rainfall_rate(reflectivity)

    def normalize_reflectivity(self, reflectivity: np.ndarray) -> np.ndarray:
        """Normalize reflectivity from [dbz_floor, dbz_ceiling] dBZ to [norm_min, norm_max]."""
        dbz_span = self.dbz_ceiling - self.dbz_floor
        norm_span = self.norm_max - self.norm_min
        return self.norm_min + (reflectivity - self.dbz_floor) * (norm_span / dbz_span)

    def denormalize_reflectivity(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize from [norm_min, norm_max] back to [dbz_floor, dbz_ceiling] dBZ."""
        dbz_span = self.dbz_ceiling - self.dbz_floor
        norm_span = self.norm_max - self.norm_min
        return self.dbz_floor + (normalized - self.norm_min) * (dbz_span / norm_span)

    def rainfall_rate_to_normalized(self, rainfall_rate: np.ndarray) -> np.ndarray:
        """Convert rainfall rate (mm/h) directly to normalized reflectivity in [-1, 1]."""
        return self.normalize_reflectivity(self.rainfall_rate_to_reflectivity(rainfall_rate))

    def normalized_to_rainfall_rate(self, normalized: np.ndarray) -> np.ndarray:
        """Convert normalized reflectivity back to rainfall rate (mm/h)."""
        return self.reflectivity_to_rainfall_rate(self.denormalize_reflectivity(normalized))

    def rainfall_flux_to_normalized(self, rainfall_flux: np.ndarray) -> np.ndarray:
        """Convert rainfall flux (kg m-2 s-1) directly to normalized reflectivity in [-1, 1]."""
        return self.normalize_reflectivity(self.rainfall_flux_to_reflectivity(rainfall_flux))

    def normalized_to_rainfall_flux(self, normalized: np.ndarray) -> np.ndarray:
        """Convert normalized reflectivity back to rainfall flux (kg m-2 s-1)."""
        return self.reflectivity_to_rainfall_flux(self.denormalize_reflectivity(normalized))

    def rainfall_amount_5min_to_normalized(self, rainfall_amount: np.ndarray) -> np.ndarray:
        """Convert 5-minute accumulated rainfall (kg m-2 = mm) to normalized reflectivity.

        Converts to an equivalent rainfall rate by dividing by the 5-minute
        accumulation period (1/12 h), then applies the standard Z-R pipeline.
        """
        rainfall_rate = rainfall_amount / _5MIN_ACCUMULATION_HOURS
        return self.rainfall_rate_to_normalized(rainfall_rate)

    def normalized_to_rainfall_amount_5min(self, normalized: np.ndarray) -> np.ndarray:
        """Convert normalized reflectivity back to 5-minute accumulated rainfall (kg m-2 = mm)."""
        return self.normalized_to_rainfall_rate(normalized) * _5MIN_ACCUMULATION_HOURS

    @property
    def normalization_registry(self) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
        """Map CF standard names to the unnormalized → normalized conversion methods."""
        return {
            "rainfall_rate": self.rainfall_rate_to_normalized,
            "rainfall_flux": self.rainfall_flux_to_normalized,
            "rainfall_amount": self.rainfall_amount_5min_to_normalized,
        }

    @property
    def denormalization_registry(self) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
        """Map CF standard names to the normalized → unnormalized conversion methods."""
        return {
            "rainfall_rate": self.normalized_to_rainfall_rate,
            "rainfall_flux": self.normalized_to_rainfall_flux,
            "rainfall_amount": self.normalized_to_rainfall_amount_5min,
        }


DEFAULT_SCALING = ReflectivityScaling()

rainfall_rate_to_reflectivity = DEFAULT_SCALING.rainfall_rate_to_reflectivity
reflectivity_to_rainfall_rate = DEFAULT_SCALING.reflectivity_to_rainfall_rate
rainfall_flux_to_reflectivity = DEFAULT_SCALING.rainfall_flux_to_reflectivity
reflectivity_to_rainfall_flux = DEFAULT_SCALING.reflectivity_to_rainfall_flux
normalize_reflectivity = DEFAULT_SCALING.normalize_reflectivity
denormalize_reflectivity = DEFAULT_SCALING.denormalize_reflectivity
rainfall_rate_to_normalized = DEFAULT_SCALING.rainfall_rate_to_normalized
normalized_to_rainfall_rate = DEFAULT_SCALING.normalized_to_rainfall_rate
rainfall_flux_to_normalized = DEFAULT_SCALING.rainfall_flux_to_normalized
normalized_to_rainfall_flux = DEFAULT_SCALING.normalized_to_rainfall_flux
rainfall_amount_5min_to_normalized = DEFAULT_SCALING.rainfall_amount_5min_to_normalized
normalized_to_rainfall_amount_5min = DEFAULT_SCALING.normalized_to_rainfall_amount_5min

NORMALIZATION_REGISTRY = DEFAULT_SCALING.normalization_registry
DENORMALIZATION_REGISTRY = DEFAULT_SCALING.denormalization_registry
