"""Detect whether a zarr data variable is rain rate or radar reflectivity.

Different MLCast source datasets use different units — e.g. IT-DPC stores
rainfall flux in `kg m-2 h-1` (equivalent to mm/h), DMI stores radar
reflectivity in `dBZ`. Sampling statistics that depend on a "wet pixel"
threshold (e.g. `frac_wet`) need a threshold in the same units as the
data, so this module inspects CF-convention attributes to classify the
variable and returns a sensible default threshold.

The classification order is:
1. `standard_name` (most authoritative, CF convention).
2. `units` (normalized to lowercase, whitespace-stripped).
3. Fallback: raise, unless the caller provides an override.
"""

from __future__ import annotations

from collections.abc import Mapping

RAIN_RATE_STANDARD_NAMES = {
    "rainfall_flux",
    "rainfall_rate",
    "precipitation_flux",
    "lwe_precipitation_rate",
}
REFLECTIVITY_STANDARD_NAMES = {
    "equivalent_reflectivity_factor",
    "radar_reflectivity",
}

RAIN_RATE_UNITS = {
    "mm/h",
    "mmh-1",
    "mmh1",
    "mm/hr",
    "mmhr-1",
    "mmh^-1",
    "kgm-2h-1",
    "kgm2h1",
}
REFLECTIVITY_UNITS = {"dbz", "db"}

# Defaults chosen to match conventional "meaningful rain" cutoffs:
#   - 0.1 mm/h is the standard drizzle floor
#   - 7 dBZ corresponds to R ≈ 0.08 mm/h via Marshall-Palmer, matching it
DEFAULT_WET_THRESHOLD = {
    "rainrate": 0.1,
    "reflectivity": 7.0,
}


def _normalize_units(units: str | None) -> str:
    if not units:
        return ""
    return units.strip().lower().replace(" ", "").replace("**", "^")


def detect_data_kind(attrs: Mapping[str, object]) -> str:
    """Classify a variable as 'rainrate' or 'reflectivity' from its attrs.

    Parameters
    ----------
    attrs
        The attribute dict of a zarr/xarray variable.

    Returns
    -------
    'rainrate' or 'reflectivity'

    Raises
    ------
    ValueError
        If the attributes don't match any known rain-rate or reflectivity
        indicator. In that case the caller should fall back to an explicit
        CLI override.
    """
    std = str(attrs.get("standard_name", "") or "").strip().lower()
    if std in RAIN_RATE_STANDARD_NAMES:
        return "rainrate"
    if std in REFLECTIVITY_STANDARD_NAMES:
        return "reflectivity"

    units = _normalize_units(str(attrs.get("units", "") or ""))
    if units in RAIN_RATE_UNITS:
        return "rainrate"
    if units in REFLECTIVITY_UNITS:
        return "reflectivity"

    raise ValueError(
        f"Cannot auto-detect data kind: standard_name={std!r}, units={units!r}. Pass --data-kind explicitly."
    )


def default_wet_threshold(data_kind: str) -> float:
    """Conventional wet-pixel threshold for a given data kind."""
    try:
        return DEFAULT_WET_THRESHOLD[data_kind]
    except KeyError:
        raise ValueError(f"Unknown data_kind {data_kind!r}") from None
