from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mlcast.data.normalization import (
    DEFAULT_SCALING,
    DENORMALIZATION_REGISTRY,
    NORMALIZATION_REGISTRY,
    ReflectivityScaling,
    normalized_to_rainfall_rate,
    rainfall_flux_to_reflectivity,
    rainfall_rate_to_normalized,
    rainfall_rate_to_reflectivity,
    reflectivity_to_rainfall_flux,
    reflectivity_to_rainfall_rate,
)


def test_rainfall_rate_to_normalized_and_inverse():
    """Verify rainfall_rate_to_normalized and its inverse are approximately symmetrical."""
    # Given an array of rainfall rates (mm/h)
    rainfall_rates = np.array([0.0, 0.5, 1.0, 5.0, 10.0, 50.0])

    # When normalizing to [-1, 1]
    normalized = rainfall_rate_to_normalized(rainfall_rates)

    # Then reversing the process
    denormalized = normalized_to_rainfall_rate(normalized)

    # The output should approximately match the input for valid rainfall rates
    # Note: Very low rainfall_rates are clipped in rainfall_rate_to_reflectivity, so we test > 0.1mm/h
    # 0.0 is clipped to 0 dBZ, which corresponds to ~0.036 mm/h.
    mask = rainfall_rates > 0.1
    np.testing.assert_allclose(rainfall_rates[mask], denormalized[mask], rtol=1e-5)

    # Check bounds
    assert np.all(normalized >= -1.0)
    assert np.all(normalized <= 1.0)


def test_normalization_registry():
    """Verify that the NORMALIZATION_REGISTRY maps CF standard names to Callables correctly."""
    assert "rainfall_rate" in NORMALIZATION_REGISTRY
    assert NORMALIZATION_REGISTRY["rainfall_rate"] == rainfall_rate_to_normalized
    assert "rainfall_flux" in NORMALIZATION_REGISTRY


def test_denormalization_registry():
    """Verify that the DENORMALIZATION_REGISTRY maps CF standard names to Callables correctly."""
    assert "rainfall_rate" in DENORMALIZATION_REGISTRY
    assert DENORMALIZATION_REGISTRY["rainfall_rate"] == normalized_to_rainfall_rate
    assert "rainfall_flux" in DENORMALIZATION_REGISTRY


def test_nan_passthrough():
    """NaN inputs must propagate untouched through both rate<->reflectivity conversions."""
    r = np.array([np.nan, 0.0, 1.0, 10.0, np.nan])
    dbz_out = rainfall_rate_to_reflectivity(r)
    assert np.isnan(dbz_out[0]) and np.isnan(dbz_out[-1])
    assert not np.isnan(dbz_out[1:-1]).any()

    d = np.array([np.nan, 0.0, 10.0, 30.0, np.nan])
    rr_out = reflectivity_to_rainfall_rate(d)
    assert np.isnan(rr_out[0]) and np.isnan(rr_out[-1])
    assert not np.isnan(rr_out[1:-1]).any()


def test_zero_point_round_trip():
    """0 mm/h <-> 0 dBZ must hold in both directions."""
    assert rainfall_rate_to_reflectivity(np.array([0.0]))[0] == 0.0
    assert reflectivity_to_rainfall_rate(np.array([0.0]))[0] == 0.0


def test_default_scaling_matches_module_shims():
    """The module-level shims must be bound to DEFAULT_SCALING."""
    assert DEFAULT_SCALING.a == 200.0
    assert DEFAULT_SCALING.b == 1.6
    assert DEFAULT_SCALING.dbz_floor == 0.0
    assert DEFAULT_SCALING.dbz_ceiling == 60.0
    assert DEFAULT_SCALING.norm_min == -1.0
    assert DEFAULT_SCALING.norm_max == 1.0
    assert rainfall_rate_to_reflectivity == DEFAULT_SCALING.rainfall_rate_to_reflectivity


def test_custom_scaling_zero_anchor():
    """A custom dbz_floor must anchor 0 mm/h <-> dbz_floor in both directions."""
    scaling = ReflectivityScaling(dbz_floor=-20.0)

    assert scaling.rainfall_rate_to_reflectivity(np.array([0.0]))[0] == -20.0
    assert scaling.reflectivity_to_rainfall_rate(np.array([-20.0]))[0] == 0.0

    rates = np.array([0.0, 0.1, 1.0, 10.0])
    back = scaling.reflectivity_to_rainfall_rate(scaling.rainfall_rate_to_reflectivity(rates))
    np.testing.assert_allclose(back[1:], rates[1:], rtol=1e-5)
    assert back[0] == 0.0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (200.0, 1.6),  # Marshall-Palmer default
        (300.0, 1.5),  # convective Z-R
        (100.0, 2.0),  # stratiform-ish
        (250.0, 1.2),
        (1.0, 1.0),  # degenerate but valid
    ],
)
def test_round_trip_across_ab(a, b):
    """Forward/inverse must be exact (modulo the zero anchor and clip) for any positive a, b."""
    scaling = ReflectivityScaling(a=a, b=b)

    # Zero anchor holds regardless of (a, b).
    assert scaling.rainfall_rate_to_reflectivity(np.array([0.0]))[0] == 0.0
    assert scaling.reflectivity_to_rainfall_rate(np.array([0.0]))[0] == 0.0

    # Round-trip from dBZ -> R -> dBZ at values strictly inside [floor, ceiling] is exact
    # for any positive (a, b), independent of where the resulting rainfall rates land.
    dbz_in = np.array([10.0, 25.0, 45.0, 55.0])
    rates = scaling.reflectivity_to_rainfall_rate(dbz_in)
    dbz_out = scaling.rainfall_rate_to_reflectivity(rates)
    np.testing.assert_allclose(dbz_out, dbz_in, rtol=1e-5)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_dtype_preserved(dtype):
    """Output dtype must match input dtype across all leaf and composite conversions."""
    scaling = ReflectivityScaling()
    r = np.array([0.0, 0.1, 1.0, 5.0, 20.0, 50.0, np.nan], dtype=dtype)

    dbz = scaling.rainfall_rate_to_reflectivity(r)
    back = scaling.reflectivity_to_rainfall_rate(dbz)

    assert dbz.dtype == dtype
    assert back.dtype == dtype
    assert back[0] == 0.0  # zero anchor
    assert np.isnan(back[-1])  # NaN propagated

    # Composite R -> normalized -> R path (the dataloader hot path).
    norm = scaling.rainfall_rate_to_normalized(r)
    back_via_norm = scaling.normalized_to_rainfall_rate(norm)
    assert norm.dtype == dtype
    assert back_via_norm.dtype == dtype

    # Round-trip accuracy: looser tolerance for lower precision dtypes.
    rtol = {np.float16: 1e-2, np.float32: 1e-5, np.float64: 1e-12}[dtype]
    np.testing.assert_allclose(back[1:-1], r[1:-1], rtol=rtol)
    np.testing.assert_allclose(back_via_norm[1:-1], r[1:-1], rtol=rtol)


def test_normalize_range_with_custom_bounds():
    """Normalization must map [dbz_floor, dbz_ceiling] to [-1, 1] regardless of the bounds."""
    scaling = ReflectivityScaling(dbz_floor=-20.0, dbz_ceiling=70.0)
    dbz = np.array([-20.0, 25.0, 70.0])
    norm = scaling.normalize_reflectivity(dbz)
    np.testing.assert_allclose(norm, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(scaling.denormalize_reflectivity(norm), dbz)


@pytest.mark.parametrize(
    ("norm_min", "norm_max"),
    [
        (-1.0, 1.0),  # default
        (0.0, 1.0),  # common [0, 1] target
        (-0.5, 0.5),
        (1.0, -1.0),  # inverted range
    ],
)
def test_normalize_custom_range(norm_min, norm_max):
    """Normalization must map [dbz_floor, dbz_ceiling] -> [norm_min, norm_max] and round-trip."""
    scaling = ReflectivityScaling(norm_min=norm_min, norm_max=norm_max)
    dbz = np.array([0.0, 30.0, 60.0])
    norm = scaling.normalize_reflectivity(dbz)
    np.testing.assert_allclose(norm, [norm_min, (norm_min + norm_max) / 2, norm_max])
    np.testing.assert_allclose(scaling.denormalize_reflectivity(norm), dbz)


def test_frozen_dataclass():
    """ReflectivityScaling instances must be immutable."""
    scaling = ReflectivityScaling()
    with pytest.raises(FrozenInstanceError):
        scaling.a = 300.0  # type: ignore[misc]


def test_rainfall_flux_warnings():
    """Verify that the rainfall_flux functions raise the expected UserWarning."""
    arr = np.array([1.0, 5.0])

    with pytest.warns(UserWarning, match="Assuming we can use the same function"):
        rainfall_flux_to_reflectivity(arr)

    with pytest.warns(UserWarning, match="Assuming we can use the same function"):
        reflectivity_to_rainfall_flux(arr)
