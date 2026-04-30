import numpy as np
import pytest

from mlcast.data.normalization import (
    DENORMALIZATION_REGISTRY,
    NORMALIZATION_REGISTRY,
    normalized_to_rainfall_rate,
    rainfall_flux_to_reflectivity,
    rainfall_rate_to_normalized,
    reflectivity_to_rainfall_flux,
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


def test_rainfall_flux_warnings():
    """Verify that the rainfall_flux functions raise the expected UserWarning."""
    arr = np.array([1.0, 5.0])

    with pytest.warns(UserWarning, match="Assuming we can use the same function"):
        rainfall_flux_to_reflectivity(arr)

    with pytest.warns(UserWarning, match="Assuming we can use the same function"):
        reflectivity_to_rainfall_flux(arr)
