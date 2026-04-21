import numpy as np

from mlcast.data.normalization import (
    normalized_to_rainrate,
    rainrate_to_normalized,
)


def test_rainrate_to_normalized_and_inverse():
    """Verify rainrate_to_normalized and its inverse are approximately symmetrical."""
    # Given an array of rain rates (mm/h)
    rainrates = np.array([0.0, 0.5, 1.0, 5.0, 10.0, 50.0])

    # When normalizing to [-1, 1]
    normalized = rainrate_to_normalized(rainrates)

    # Then reversing the process
    denormalized = normalized_to_rainrate(normalized)

    # The output should approximately match the input for valid rain rates
    # Note: Very low rainrates are clipped in rainrate_to_reflectivity, so we test > 0.1mm/h
    # 0.0 is clipped to 0 dBZ, which corresponds to ~0.036 mm/h.
    mask = rainrates > 0.1
    np.testing.assert_allclose(rainrates[mask], denormalized[mask], rtol=1e-5)

    # Check bounds
    assert np.all(normalized >= -1.0)
    assert np.all(normalized <= 1.0)
