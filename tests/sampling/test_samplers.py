"""Unit tests for the candidate-selection schemes and the sampler registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlcast.sampling import ImportanceSampler, UniformSampler, get_sampler
from mlcast.sampling.samplers import SAMPLER_REGISTRY


def _dry_heavy_pool(n_dry: int = 900, n_wet: int = 100) -> pd.DataFrame:
    """Mostly-dry pool with a wet tail (mean 0.01 vs 20.0)."""
    mean = np.concatenate([np.full(n_dry, 0.01), np.full(n_wet, 20.0)])
    return pd.DataFrame({"t": np.arange(len(mean)), "x": 0, "y": 0, "mean": mean})


def test_importance_selection_reshapes_toward_extremes_and_is_reproducible() -> None:
    pool = _dry_heavy_pool()
    sampler = ImportanceSampler()
    kept = sampler.select(pool, np.random.default_rng(0))

    # a subset, each row at most once
    assert kept.ndim == 1 and len(kept) <= len(pool)
    assert len(np.unique(kept)) == len(kept)
    # wet rows are only 10% of the pool but dominate the kept set
    wet_frac_pool = (pool["mean"].to_numpy() > 1.0).mean()
    wet_frac_kept = (pool["mean"].to_numpy()[kept] > 1.0).mean()
    assert wet_frac_pool == pytest.approx(0.1)
    assert wet_frac_kept > 0.5
    # reproducible given the rng
    again = sampler.select(pool, np.random.default_rng(0))
    assert np.array_equal(kept, again)


def test_importance_tuning_changes_how_much_is_kept() -> None:
    pool = _dry_heavy_pool()
    # a higher floor (q_min) keeps more of the dry majority
    low = ImportanceSampler(q_min=1e-4).select(pool, np.random.default_rng(1))
    high = ImportanceSampler(q_min=0.05).select(pool, np.random.default_rng(1))
    assert len(high) > len(low)


def test_importance_sampler_requires_mean_column() -> None:
    pool = pd.DataFrame({"t": [0, 1], "x": [0, 0], "y": [0, 0]})
    with pytest.raises(ValueError, match="mean"):
        ImportanceSampler().select(pool, np.random.default_rng(0))


def test_importance_sampler_can_weight_on_a_different_column() -> None:
    # 'mean' is flat (would keep ~all); we instead weight on 'sum', whose high
    # tail should dominate the kept set
    pool = pd.DataFrame(
        {
            "t": np.arange(1000),
            "x": 0,
            "y": 0,
            "mean": 0.0,
            "sum": np.concatenate([np.full(900, 1.0), np.full(100, 1000.0)]),
        }
    )
    kept = ImportanceSampler(column="sum", scale=1000.0).select(pool, np.random.default_rng(0))
    assert (pool["sum"].to_numpy()[kept] > 100).mean() > 0.5


def test_importance_sampler_missing_chosen_column_raises() -> None:
    pool = pd.DataFrame({"t": [0, 1], "x": 0, "y": 0, "mean": [0.1, 0.2]})
    with pytest.raises(ValueError, match="frac_wet"):
        ImportanceSampler(column="frac_wet").select(pool, np.random.default_rng(0))


def test_uniform_sampler_keep_fraction() -> None:
    pool = pd.DataFrame({"t": np.arange(1000), "x": 0, "y": 0})  # no mean column needed
    assert len(UniformSampler(keep_fraction=1.0).select(pool, np.random.default_rng(0))) == 1000
    half = UniformSampler(keep_fraction=0.5).select(pool, np.random.default_rng(0))
    assert 400 < len(half) < 600


def test_uniform_sampler_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="keep_fraction"):
        UniformSampler(keep_fraction=1.5)


def test_registry_lookup_and_unknown() -> None:
    assert {"importance", "uniform"} <= set(SAMPLER_REGISTRY)
    sampler = get_sampler("importance", scale=2.0)
    assert isinstance(sampler, ImportanceSampler) and sampler.scale == 2.0
    assert isinstance(get_sampler("uniform"), UniformSampler)
    with pytest.raises(ValueError, match="Unknown sampler"):
        get_sampler("does-not-exist")
