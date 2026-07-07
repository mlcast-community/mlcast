"""Pluggable candidate-selection schemes for the precomputed-sampling dataset.

A :class:`Sampler` filters the candidate pool (the rows of a stats parquet) to
a training subset once, at dataset init. Add a scheme by subclassing
:class:`Sampler`, decorating it with ``@register_sampler("name")``, and
implementing :meth:`Sampler.select`; it is then available via
:func:`get_sampler` (e.g. from a config).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Sampler(ABC):
    """Selects a subset of candidate rows via a per-row keep/discard decision."""

    @abstractmethod
    def select(self, coords: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """Return the positions of the kept rows (each selected at most once)."""


SAMPLER_REGISTRY: dict[str, type[Sampler]] = {}


def register_sampler(name: str):
    """Class decorator registering a :class:`Sampler` subclass under ``name``."""

    def decorator(cls: type[Sampler]) -> type[Sampler]:
        SAMPLER_REGISTRY[name] = cls
        cls.sampler_name = name
        return cls

    return decorator


@register_sampler("uniform")
class UniformSampler(Sampler):
    """Keep each candidate with a fixed probability, independent of its stats.

    Parameters
    ----------
    keep_fraction : float
        Per-row keep probability in ``[0, 1]``. ``1.0`` (default) keeps the
        whole pool; smaller values take a random uniform subsample.
    """

    def __init__(self, keep_fraction: float = 1.0) -> None:
        if not 0.0 <= keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in [0, 1], got {keep_fraction}")
        self.keep_fraction = keep_fraction

    def select(self, coords: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        if self.keep_fraction >= 1.0:
            return np.arange(len(coords))
        return np.flatnonzero(rng.random(len(coords)) < self.keep_fraction)


@register_sampler("importance")
class ImportanceSampler(Sampler):
    """Keep each candidate with probability ``w / w.max()``, where the weight
    ``w = q_min + mean_weight * (1 - exp(-s / scale))`` rises with a per-row
    statistic ``s`` (the ``column``). High-statistic datacubes are kept
    preferentially and common ones thinned out, without duplication. Needs the
    chosen ``column`` (a legacy CSV index has none).

    Parameters
    ----------
    column : str
        The stats-parquet column to weight on, e.g. ``"mean"`` (default),
        ``"sum"``, or ``"frac_wet"``.
    q_min : float
        Floor weight on every candidate, keeping some low-statistic windows.
    scale : float
        Saturation scale of ``1 - exp(-s / scale)``; set it on the order of the
        column's typical magnitude (``mean``/``frac_wet`` ~ O(1); ``sum`` large).
    mean_weight : float
        Weight given to the statistic; relative to ``q_min`` it sets how hard
        low-statistic windows are thinned versus high ones.
    """

    def __init__(self, column: str = "mean", q_min: float = 1e-4, scale: float = 1.0, mean_weight: float = 0.1) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if q_min < 0 or mean_weight < 0:
            raise ValueError(f"q_min and mean_weight must be non-negative, got {q_min}, {mean_weight}")
        self.column = column
        self.q_min = q_min
        self.scale = scale
        self.mean_weight = mean_weight

    def select(self, coords: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        if self.column not in coords.columns:
            raise ValueError(
                f"ImportanceSampler needs the {self.column!r} statistic column, absent from this "
                f"index (columns: {list(coords.columns)}); a legacy CSV index carries only "
                f"(t, x, y), so use it without a sampler (sampler=None)."
            )
        # floor weight + a saturating response to the statistic; NaNs floored to q_min
        stat = np.nan_to_num(coords[self.column].to_numpy(dtype=float), nan=0.0)
        weights = self.q_min + self.mean_weight * (1.0 - np.exp(-stat / self.scale))
        w_max = weights.max(initial=0.0)
        probs = weights / w_max if w_max > 0 else np.zeros_like(weights)
        return np.flatnonzero(rng.random(len(coords)) < probs)


def get_sampler(name: str, **kwargs) -> Sampler:
    """Construct a registered sampler by name, e.g. ``get_sampler("importance", scale=2.0)``."""
    try:
        cls = SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown sampler {name!r}; available: {sorted(SAMPLER_REGISTRY)}") from None
    return cls(**kwargs)
