"""Pluggable candidate selectors for the sampling-index dataset.

A :class:`CandidateSelector` is a keep/discard rule over the candidate pool
(the rows of a sampling index). It is applied **once, at dataset init**, and
the kept subset is reused for the whole training: candidates are not re-drawn
every epoch, so the dataset length is fixed and the val/test sets are stable.

Add a scheme by subclassing :class:`CandidateSelector`, decorating it with
``@register_selector("name")``, and implementing
:meth:`CandidateSelector.select`; it is then available via
:func:`get_selector` (e.g. from a config).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class CandidateSelector(ABC):
    """Selects a subset of candidate rows via a per-row keep/discard decision."""

    @abstractmethod
    def select(self, candidates: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """Return the positions of the kept rows (each selected at most once)."""


SELECTOR_REGISTRY: dict[str, type[CandidateSelector]] = {}


def register_selector(name: str):
    """Class decorator registering a :class:`CandidateSelector` subclass under ``name``."""

    def decorator(cls: type[CandidateSelector]) -> type[CandidateSelector]:
        SELECTOR_REGISTRY[name] = cls
        cls.selector_name = name
        return cls

    return decorator


@register_selector("uniform")
class UniformSelector(CandidateSelector):
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

    def select(self, candidates: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        if self.keep_fraction >= 1.0:
            return np.arange(len(candidates))
        return np.flatnonzero(rng.random(len(candidates)) < self.keep_fraction)


@register_selector("importance")
class ImportanceSelector(CandidateSelector):
    """Keep each candidate with probability ``w / w.max()``, where the weight
    ``w = q_min + mean_weight * (1 - exp(-s / scale))`` rises with a per-row
    statistic ``s`` (the ``column``). High-statistic datacubes are kept
    preferentially and common ones thinned out, without duplication. Needs the
    chosen ``column`` (a legacy CSV index has none).

    Parameters
    ----------
    column : str
        The sampling-index column to weight on, e.g. ``"mean"`` (default),
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

    def select(self, candidates: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        if self.column not in candidates.columns:
            raise ValueError(
                f"ImportanceSelector needs the {self.column!r} statistic column, absent from this "
                f"index (columns: {list(candidates.columns)}); a legacy CSV index carries only "
                f"(t, x, y), so use it without a selector (selector=None)."
            )
        # floor weight + a saturating response to the statistic; NaNs floored to q_min
        stat = np.nan_to_num(candidates[self.column].to_numpy(dtype=float), nan=0.0)
        weights = self.q_min + self.mean_weight * (1.0 - np.exp(-stat / self.scale))
        w_max = weights.max(initial=0.0)
        probs = weights / w_max if w_max > 0 else np.zeros_like(weights)
        return np.flatnonzero(rng.random(len(candidates)) < probs)


def get_selector(name: str, **kwargs) -> CandidateSelector:
    """Construct a registered selector by name, e.g. ``get_selector("importance", scale=2.0)``."""
    try:
        cls = SELECTOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown selector {name!r}; available: {sorted(SELECTOR_REGISTRY)}") from None
    return cls(**kwargs)
