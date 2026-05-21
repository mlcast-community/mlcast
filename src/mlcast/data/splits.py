"""
Utilities to facilitate dataset splitting based on coordinate values, supporting two modes of specification:

1. Fraction mode: each split is defined by a fraction of the total coordinate
range (e.g., 0.7 for training, 0.2 for validation, and 0.1 for testing).
The fractions are resolved into coordinate value range tuples by inspecting
the coordinate values of the source dataset.

2. Tuple-range mode: each split is defined by an explicit (start, end) tuple of
coordinate values (e.g., ("2020-01-01", "2020-12-31") for training). In this
mode, the split values are passed through directly as the subset configuration
for each split.
"""

from collections.abc import Callable
from numbers import Real
from typing import Any

import xarray as xr
from loguru import logger
from torch.utils.data import Dataset

_SPLIT_NAMES = frozenset({"train", "val", "test"})
_SUPPORTED_COORDS = frozenset({"time"})


def splitting_uses_fractions(coord_splits: dict[str, Any]) -> bool:
    """Return whether a coordinate split config uses fraction mode.

    Parameters
    ----------
    coord_splits : dict[str, Any]
        Split configuration for a single coordinate.

    Returns
    -------
    bool
        ``True`` when all defined split values use numeric fractions rather than
        datetime tuples.
    """
    return all(
        split_val is None or (isinstance(split_val, Real) and not isinstance(split_val, bool))
        for split_val in coord_splits.values()
    )


def splitting_uses_tuple_ranges(coord_splits: dict[str, Any]) -> bool:
    """Return whether a coordinate split config uses tuple-range mode.

    Parameters
    ----------
    coord_splits : dict[str, Any]
        Split configuration for a single coordinate.

    Returns
    -------
    bool
        ``True`` when all defined split values use ``(start, end)`` tuples
        rather than numeric fractions.
    """
    return all(split_val is None or isinstance(split_val, tuple) for split_val in coord_splits.values())


def validate_splits(splits: dict[str, dict[str, Any]]) -> None:
    """Validate the nested ``splits`` configuration for the data module.

    Validates that:

    - the configuration is not empty;
    - each coordinate name is supported;
    - each split name is one of ``train``, ``val``, or ``test``;
    - each coordinate defines both ``train`` and ``val``;
    - each coordinate uses exactly one supported split mode;
    - fraction mode uses only float-like values for defined splits;
    - fraction mode split values sum to at most ``1.0``; and
    - tuple-range mode defines ``test`` explicitly as either a tuple or ``None``.

    Parameters
    ----------
    splits : dict[str, dict[str, Any]]
        Nested mapping ``{coord: {split_name: value, ...}, ...}`` describing
        dataset splits.

    Raises
    ------
    ValueError
        If the split configuration is empty, uses unsupported coordinates or
        split names, omits required split entries, mixes split modes within a
        coordinate, or provides invalid values for the selected mode.
    """
    if not splits:
        raise ValueError("splits must not be empty.")

    unknown_coords = set(splits) - _SUPPORTED_COORDS
    if unknown_coords:
        raise ValueError(
            f"Unknown coordinate(s) in splits: {sorted(unknown_coords)}. Supported: {sorted(_SUPPORTED_COORDS)}."
        )

    for coord, coord_splits in splits.items():
        unknown_names = set(coord_splits) - _SPLIT_NAMES
        if unknown_names:
            raise ValueError(
                f"Unknown split name(s) in splits[{coord!r}]: {sorted(unknown_names)}. "
                f"Must be one of {sorted(_SPLIT_NAMES)}."
            )

        for required in ("train", "val"):
            if required not in coord_splits:
                raise ValueError(f"splits[{coord!r}] must contain '{required}'.")

        train_is_tuple = isinstance(coord_splits["train"], tuple)
        val_is_tuple = isinstance(coord_splits["val"], tuple)
        if train_is_tuple != val_is_tuple:
            raise ValueError(
                f"Cannot mix datetime tuples and float ratios in splits[{coord!r}]. "
                "'train' and 'val' must both be floats (fraction mode) or both be "
                "(start, end) tuples (tuple-range mode)."
            )

        if splitting_uses_fractions(coord_splits):
            for split_name in ("train", "val"):
                split_val = coord_splits[split_name]
                if not isinstance(split_val, Real) or isinstance(split_val, bool):
                    raise ValueError(
                        f"In fraction mode splits[{coord!r}]['{split_name}'] must be float-like, got {split_val!r}."
                    )
            ratio_sum = coord_splits["train"] + coord_splits["val"]
            test_val = coord_splits.get("test")
            if test_val is not None and (not isinstance(test_val, Real) or isinstance(test_val, bool)):
                raise ValueError(
                    f"In fraction mode splits[{coord!r}]['test'] must be float-like or None, got {test_val!r}."
                )
            if isinstance(test_val, Real) and not isinstance(test_val, bool):
                ratio_sum += test_val
            if ratio_sum > 1.0 + 1e-9:
                raise ValueError(f"Split fractions in splits[{coord!r}] sum to {ratio_sum:.4f}, which exceeds 1.0.")
            if abs(ratio_sum - 1.0) > 1e-9:
                logger.warning(
                    "Split fractions in splits[{}] sum to {:.4f}, not 1.0. Any unallocated remainder will be unused.",
                    coord,
                    ratio_sum,
                )
        elif splitting_uses_tuple_ranges(coord_splits):
            if "test" not in coord_splits:
                raise ValueError(
                    f"In tuple-range mode splits[{coord!r}] must contain 'test' "
                    "(set to a (start, end) tuple or None to skip the test split)."
                )
            test_val = coord_splits["test"]
            if test_val is not None and not isinstance(test_val, tuple):
                raise ValueError(
                    f"In tuple-range mode splits[{coord!r}]['test'] must be a "
                    f"(start, end) tuple or None, got {test_val!r}."
                )
        else:
            raise ValueError(
                f"splits[{coord!r}] must use a single supported mode: all defined split values must be either "
                "float-like fractions or tuple ranges."
            )


def compute_split_ranges_from_splitting_ratios(
    dataset_factory: Callable[..., Dataset],
    coord: str,
    coord_splits: dict[str, Any],
) -> dict[str, tuple[str, str]]:
    """Resolve fraction-mode splits into inclusive coordinate ranges.

    Parameters
    ----------
    dataset_factory : Callable[..., Dataset]
        Dataset factory carrying the ``zarr_path`` and optional
        ``storage_options`` needed to open the source zarr store.
    coord : str
        Coordinate name to split. Currently this is expected to be ``"time"``.
    coord_splits : dict[str, Any]
        Fraction-mode split configuration for a single coordinate. Must contain
        float values for ``"train"`` and ``"val"``. ``"test"`` is optional;
        when omitted or set to ``None``, no test split range is returned.

    Returns
    -------
    dict[str, tuple[str, str]]
        Inclusive ``(start, end)`` coordinate ranges for the configured splits.
    """
    zarr_path = getattr(dataset_factory, "zarr_path", None) or dataset_factory.keywords["zarr_path"]
    storage_options = getattr(dataset_factory, "storage_options", None) or dataset_factory.keywords.get(
        "storage_options"
    )
    ds = xr.open_zarr(zarr_path, storage_options=storage_options)
    coord_vals = ds.indexes[coord]
    n = len(coord_vals)

    train_end = int(n * coord_splits["train"])
    val_end = train_end + int(n * coord_splits["val"])

    split_ranges = {
        "train": (str(coord_vals[0]), str(coord_vals[train_end - 1])),
        "val": (str(coord_vals[train_end]), str(coord_vals[val_end - 1])),
    }

    test_fraction = coord_splits.get("test")
    if isinstance(test_fraction, Real) and not isinstance(test_fraction, bool):
        test_end = val_end + int(n * test_fraction)
        split_ranges["test"] = (str(coord_vals[val_end]), str(coord_vals[test_end - 1]))

    return split_ranges
