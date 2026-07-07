"""GPU (PyTorch) backend for the per-chunk stats windowing.

Mirrors the CPU `stats._process_chunk` on CUDA tensors: the chunk is moved
to the GPU once, the three windowed stats are reduced onto the strided
candidate grid, and only the survivors are copied back. `nan_count` and
`frac_wet` match the CPU exactly; `sum`/`mean` agree to a few float32 ULP
(the GPU sums in a different order).

This module is imported only when ``--device cuda`` is selected, so torch
stays an optional dependency (the ``gpu`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _dim_window(a: torch.Tensor, dim: int, delta: int) -> torch.Tensor:
    """Sliding-window sum of size `delta` along `dim` (output len = n-delta+1)."""
    cs = torch.cumsum(a, dim=dim)
    pad_shape = list(cs.shape)
    pad_shape[dim] = 1
    padded = torch.cat([torch.zeros(pad_shape, dtype=cs.dtype, device=cs.device), cs], dim=dim)
    n = a.shape[dim]
    return padded.narrow(dim, delta, n - delta + 1) - padded.narrow(dim, 0, n - delta + 1)


def _strided_window(
    a: torch.Tensor,
    deltas: tuple[int, int, int],
    off_t: int,
    steps: tuple[int, int, int],
    keep_t: torch.Tensor,
) -> torch.Tensor:
    """Windowed sum reduced to the strided, gap-free candidate grid (see CPU twin)."""
    Dt, w, h = deltas
    step_t, step_x, step_y = steps
    s = _dim_window(a, 0, Dt)[off_t::step_t][keep_t]
    s = _dim_window(s, 1, w)[:, 0::step_x]
    s = _dim_window(s, 2, h)[:, :, 0::step_y]
    return s


def process_chunk(
    time_range: tuple[int, int],
    t_start_idx: int,
    chunk_np: NDArray,
    max_nan: int,
    wet_threshold: float,
    deltas: tuple[int, int, int],
    steps: tuple[int, int, int],
    valid_start_mask: NDArray[np.bool_],
    device: torch.device,
) -> dict[str, NDArray]:
    """GPU twin of `stats._process_chunk`. `chunk_np` is read on the CPU; this
    moves it to `device`, computes the strided stats, and returns CPU numpy
    arrays in the same column layout as the CPU path.
    """
    start_t, end_t = time_range
    # Data axes follow the MLCast source-data spec (§4.3): (time, y=height, x=width).
    Dt, dy, dx = deltas
    step_t, step_y, step_x = steps
    total_px = Dt * dy * dx
    off_t = (-(start_t + t_start_idx)) % step_t

    chunk = torch.from_numpy(chunk_np).to(device, non_blocking=True)

    # Strided, gap-free time-window starts (computed on the CPU, cheap).
    nt_win = chunk.shape[0] - Dt + 1
    t_rel_strided = np.arange(off_t, nt_win, step_t, dtype=np.int64)
    keep_np = valid_start_mask[t_rel_strided + start_t]
    keep_t = torch.from_numpy(keep_np).to(device)
    t_rel_kept = torch.from_numpy(t_rel_strided[keep_np]).to(device)

    nan_mask = torch.isnan(chunk)

    # Pass A: nan_count. cumsum keeps exact integer counts (< 2^31).
    ncw = _strided_window(nan_mask.to(torch.int32), deltas, off_t, steps, keep_t)
    a, b, c = torch.nonzero(ncw <= max_nan, as_tuple=True)  # a, b, c = axes time, y, x
    nan_count = ncw[a, b, c]

    # Pass B/C on the zero-filled chunk.
    chunk = torch.nan_to_num(chunk, nan=0.0)
    sum_vals = _strided_window(chunk, deltas, off_t, steps, keep_t)[a, b, c]
    wet_count = _strided_window((chunk > wet_threshold).to(torch.int32), deltas, off_t, steps, keep_t)[a, b, c]

    idx_t_abs = (t_rel_kept[a] + (start_t + t_start_idx)).to(torch.int32)
    idx_y = (b * step_y).to(torch.int32)
    idx_x = (c * step_x).to(torch.int32)

    valid_count = (total_px - nan_count).to(torch.float32)
    mean_vals = torch.where(valid_count > 0, sum_vals / valid_count, torch.full_like(sum_vals, float("nan")))
    frac_wet = wet_count.to(torch.float32) / total_px

    return {
        "t": idx_t_abs.cpu().numpy(),
        "x": idx_x.cpu().numpy(),
        "y": idx_y.cpu().numpy(),
        "nan_count": nan_count.to(torch.int32).cpu().numpy(),
        "sum": sum_vals.to(torch.float32).cpu().numpy(),
        "mean": mean_vals.to(torch.float32).cpu().numpy(),
        "frac_wet": frac_wet.cpu().numpy(),
    }
