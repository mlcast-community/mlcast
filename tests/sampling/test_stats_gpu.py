"""GPU backend tests — skipped unless torch + a CUDA GPU are available.

Validates the torch `_stats_gpu.process_chunk` against the CPU
`_process_chunk` and the independent brute-force oracle: integer columns
exact, float sum/mean to ``allclose`` (GPU float reduction order differs).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Reuse the CPU test's data generator, brute force, lexsort, and cases.
from test_stats_process import CASES, _brute_force, _lexsort, _make_data

from mlcast.sampling.commands import _stats_gpu
from mlcast.sampling.commands.stats import _process_chunk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU available")


@pytest.mark.parametrize("seed,deltas,steps,max_nan,wet_thr,start_t,t_start_idx", CASES)
def test_gpu_matches_cpu_and_brute(seed, deltas, steps, max_nan, wet_thr, start_t, t_start_idx):
    T_total, X, Y = 60, 44, 40
    data = _make_data(seed, T_total, X, Y)
    size_T = T_total - t_start_idx
    valid_start_mask = np.ones(size_T, dtype=bool)
    end_t = size_T - start_t

    chunk = data[start_t + t_start_idx : end_t + t_start_idx, :, :].astype(np.float32)  # snapshot

    gpu = _lexsort(
        _stats_gpu.process_chunk(
            (start_t, end_t),
            t_start_idx,
            chunk,
            max_nan,
            wet_thr,
            deltas,
            steps,
            valid_start_mask,
            torch.device("cuda"),
        )
    )
    cpu = _lexsort(
        _process_chunk(
            (start_t, end_t),
            t_start_idx,
            data.copy(),
            max_nan,
            wet_thr,
            deltas,
            steps,
            valid_start_mask,
        )
    )
    ref = _brute_force(chunk, deltas, steps, max_nan, wet_thr, start_t, t_start_idx, valid_start_mask)

    assert gpu["t"].size == cpu["t"].size == ref["t"].size
    # dtypes match the parquet schema
    for k in ("t", "x", "y", "nan_count"):
        assert gpu[k].dtype == np.int32
    for k in ("sum", "mean", "frac_wet"):
        assert gpu[k].dtype == np.float32

    # exact integer columns vs CPU and brute force
    for k in ("t", "x", "y", "nan_count"):
        assert np.array_equal(gpu[k], cpu[k]), f"{k} differs from CPU"
        assert np.array_equal(gpu[k].astype(np.int64), ref[k].astype(np.int64)), f"{k} differs from brute"

    # float columns: allclose. wet_count is exact (integer), but frac_wet's
    # division and the sum reduction round ~1 ULP differently on the GPU.
    assert np.allclose(gpu["frac_wet"], cpu["frac_wet"], rtol=1e-6, atol=1e-7), "frac_wet vs CPU"
    assert np.allclose(gpu["sum"], ref["sum"], rtol=1e-4, atol=1e-2), "sum mismatch vs brute"
    assert np.allclose(gpu["mean"], ref["mean"], rtol=1e-4, atol=1e-3, equal_nan=True), "mean mismatch"
