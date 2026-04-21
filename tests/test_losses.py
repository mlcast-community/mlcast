import pytest
import torch

from mlcast.losses import CRPS, afCRPS


@pytest.mark.parametrize("loss_class", [CRPS, afCRPS])
def test_crps_loss_shapes(loss_class):
    """Verify that CRPS and afCRPS losses output scalar for mean reduction,
    and correct shapes for other reductions."""
    # (B, T, M, *D) - M is ensemble size
    B, T, M, H, W = 2, 4, 3, 16, 16
    preds = torch.randn(B, T, M, H, W)
    # (B, T, C, *D) - C is usually 1
    target = torch.randn(B, T, 1, H, W)

    # mean reduction (default)
    loss_fn = loss_class(reduction="mean")
    loss = loss_fn(preds, target)
    assert loss.ndim == 0, f"{loss_class.__name__} with mean reduction should be scalar"

    # sum reduction
    loss_fn = loss_class(reduction="sum")
    loss = loss_fn(preds, target)
    assert loss.ndim == 0, f"{loss_class.__name__} with sum reduction should be scalar"

    # none reduction (elementwise)
    loss_fn = loss_class(reduction="none")
    loss = loss_fn(preds, target)
    # Expected output for none reduction is (B, T, 1, H, W)
    assert loss.shape == (B, T, 1, H, W), f"{loss_class.__name__} none reduction shape mismatch"

    # test with temporal_lambda > 0
    loss_fn_temporal = loss_class(reduction="none", temporal_lambda=0.1)
    loss_temporal = loss_fn_temporal(preds, target)
    # Temporal regularization usually removes the T dimension or broadcasts over it.
    # The current implementation returns (B, 1, 1, H, W) when temporal_lambda > 0
    assert loss_temporal.shape == (B, 1, 1, H, W), f"{loss_class.__name__} temporal shape mismatch"
