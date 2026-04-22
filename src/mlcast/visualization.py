import torch
import torchvision
from PIL import Image
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger


def apply_radar_colormap(tensor: torch.Tensor) -> torch.Tensor:
    """Convert grayscale radar values to RGB using the STEPS-BE colorscale.

    Maps normalized values in [0, 1] (representing 0-60 dBZ) to a 14-color
    discrete colormap. Pixels below 10 dBZ are rendered as white.

    Parameters
    ----------
    tensor : torch.Tensor
        Grayscale tensor with values in [0, 1], of shape ``(N, 1, H, W)``.

    Returns
    -------
    rgb : torch.Tensor
        RGB tensor of shape ``(N, 3, H, W)`` with values in [0, 1].
    """
    colors = (
        torch.tensor(
            [
                [0, 255, 255],
                [0, 191, 255],
                [30, 144, 255],
                [0, 0, 255],
                [127, 255, 0],
                [50, 205, 50],
                [0, 128, 0],
                [0, 100, 0],
                [255, 255, 0],
                [255, 215, 0],
                [255, 165, 0],
                [255, 0, 0],
                [255, 0, 255],
                [139, 0, 139],
            ],
            dtype=torch.float32,
            device=tensor.device,
        )
        / 255.0
    )

    num_colors = len(colors)
    min_dbz_norm = 10 / 60
    max_dbz_norm = 1.0
    thresholds = torch.linspace(min_dbz_norm, max_dbz_norm, num_colors + 1, device=tensor.device)

    N, _, H, W = tensor.shape
    output = torch.ones(N, 3, H, W, dtype=torch.float32, device=tensor.device)

    for i in range(num_colors - 1):
        mask = (tensor[:, 0] >= thresholds[i]) & (tensor[:, 0] < thresholds[i + 1])
        for c in range(3):
            output[:, c][mask] = colors[i, c]

    mask = tensor[:, 0] >= thresholds[num_colors - 1]
    for c in range(3):
        output[:, c][mask] = colors[-1, c]

    return output


def _tensor_to_pil(grid: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in [0, 1] to a PIL image."""
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def log_images(
    past: torch.Tensor,
    future: torch.Tensor,
    preds: torch.Tensor,
    logger: TensorBoardLogger | MLFlowLogger,
    global_step: int,
    ensemble_size: int = 1,
    split: str = "val",
) -> None:
    """Log radar image grids to TensorBoard or MLflow."""
    sample_idx = 0

    past_sample = past[sample_idx]
    if ensemble_size > 1:
        past_sample = past_sample.mean(dim=1, keepdim=True)
    past_norm = (past_sample + 1) / 2
    past_rgb = apply_radar_colormap(past_norm)
    past_grid = torchvision.utils.make_grid(past_rgb, nrow=past_sample.shape[0])

    future_sample = future[sample_idx]
    preds_sample = preds[sample_idx]

    if ensemble_size > 1:
        preds_avg = preds_sample.mean(dim=1, keepdim=True)
        num_members_to_log = min(3, preds_sample.shape[1])
        rows = [future_sample, preds_avg] + [preds_sample[:, i : i + 1] for i in range(num_members_to_log)]
    else:
        rows = [future_sample, preds_sample]

    all_frames = torch.cat(rows, dim=0)
    all_frames_norm = (all_frames + 1) / 2
    all_frames_rgb = apply_radar_colormap(all_frames_norm)
    preds_grid = torchvision.utils.make_grid(all_frames_rgb, nrow=future_sample.shape[0])

    if isinstance(logger, TensorBoardLogger):
        logger.experiment.add_image(f"{split}/past", past_grid, global_step)
        logger.experiment.add_image(f"{split}/preds", preds_grid, global_step)
    elif isinstance(logger, MLFlowLogger):
        run_id = logger.run_id
        logger.experiment.log_image(run_id, _tensor_to_pil(past_grid), f"{split}/past_step{global_step}.png")
        logger.experiment.log_image(run_id, _tensor_to_pil(preds_grid), f"{split}/preds_step{global_step}.png")
