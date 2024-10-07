from typing import Optional

import torch
from torchtyping import TensorType


def apply_pca_colormap_return_proj(
    image: TensorType["bs":..., "d"],
    proj_V: Optional[TensorType] = None,
    low_rank_min: Optional[TensorType] = None,
    low_rank_max: Optional[TensorType] = None,
    niter: int = 5,
        n_components: int = 3,
) -> TensorType["bs":..., "rgb":3]:
    """Convert a multichannel image to color using PCA.

    Args:
        image: Multichannel image.
        proj_V: Projection matrix to use. If None, use torch low rank PCA.

    Returns:
        Colored PCA image of the multichannel input image.
    """
    image_flat = image.reshape(-1, image.shape[-1])

    # Modified from https://github.com/pfnet-research/distilled-feature-fields/blob/master/train.py
    if proj_V is None:
        mean = image_flat.mean(0)
        with torch.no_grad():
            U, S, V = torch.pca_lowrank(image_flat - mean, niter=niter)
        proj_V = V[:, :n_components]

    low_rank = image_flat @ proj_V
    if low_rank_min is None:
        low_rank_min = torch.quantile(low_rank, 0.01, dim=0)
    if low_rank_max is None:
        low_rank_max = torch.quantile(low_rank, 0.99, dim=0)

    low_rank = (low_rank - low_rank_min) / (low_rank_max - low_rank_min)
    low_rank = torch.clamp(low_rank, 0, 1)

    colored_image = low_rank.reshape(image.shape[:-1] + (3,))
    return colored_image, (proj_V, low_rank_min, low_rank_max)


def apply_pca_colormap(
    image: TensorType["bs":..., "d"],
    project_info=None,
    niter: int = 5,
) -> TensorType["bs":..., "rgb":3]:
    if project_info is None:
        project_info = (None, None, None)
    proj_V, low_rank_min, low_rank_max = project_info
    return apply_pca_colormap_return_proj(image, proj_V, low_rank_min, low_rank_max, niter)