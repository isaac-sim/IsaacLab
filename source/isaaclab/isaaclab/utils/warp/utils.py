# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp utility functions."""

from collections.abc import Sequence

import torch
import warp as wp


def make_mask_from_torch_ids(num_envs: int, env_ids: Sequence[int] | torch.Tensor, device: str) -> wp.array:
    """Create a warp boolean mask array from environment indices.

    Args:
        num_envs: Total number of environments.
        env_ids: Sequence or tensor of environment indices to set as True.
        device: Device to create the mask on.

    Returns:
        A warp array of shape (num_envs,) with dtype wp.bool, where True indicates
        the environment is in env_ids.
    """
    # Create a torch bool tensor
    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    mask[env_ids] = True
    # Convert to warp array
    return wp.from_torch(mask.contiguous(), dtype=wp.bool)
