# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import patterns_cfg


def grid_pattern(cfg: patterns_cfg.GridPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A regular grid pattern for ray casting.

    The grid pattern is made from rays that are parallel to each other. They span a 2D grid in the sensor's
    local coordinates from ``(-length/2, -width/2)`` to ``(length/2, width/2)``, which is defined
    by the ``size = (length, width)`` and ``resolution`` parameters in the config.

    Args:
        cfg (GridPatternCfg): The configuration instance for the pattern.
        device (str): The device to create the pattern on.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The starting positions and directions of the rays.
    """
    x = torch.arange(start=-cfg.size[0] / 2, end=cfg.size[0] / 2 + 1.0e-9, step=cfg.resolution, device=device)
    y = torch.arange(start=-cfg.size[1] / 2, end=cfg.size[1] / 2 + 1.0e-9, step=cfg.resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    num_rays = grid_x.numel()
    ray_starts = torch.zeros(num_rays, 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 1] = grid_y.flatten()

    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(cfg.direction), device=device)
    return ray_starts, ray_directions


def pinhole_camera_pattern(cfg: patterns_cfg.PinholeCameraPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The depth-image pattern for ray casting.

    Args:
        cfg (DepthImagePatternCfg): The configuration instance for the pattern.
        device (str): The device to create the pattern on.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The starting positions and directions of the rays.
    """
    x_grid = torch.full((cfg.height, cfg.width), cfg.far_plane, device=device)
    y_range = np.tan(np.deg2rad(cfg.horizontal_fov) / 2.0) * cfg.far_plane
    y = torch.linspace(y_range, -y_range, cfg.width, device=device)
    z_range = y_range * cfg.height / cfg.width
    z = torch.linspace(z_range, -z_range, cfg.height, device=device)
    y_grid, z_grid = torch.meshgrid(y, z, indexing="xy")

    ray_directions = torch.cat([x_grid.unsqueeze(2), y_grid.unsqueeze(2), z_grid.unsqueeze(2)], dim=2)
    ray_directions = torch.nn.functional.normalize(ray_directions, p=2.0, dim=-1).view(-1, 3)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions


def bpearl_pattern(cfg: patterns_cfg.BpearlPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The RS-Bpearl pattern for ray casting.

    The `Robosense RS-Bpearl`_ is a short-range LiDAR that has a 360 degrees x 90 degrees super wide
    field of view. It is designed for near-field blind-spots detection.

    .. _Robosense RS-Bpearl: https://www.roscomponents.com/en/lidar-laser-scanner/267-rs-bpearl.html

    Args:
        cfg (BpearlPatternCfg): The configuration instance for the pattern.
        device (str): The device to create the pattern on.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The starting positions and directions of the rays.
    """
    h = torch.arange(-cfg.horizontal_fov / 2, cfg.horizontal_fov / 2, cfg.horizontal_res, device=device)
    v = torch.tensor(list(cfg.vertical_ray_angles), device=device)

    pitch, yaw = torch.meshgrid(v, h, indexing="xy")
    pitch, yaw = torch.deg2rad(pitch.reshape(-1)), torch.deg2rad(yaw.reshape(-1))
    pitch += torch.pi / 2
    x = torch.sin(pitch) * torch.cos(yaw)
    y = torch.sin(pitch) * torch.sin(yaw)
    z = torch.cos(pitch)

    ray_directions = -torch.stack([x, y, z], dim=1)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions
