# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
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
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.

    Raises:
        ValueError: If the ordering is not "xy" or "yx".
        ValueError: If the resolution is less than or equal to 0.
    """
    # check valid arguments
    if cfg.ordering not in ["xy", "yx"]:
        raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{cfg.ordering}'.")
    if cfg.resolution <= 0:
        raise ValueError(f"Resolution must be greater than 0. Received: '{cfg.resolution}'.")

    # resolve mesh grid indexing (note: torch meshgrid is different from numpy meshgrid)
    # check: https://github.com/pytorch/pytorch/issues/15301
    indexing = cfg.ordering if cfg.ordering == "xy" else "ij"
    # define grid pattern
    x = torch.arange(start=-cfg.size[0] / 2, end=cfg.size[0] / 2 + 1.0e-9, step=cfg.resolution, device=device)
    y = torch.arange(start=-cfg.size[1] / 2, end=cfg.size[1] / 2 + 1.0e-9, step=cfg.resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing=indexing)

    # store into ray starts
    num_rays = grid_x.numel()
    ray_starts = torch.zeros(num_rays, 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 1] = grid_y.flatten()

    # define ray-cast directions
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(cfg.direction), device=device)

    return ray_starts, ray_directions


def pinhole_camera_pattern(
    cfg: patterns_cfg.PinholeCameraPatternCfg, intrinsic_matrices: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """The image pattern for ray casting.

    .. caution::
        This function does not follow the standard pattern interface. It requires the intrinsic matrices
        of the cameras to be passed in. This is because we want to be able to randomize the intrinsic
        matrices of the cameras, which is not possible with the standard pattern interface.

    Args:
        cfg: The configuration instance for the pattern.
        intrinsic_matrices: The intrinsic matrices of the cameras. Shape is (N, 3, 3).
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays. The shape of the tensors are
        (N, H * W, 3) and (N, H * W, 3) respectively.
    """
    # get image plane mesh grid
    grid = torch.meshgrid(
        torch.arange(start=0, end=cfg.width, dtype=torch.int32, device=device),
        torch.arange(start=0, end=cfg.height, dtype=torch.int32, device=device),
        indexing="xy",
    )
    pixels = torch.vstack(list(map(torch.ravel, grid))).T
    # convert to homogeneous coordinate system
    pixels = torch.hstack([pixels, torch.ones((len(pixels), 1), device=device)])
    # move each pixel coordinate to the center of the pixel
    pixels += torch.tensor([[0.5, 0.5, 0]], device=device)
    # get pixel coordinates in camera frame
    pix_in_cam_frame = torch.matmul(torch.inverse(intrinsic_matrices), pixels.T)

    # robotics camera frame is (x forward, y left, z up) from camera frame with (x right, y down, z forward)
    # transform to robotics camera frame
    transform_vec = torch.tensor([1, -1, -1], device=device).unsqueeze(0).unsqueeze(2)
    pix_in_cam_frame = pix_in_cam_frame[:, [2, 0, 1], :] * transform_vec
    # normalize ray directions
    ray_directions = (pix_in_cam_frame / torch.norm(pix_in_cam_frame, dim=1, keepdim=True)).permute(0, 2, 1)
    # for camera, we always ray-cast from the sensor's origin
    ray_starts = torch.zeros_like(ray_directions, device=device)

    return ray_starts, ray_directions


def bpearl_pattern(cfg: patterns_cfg.BpearlPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The RS-Bpearl pattern for ray casting.

    The `Robosense RS-Bpearl`_ is a short-range LiDAR that has a 360 degrees x 90 degrees super wide
    field of view. It is designed for near-field blind-spots detection.

    .. _Robosense RS-Bpearl: https://www.roscomponents.com/en/lidar-laser-scanner/267-rs-bpearl.html

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
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


def lidar_pattern(cfg: patterns_cfg.LidarPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # Vertical angles
    vertical_angles = torch.linspace(cfg.vertical_fov_range[0], cfg.vertical_fov_range[1], cfg.channels)

    # If the horizontal field of view is 360 degrees, exclude the last point to avoid overlap
    if abs(abs(cfg.horizontal_fov_range[0] - cfg.horizontal_fov_range[1]) - 360.0) < 1e-6:
        up_to = -1
    else:
        up_to = None

    # Horizontal angles
    num_horizontal_angles = math.ceil((cfg.horizontal_fov_range[1] - cfg.horizontal_fov_range[0]) / cfg.horizontal_res)
    horizontal_angles = torch.linspace(cfg.horizontal_fov_range[0], cfg.horizontal_fov_range[1], num_horizontal_angles)[
        :up_to
    ]

    # Convert degrees to radians
    vertical_angles_rad = torch.deg2rad(vertical_angles)
    horizontal_angles_rad = torch.deg2rad(horizontal_angles)

    # Meshgrid to create a 2D array of angles
    v_angles, h_angles = torch.meshgrid(vertical_angles_rad, horizontal_angles_rad, indexing="ij")

    # Spherical to Cartesian conversion (assuming Z is up)
    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)

    return ray_starts, ray_directions
