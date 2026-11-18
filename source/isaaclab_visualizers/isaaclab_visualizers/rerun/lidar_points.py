# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-frame lidar hits for the Rerun visualizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.sensors.lidar.lidar import Lidar


def _iso8855_to_prim(points: torch.Tensor) -> torch.Tensor:
    """Map ISO 8855 sensor points onto the prim's camera axes.

    The beam travels along the prim's local ``-Z``. ``SENSOR`` returns use a different
    frame: ``+X`` is forward, ``+Y`` is left, and ``+Z`` is up. The pose quaternion
    rotates the prim axes, so a return has to be rewritten into those axes first.
    """
    forward, left, up = points.unbind(dim=-1)
    return torch.stack((-left, up, -forward), dim=-1)


def lidar_world_points(sensor: Lidar, env_ids: list[int] | None) -> np.ndarray:
    """Return finite lidar hits in world coordinates.

    ``WORLD`` returns are already in world coordinates. ``SENSOR`` returns are in the
    ISO 8855 sensor frame. They are rewritten onto the prim's camera axes, then
    rotated and translated by the pose the sensor writes to the renderer. Misses are
    omitted.

    Args:
        sensor: Lidar whose latest buffers are read.
        env_ids: Environments to include. None includes every environment.

    Returns:
        Hit positions with shape ``(num_hits, 3)``. An empty cloud has shape ``(0, 3)``.
    """
    points = sensor.data.points.torch
    valid = sensor.data.valid.torch
    if env_ids is not None:
        index = torch.as_tensor(env_ids, device=points.device, dtype=torch.long)
        points = points.index_select(0, index)
        valid = valid.index_select(0, index)
    else:
        index = None
    if sensor.cfg.frame_of_reference == "SENSOR":
        position = sensor._pos_w
        rotation = sensor._quat_w
        if position is None or rotation is None:
            return np.empty((0, 3), dtype=np.float32)
        position = position.torch
        rotation = rotation.torch
        if index is not None:
            position = position.index_select(0, index)
            rotation = rotation.index_select(0, index)
        points = quat_apply(rotation.unsqueeze(1), _iso8855_to_prim(points)) + position.unsqueeze(1)
    hits = points[valid & torch.isfinite(points).all(dim=-1)]
    return hits.detach().to(dtype=torch.float32, device="cpu").numpy()
