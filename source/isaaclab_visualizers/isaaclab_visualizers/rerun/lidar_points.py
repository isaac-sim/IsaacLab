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


def lidar_world_points(sensor: Lidar, env_ids: list[int] | None) -> np.ndarray:
    """Return finite lidar hits in world coordinates.

    ``WORLD`` returns are already in world coordinates. ``SENSOR`` returns are rotated
    and translated by the pose the sensor writes to the renderer. Misses are omitted.

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
        points = quat_apply(rotation.unsqueeze(1), points) + position.unsqueeze(1)
    hits = points[valid & torch.isfinite(points).all(dim=-1)]
    return hits.detach().to(dtype=torch.float32, device="cpu").numpy()
