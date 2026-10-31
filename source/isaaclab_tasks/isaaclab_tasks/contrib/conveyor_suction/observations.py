# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse collision-geometry observations from the shared Newton ray caster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def collision_point_cloud(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return robot-frame ray hits divided by scan range, with a validity bit per point.

    The fixed overhead scan observes collision geometry, including robot occlusion.
    Missed rays have zero coordinates and a zero validity bit. The result has shape
    ``(num_envs, num_rays * 4)`` and does not require a camera or mesh resampling.
    """
    sensor = env.scene.sensors["collision_scan"]
    hits = sensor.data.ray_hits_w.torch
    valid = torch.isfinite(hits).all(dim=-1)
    robot = env.scene["robot"]
    relative = torch.where(valid[..., None], hits - robot.data.root_pos_w.torch[:, None], 0.0)
    rotation = robot.data.root_quat_w.torch[:, None].expand(-1, hits.shape[1], -1)
    points = math_utils.quat_apply_inverse(rotation, relative) / sensor.cfg.max_distance
    return torch.cat((points, valid[..., None].float()), dim=-1).flatten(start_dim=1)
