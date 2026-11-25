# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Cartpole lidar tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.images import CameraFrameStack

from ..cartpole_lidar_common import cartpole_lidar_range_image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _LidarRangeImage(ManagerTermBase):
    """Stack normalized lidar range images for the cartpole policy, oldest scan first."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._frames = CameraFrameStack(env.num_envs, env.device, frame_stack=2, channel_first=True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._frames.reset(env_ids)

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        points = env.scene[sensor_cfg.name].data.points.torch
        return self._frames(cartpole_lidar_range_image(points).unsqueeze(-1))
