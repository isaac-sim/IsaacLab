# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow cartpole environment driven by lidar range images."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.images import CameraFrameStack

from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

from .cartpole_lidar_common import (
    CARTPOLE_LIDAR_AZIMUTH_COUNT,
    CARTPOLE_LIDAR_ELEVATION_COUNT,
    cartpole_lidar_range_image,
)

if TYPE_CHECKING:
    from .cartpole_lidar_direct_env_cfg import CartpoleLidarEnvCfg


class CartpoleLidarEnv(CartpoleEnv):
    """Cartpole environment driven by stacked lidar range images.

    Returns two stacked range images, one channel per scan, shaped ``(frame_stack, elevation, azimuth)``.
    Misses are 0 and a hit at the far range is 1. The critic still reads the joint state.
    """

    cfg: CartpoleLidarEnvCfg

    def __init__(self, cfg: CartpoleLidarEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.frame_stack = max(1, cfg.frame_stack)
        if isinstance(cfg.observation_space, list):
            cfg.observation_space = [
                cfg.frame_stack,
                CARTPOLE_LIDAR_ELEVATION_COUNT,
                CARTPOLE_LIDAR_AZIMUTH_COUNT,
            ]

        super().__init__(cfg, render_mode, **kwargs)

        self._lidar = self.scene["lidar"]
        self._frames = CameraFrameStack(self.num_envs, self.device, self.cfg.frame_stack, channel_first=True)

    def _get_observations(self) -> dict:
        critic_obs = super()._get_observations()["policy"]
        # channel-last single-channel image; the frame stack returns (B, frame_stack, H, W)
        scan = cartpole_lidar_range_image(self._lidar.data.points.torch).unsqueeze(-1)
        return {"policy": self._frames(scan), "critic": critic_obs}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self._frames.reset(env_ids)
