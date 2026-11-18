# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-workflow cartpole environment driven by camera observations."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.sensors import save_images_to_file
from isaaclab.utils.images import CameraFrameStack, normalize_camera_image

from .cartpole_common import (
    CARTPOLE_LIDAR_AZIMUTH_COUNT,
    CARTPOLE_LIDAR_ELEVATION_COUNT,
    cartpole_lidar_range_image,
)
from .cartpole_direct_env import CartpoleEnv

if TYPE_CHECKING:
    from .cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg


class CartpoleCameraEnv(CartpoleEnv):
    """Cartpole environment driven by stacked camera observations.

    The ``lidar`` preset instead returns two stacked range images, one channel per scan, shaped
    ``(frame_stack, elevation, azimuth)``. Misses are 0 and a hit at the far range is 1. The critic
    still reads the joint state. The side RGB camera remains in the scene.
    """

    cfg: CartpoleCameraEnvCfg

    def __init__(self, cfg: CartpoleCameraEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.frame_stack = max(1, cfg.frame_stack)
        uses_lidar = getattr(cfg.scene, "lidar", None) is not None
        if isinstance(cfg.observation_space, list):
            if uses_lidar:
                height, width = CARTPOLE_LIDAR_ELEVATION_COUNT, CARTPOLE_LIDAR_AZIMUTH_COUNT
            else:
                height, width = int(cfg.scene.tiled_camera.height), int(cfg.scene.tiled_camera.width)
            cfg.observation_space = [int(cfg.observation_space[0]) * cfg.frame_stack, height, width]

        super().__init__(cfg, render_mode, **kwargs)

        self._tiled_camera = self.scene["tiled_camera"]
        self._lidar = self.scene["lidar"] if uses_lidar else None
        if self._lidar is None and len(self.cfg.scene.tiled_camera.data_types) != 1:
            raise ValueError(
                "The Cartpole camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.scene.tiled_camera.data_types}"
            )

        self._frames = CameraFrameStack(self.num_envs, self.device, self.cfg.frame_stack, channel_first=True)

    def _get_observations(self) -> dict:
        critic_obs = super()._get_observations()["policy"]
        if self._lidar is not None:
            # channel-last single-channel image; the frame stack returns (B, frame_stack, H, W)
            scan = cartpole_lidar_range_image(self._lidar.data.points.torch).unsqueeze(-1)
            return {"policy": self._frames(scan), "critic": critic_obs}

        data_type = self.cfg.scene.tiled_camera.data_types[0]
        images = self._tiled_camera.data.output[data_type].torch
        if data_type == "albedo":
            # albedo carries an extra alpha channel that the policy does not use
            images = images[..., :3]
        # channel-first [B, C, H, W] as expected by the CNN policies (rsl_rl, rl_games, skrl)
        obs = self._frames(images, functools.partial(normalize_camera_image, data_type=data_type))

        if self.cfg.write_image_to_file:
            save_images_to_file(self._tiled_camera.data.output[data_type].torch / 255.0, f"cartpole_{data_type}.png")

        return {"policy": obs, "critic": critic_obs}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if self._frames is not None:
            self._frames.reset(env_ids)
