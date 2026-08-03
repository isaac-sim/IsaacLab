# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.images import is_rgb_like, normalize_camera_image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera


class CameraImageStack(ManagerTermBase):
    """Return normalized channel-first camera images with optional frame stacking."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        frame_stack = max(1, env.cfg.frame_stack)
        env.cfg.frame_stack = frame_stack

        self._stack = None
        if frame_stack > 1:
            self._stack = CircularBuffer(
                max_len=frame_stack,
                batch_size=env.num_envs,
                device=env.device,
                stack_dim=1,
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self._stack is not None:
            self._stack.reset(env_ids)

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
        camera: Camera = env.scene.sensors[sensor_cfg.name]
        camera_data = camera.data.output[data_type]

        rgb_like = is_rgb_like(data_type)
        defer_normalize = self._stack is not None and rgb_like
        if data_type == "albedo":
            camera_data = camera_data[..., :3]
        if rgb_like and not defer_normalize:
            camera_data = normalize_camera_image(camera_data, data_type)
        elif data_type == "depth":
            camera_data[camera_data == float("inf")] = 0

        observation = camera_data.permute(0, 3, 1, 2).contiguous()
        if self._stack is not None:
            self._stack.append(observation)
            observation = self._stack.stacked

        if defer_normalize:
            observation = normalize_camera_image(observation, data_type, channel_dim=1)
        elif self._stack is not None:
            observation = observation.clone()
        return observation
