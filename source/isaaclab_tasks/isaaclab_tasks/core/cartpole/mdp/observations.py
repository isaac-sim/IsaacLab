# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the cartpole environments."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from typing_extensions import deprecated

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.images import CameraFrameStack, normalize_camera_image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera


@deprecated(
    "CameraImageStack is deprecated; use isaaclab.envs.mdp.image_rgb, image_depth or image_segmentation with"
    " channel_first=True and frame_stack instead. CameraImageStack will be removed in a future release."
)
class CameraImageStack(ManagerTermBase):
    """Return normalized channel-first camera images with optional frame stacking.

    .. deprecated::
        Use :class:`~isaaclab.envs.mdp.image_rgb`, :class:`~isaaclab.envs.mdp.image_depth` or
        :class:`~isaaclab.envs.mdp.image_segmentation` with ``channel_first=True`` and ``frame_stack``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        frame_stack = max(1, getattr(env.cfg, "frame_stack", 1))
        self._frames = CameraFrameStack(env.num_envs, env.device, frame_stack=frame_stack, channel_first=True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._frames.reset(env_ids)

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
        camera: Camera = env.scene.sensors[sensor_cfg.name]
        images = camera.data.output[data_type].torch
        if data_type == "albedo":
            images = images[..., :3]
        return self._frames(images, functools.partial(normalize_camera_image, data_type=data_type))
