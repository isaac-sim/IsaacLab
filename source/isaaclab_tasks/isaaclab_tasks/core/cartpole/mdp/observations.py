# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers.renderer import Renderer
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.images import is_rgb_like, normalize_camera_image
from isaaclab.utils.string import string_to_callable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera, CameraCfg


def resolve_camera_frame_stack(camera_cfg: CameraCfg, physics_cfg: PhysicsCfg) -> int:
    """Resolve the camera frame-stack size from backend capabilities.

    Two frames are required when the physics backend has no implicit damping and the
    renderer output contains no temporal information. Otherwise a single frame is used.

    Args:
        camera_cfg: Camera configuration containing the renderer and requested data type.
        physics_cfg: Physics backend configuration.

    Returns:
        The default number of camera frames to stack.
    """
    class_type = getattr(physics_cfg, "class_type", None)
    if class_type is None:
        return 1
    physics_manager_cls = string_to_callable(str(class_type)) if isinstance(class_type, str) else class_type
    if physics_manager_cls.provides_implicit_damping():
        return 1

    renderer_cfg = getattr(camera_cfg, "renderer_cfg", None)
    if renderer_cfg is None:
        return 2
    data_types = getattr(camera_cfg, "data_types", None) or []
    data_type = data_types[0] if data_types else ""
    renderer_cls = Renderer.resolve_class(renderer_cfg)
    return 1 if renderer_cls.provides_temporal_camera_data(data_type) else 2


class CameraImageStack(ManagerTermBase):
    """Return normalized channel-first camera images with optional frame stacking."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        camera: Camera = env.scene.sensors[sensor_cfg.name]
        frame_stack = getattr(env.cfg, "frame_stack", 1)
        if frame_stack < 0:
            frame_stack = resolve_camera_frame_stack(camera.cfg, env.cfg.sim.physics)
        elif frame_stack == 0:
            frame_stack = 1
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
