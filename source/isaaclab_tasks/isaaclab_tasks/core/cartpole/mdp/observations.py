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

from ..cartpole_common import cartpole_lidar_range_image

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


class LidarRangeImage(ManagerTermBase):
    """Stacked lidar range image, channel-first, for the cartpole policy.

    Each frame is :func:`~isaaclab_tasks.core.cartpole.cartpole_common.cartpole_lidar_range_image`:
    hit distance divided by the far range, with misses at 0. Frames are stacked oldest to newest.
    The stack is built from the term config and replaced when a call passes a different
    ``frame_stack`` or ``channel_first``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._frame_stack = cfg.params["frame_stack"]
        self._channel_first = cfg.params["channel_first"]
        self._frames = self._make_frames(env)

    def _make_frames(self, env: ManagerBasedRLEnv) -> CameraFrameStack:
        return CameraFrameStack(
            env.num_envs,
            env.device,
            frame_stack=self._frame_stack,
            channel_first=self._channel_first,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._frames.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
        channel_first: bool = True,
        frame_stack: int = 2,
    ) -> torch.Tensor:
        """Return the stacked range image for the named lidar.

        Args:
            env: Environment whose scene owns the lidar.
            sensor_cfg: Scene entity that names the lidar sensor. Defaults to ``SceneEntityCfg("lidar")``.
            channel_first: Whether to return ``(num_envs, frame_stack, elevation, azimuth)``.
                A value that differs from the current stack replaces that stack. Defaults to True.
            frame_stack: Number of recent scans to stack along the channel axis. A value that differs
                from the current stack replaces that stack. Defaults to 2.

        Returns:
            The stacked range image.
        """
        if frame_stack != self._frame_stack or channel_first != self._channel_first:
            self._frame_stack = frame_stack
            self._channel_first = channel_first
            self._frames = self._make_frames(env)

        points = env.scene.sensors[sensor_cfg.name].data.points.torch
        scan = cartpole_lidar_range_image(points).unsqueeze(-1)
        return self._frames(scan)
