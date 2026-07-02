# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, save_images_to_file
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.images import is_rgb_like, normalize_camera_image

from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

if TYPE_CHECKING:
    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg


class CartpoleCameraEnv(CartpoleEnv):
    """Cartpole environment driven by stacked camera observations."""

    cfg: CartpoleCameraEnvCfg

    def __init__(self, cfg: CartpoleCameraEnvCfg, render_mode: str | None = None, **kwargs):
        frame_stack = max(1, cfg.frame_stack)
        cfg.frame_stack = frame_stack
        if frame_stack > 1:
            single_channels = int(cfg.observation_space[0])
            cfg.observation_space = [single_channels * frame_stack, *cfg.observation_space[1:]]

        super().__init__(cfg, render_mode, **kwargs)

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "The Cartpole camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}"
            )

        self._stack: CircularBuffer | None = None
        if frame_stack > 1:
            # Channel-stack mode: buffer storage is laid out so that .stacked is a free
            # contiguous reshape into (B, K*C, H, W) -- no per-step permute/reshape alloc.
            self._stack = CircularBuffer(max_len=frame_stack, batch_size=self.num_envs, device=self.device, stack_dim=1)

    def _setup_scene(self):
        """Setup the scene with the cartpole and camera (no ground plane, which obstructs the view)."""
        self.cartpole = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = Camera(self.cfg.tiled_camera)
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.ClonePlan.from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)

        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions(global_prim_paths=[])

        # add articulation and sensors to scene
        self.scene.articulations["cartpole"] = self.cartpole
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        # quaternion for euler angles (roll, pitch, yaw) = (0, -45, -45) degrees
        light_orientation = (-0.14644663035869598, -0.3535534143447876, -0.3535534143447876, 0.8535533547401428)
        light_cfg.func("/World/Light", light_cfg, orientation=light_orientation)

    def _get_observations(self) -> dict:
        data_type = self.cfg.tiled_camera.data_types[0]
        camera_data = self._tiled_camera.data.output[data_type]

        rgb_like = is_rgb_like(data_type)
        # Defer normalize past the ring buffer when stacking RGB-like data so the ring holds
        # uint8 (4x cheaper per-step copies). Math is identical -- K frames live in disjoint
        # channel slices of (B, K*C, H, W).
        defer_normalize = self._stack is not None and rgb_like

        if data_type == "albedo":
            # albedo carries an extra alpha channel that the policy does not use
            camera_data = camera_data[..., :3]
        if rgb_like and not defer_normalize:
            camera_data = normalize_camera_image(camera_data, data_type)
        elif data_type == "depth":
            camera_data[camera_data == float("inf")] = 0

        # convert to channel-first [B, C, H, W] expected by the CNN policies (rsl_rl, rl_games, skrl)
        obs = camera_data.permute(0, 3, 1, 2).contiguous()

        if self._stack is not None:
            self._stack.append(obs)
            obs = self._stack.stacked

        if defer_normalize:
            # No ``out=`` -- a fresh float32 tensor is allocated per call. The caching
            # allocator returns a different block than the previous step's (still
            # referenced by the trainer), so the previous-iteration ``observations``
            # is not overwritten before ``record_transition`` reads it. See
            # :func:`isaaclab.utils.warp.ops.normalize_image_uint8` for the aliasing
            # hazard documentation.
            obs = normalize_camera_image(obs, data_type, channel_dim=1)
        elif self._stack is not None:
            # ``stacked`` is a view of the ring buffer storage which is overwritten on
            # the next ``env.step``; clone so the returned tensor outlives the next step.
            obs = obs.clone()

        if self.cfg.write_image_to_file:
            save_images_to_file(self._tiled_camera.data.output[data_type] / 255.0, f"cartpole_{data_type}.png")

        critic_obs = super()._get_observations()["policy"]
        return {"policy": obs, "critic": critic_obs}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if self._stack is not None:
            self._stack.reset(env_ids)
