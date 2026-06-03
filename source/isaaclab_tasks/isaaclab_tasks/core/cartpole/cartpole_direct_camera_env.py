# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, save_images_to_file
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.configclass import resolve_cfg_presets

from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

if TYPE_CHECKING:
    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg

SIMPLE_SHADING_TYPES = {
    "simple_shading_constant_diffuse",
    "simple_shading_diffuse_mdl",
    "simple_shading_full_mdl",
}


class CartpoleCameraEnv(CartpoleEnv):
    """Cartpole environment driven by camera observations.

    Uses temporal observations for the Newton + Warp combo as it does not have the same implicit benefit
    as the RTX renderer (implicit temporal anti-aliasing).
    """

    cfg: CartpoleCameraEnvCfg

    @staticmethod
    def _resolve_frame_stack_default(camera_cfg, physics_cfg) -> int:
        """Return ``2`` for the Newton + Warp combo (no implicit damping, no temporal AA),
        ``1`` otherwise."""
        from isaaclab_newton.physics import NewtonCfg
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        is_newton_warp = isinstance(physics_cfg, NewtonCfg) and isinstance(
            getattr(camera_cfg, "renderer_cfg", None), NewtonWarpRendererCfg
        )
        return 2 if is_newton_warp else 1

    def __init__(self, cfg: CartpoleCameraEnvCfg, render_mode: str | None = None, **kwargs):
        # Flatten preset wrappers so the frame-stack resolution below sees concrete types.
        # Idempotent — base ``DirectRLEnv.__init__`` calls this again with no effect.
        resolve_cfg_presets(cfg)

        frame_stack = getattr(cfg, "frame_stack", 1)
        if frame_stack < 0:
            frame_stack = self._resolve_frame_stack_default(cfg.tiled_camera, cfg.sim.physics)
        elif frame_stack == 0:
            frame_stack = 1
        if hasattr(cfg, "frame_stack"):
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
            self._stack = CircularBuffer(max_len=frame_stack, batch_size=self.num_envs, device=self.device)

    def _setup_scene(self):
        """Setup the scene with the cartpole and camera (no ground plane, which obstructs the view)."""
        self.cartpole = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = Camera(self.cfg.tiled_camera)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions(global_prim_paths=[])

        # add articulation and sensors to scene
        self.scene.articulations["cartpole"] = self.cartpole
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        data_type = self.cfg.tiled_camera.data_types[0]
        camera_data = self._tiled_camera.data.output[data_type]

        if data_type == "albedo" or data_type == "rgb" or data_type in SIMPLE_SHADING_TYPES:
            # albedo carries an extra alpha channel that the policy does not use
            if data_type == "albedo":
                camera_data = camera_data[..., :3]
            # scale to [0, 1] and mean-center per image for better training results
            camera_data = camera_data / 255.0
            camera_data -= torch.mean(camera_data, dim=(1, 2), keepdim=True)
        elif data_type == "depth":
            camera_data[camera_data == float("inf")] = 0

        # convert to channel-first [B, C, H, W] expected by the CNN policies (rsl_rl, rl_games, skrl)
        obs = camera_data.permute(0, 3, 1, 2).contiguous()

        if self._stack is not None:
            self._stack.append(obs)
            # CircularBuffer.buffer is (B, K, C, H, W) oldest->newest along dim 1.
            # Channel-stack: flatten the adjacent (K, C) dims so the channel axis
            # reads oldest_C, ..., newest_C.
            stacked = self._stack.buffer
            b, k, c, h, w = stacked.shape
            obs = stacked.reshape(b, k * c, h, w).clone()

        if self.cfg.write_image_to_file:
            save_images_to_file(self._tiled_camera.data.output[data_type] / 255.0, f"cartpole_{data_type}.png")

        return {"policy": obs}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if self._stack is not None:
            self._stack.reset(env_ids)
