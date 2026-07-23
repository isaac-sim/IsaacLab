# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first curriculum terms for velocity locomotion environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_experimental.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


@wp.kernel
def _compute_terrain_level_updates(
    env_mask: wp.array(dtype=wp.bool),
    root_pos_w: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    command: wp.array(dtype=wp.float32, ndim=2),
    terrain_half_length: float,
    max_episode_length_s: float,
    move_up: wp.array(dtype=wp.bool),
    move_down: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()
    if env_mask[env_id]:
        position_delta = root_pos_w[env_id] - env_origins[env_id]
        distance = wp.sqrt(position_delta[0] * position_delta[0] + position_delta[1] * position_delta[1])
        command_distance = (
            wp.sqrt(command[env_id, 0] * command[env_id, 0] + command[env_id, 1] * command[env_id, 1])
            * max_episode_length_s
            * 0.5
        )
        should_move_up = distance > terrain_half_length
        move_up[env_id] = should_move_up
        move_down[env_id] = (distance < command_distance) and (not should_move_up)
    else:
        move_up[env_id] = False
        move_down[env_id] = False


@wp.kernel
def _compute_terrain_level_mean(
    terrain_levels: wp.array(dtype=wp.int64),
    scale: float,
    out: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    wp.atomic_add(out, 0, wp.float32(terrain_levels[env_id]) * scale)


class terrain_levels_vel(ManagerTermBase):
    """Update terrain levels from commanded locomotion progress using a boolean environment mask."""

    _curriculum_mask_native = True

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        """Initialize persistent views and decision buffers.

        Args:
            cfg: Curriculum term configuration.
            env: Environment containing the robot and terrain.
        """
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._terrain: TerrainImporter = env.scene.terrain
        self._root_pos_w_wp = self._asset.data.root_pos_w.warp
        self._env_origins_wp = env.env_origins_wp
        self._command_wp = env.command_manager.get_command_wp("base_velocity")
        self._terrain_levels_wp = self._terrain.terrain_levels_pa.warp
        self._move_up_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._move_down_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._terrain_half_length = float(self._terrain.cfg.terrain_generator.size[0]) * 0.5
        self._max_episode_length_s = float(env.max_episode_length_s)
        self._mean_scale = 1.0 / self.num_envs

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_mask: wp.array(dtype=wp.bool),
        out: wp.array(dtype=wp.float32),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Update selected terrain levels and write the global mean level.

        Args:
            env: Environment containing the per-environment random state.
            env_mask: Boolean Warp mask selecting environments to update.
            out: Persistent scalar output for the mean terrain level.
            asset_cfg: Robot scene entity configuration.
        """
        del asset_cfg
        wp.launch(
            kernel=_compute_terrain_level_updates,
            dim=self.num_envs,
            inputs=[
                env_mask,
                self._root_pos_w_wp,
                self._env_origins_wp,
                self._command_wp,
                self._terrain_half_length,
                self._max_episode_length_s,
                self._move_up_wp,
                self._move_down_wp,
            ],
            device=self.device,
        )
        self._terrain.update_env_origins_mask(
            env_mask,
            self._move_up_wp,
            self._move_down_wp,
            env.rng_state_wp,
        )
        wp.launch(
            kernel=_compute_terrain_level_mean,
            dim=self.num_envs,
            inputs=[self._terrain_levels_wp, self._mean_scale, out],
            device=self.device,
        )
