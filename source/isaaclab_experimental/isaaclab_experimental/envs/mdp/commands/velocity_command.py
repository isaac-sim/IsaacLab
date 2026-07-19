# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native velocity command generator."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands._debug_vis import _VelocityCommandDebugVis

from isaaclab_experimental.managers import CommandTerm
from isaaclab_experimental.utils.warp import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformVelocityCommandCfg

logger = logging.getLogger(__name__)


@wp.kernel
def _accumulate_velocity_metrics(
    command: wp.array(dtype=wp.float32, ndim=2),
    root_lin_vel_b: wp.array(dtype=wp.vec3f),
    root_ang_vel_b: wp.array(dtype=wp.vec3f),
    error_xy_sum: wp.array(dtype=wp.float32),
    error_yaw_sum: wp.array(dtype=wp.float32),
    step_count: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    dx = command[env_id, 0] - root_lin_vel_b[env_id][0]
    dy = command[env_id, 1] - root_lin_vel_b[env_id][1]
    error_xy_sum[env_id] += wp.sqrt(dx * dx + dy * dy)
    error_yaw_sum[env_id] += wp.abs(command[env_id, 2] - root_ang_vel_b[env_id][2])
    step_count[env_id] += 1.0


@wp.kernel
def _finalize_velocity_metrics(
    env_mask: wp.array(dtype=wp.bool),
    error_xy_sum: wp.array(dtype=wp.float32),
    error_yaw_sum: wp.array(dtype=wp.float32),
    step_count: wp.array(dtype=wp.float32),
    error_xy: wp.array(dtype=wp.float32),
    error_yaw: wp.array(dtype=wp.float32),
    success_rate: wp.array(dtype=wp.float32),
    error_xy_threshold: float,
    error_yaw_threshold: float,
):
    env_id = wp.tid()
    if env_mask[env_id]:
        denominator = wp.max(step_count[env_id], 1.0)
        mean_error_xy = error_xy_sum[env_id] / denominator
        mean_error_yaw = error_yaw_sum[env_id] / denominator
        error_xy[env_id] = mean_error_xy
        error_yaw[env_id] = mean_error_yaw
        success_rate[env_id] = wp.where(
            (mean_error_xy < error_xy_threshold) and (mean_error_yaw < error_yaw_threshold), 1.0, 0.0
        )
        error_xy_sum[env_id] = 0.0
        error_yaw_sum[env_id] = 0.0
        step_count[env_id] = 0.0


@wp.kernel
def _resample_velocity_command(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    command: wp.array(dtype=wp.float32, ndim=2),
    heading_target: wp.array(dtype=wp.float32),
    is_heading_env: wp.array(dtype=wp.bool),
    is_standing_env: wp.array(dtype=wp.bool),
    lin_vel_x_min: float,
    lin_vel_x_max: float,
    lin_vel_y_min: float,
    lin_vel_y_max: float,
    ang_vel_z_min: float,
    ang_vel_z_max: float,
    heading_min: float,
    heading_max: float,
    heading_command: bool,
    rel_heading_envs: float,
    rel_standing_envs: float,
):
    env_id = wp.tid()
    if env_mask[env_id]:
        state = rng_state[env_id]
        command[env_id, 0] = wp.randf(state, lin_vel_x_min, lin_vel_x_max)
        command[env_id, 1] = wp.randf(state, lin_vel_y_min, lin_vel_y_max)
        command[env_id, 2] = wp.randf(state, ang_vel_z_min, ang_vel_z_max)
        if heading_command:
            heading_target[env_id] = wp.randf(state, heading_min, heading_max)
            is_heading_env[env_id] = wp.randf(state, 0.0, 1.0) <= rel_heading_envs
        else:
            is_heading_env[env_id] = False
        is_standing_env[env_id] = wp.randf(state, 0.0, 1.0) <= rel_standing_envs
        rng_state[env_id] = state


@wp.kernel
def _update_velocity_command(
    command: wp.array(dtype=wp.float32, ndim=2),
    heading_target: wp.array(dtype=wp.float32),
    is_heading_env: wp.array(dtype=wp.bool),
    is_standing_env: wp.array(dtype=wp.bool),
    heading_w: wp.array(dtype=wp.float32),
    heading_command: bool,
    heading_control_stiffness: float,
    ang_vel_z_min: float,
    ang_vel_z_max: float,
):
    env_id = wp.tid()
    if heading_command and is_heading_env[env_id]:
        heading_error = wrap_to_pi(heading_target[env_id] - heading_w[env_id])
        command[env_id, 2] = wp.clamp(
            heading_control_stiffness * heading_error,
            ang_vel_z_min,
            ang_vel_z_max,
        )
    if is_standing_env[env_id]:
        command[env_id, 0] = 0.0
        command[env_id, 1] = 0.0
        command[env_id, 2] = 0.0


class UniformVelocityCommand(_VelocityCommandDebugVis, CommandTerm):
    """Generate an SE(2) velocity command from uniform distributions."""

    cfg: UniformVelocityCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: Configuration for the command generator.
            env: Environment containing the commanded articulation.

        Raises:
            ValueError: If heading commands are enabled without a heading range.
        """
        super().__init__(cfg, env)

        if cfg.heading_command and cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if cfg.ranges.heading and not cfg.heading_command:
            logger.warning(
                f"The velocity command has the 'ranges.heading' attribute set to '{cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        self.robot: Articulation = env.scene[cfg.asset_name]

        self._vel_command_b_wp = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)
        self.vel_command_b = wp.to_torch(self._vel_command_b_wp)
        self._heading_target_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.heading_target = wp.to_torch(self._heading_target_wp)
        self._is_heading_env_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self.is_heading_env = wp.to_torch(self._is_heading_env_wp)
        self._is_standing_env_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self.is_standing_env = wp.to_torch(self._is_standing_env_wp)

        self.metrics["error_vel_xy"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.metrics["error_vel_yaw"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.metrics["success_rate"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._error_xy_sum_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._error_yaw_sum_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._step_count_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/velocity"
        self.cfg.element_names = self.cfg.element_names or ["lin_vel_x", "lin_vel_y", "ang_vel_z"]

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Desired base velocity command [m/s, m/s, rad/s], shape ``(num_envs, 3)``."""
        return self.vel_command_b

    @property
    def command_wp(self) -> wp.array(dtype=wp.float32, ndim=2):
        """Pointer-stable desired base velocity command [m/s, m/s, rad/s]."""
        return self._vel_command_b_wp

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        env_mask: wp.array | None = None,
    ) -> dict[str, torch.Tensor]:
        """Finalize episode metrics and reset selected command state.

        Args:
            env_ids: Environment indices used by compatibility call sites.
            env_mask: Boolean Warp mask selecting environments to reset. Takes precedence over ``env_ids``.

        Returns:
            Persistent scalar metric views for logging.
        """
        env_mask = self._resolve_reset_mask(env_ids, env_mask)
        wp.launch(
            kernel=_finalize_velocity_metrics,
            dim=self.num_envs,
            inputs=[
                env_mask,
                self._error_xy_sum_wp,
                self._error_yaw_sum_wp,
                self._step_count_wp,
                self.metrics["error_vel_xy"],
                self.metrics["error_vel_yaw"],
                self.metrics["success_rate"],
                self.cfg.vel_xy_success_threshold,
                self.cfg.vel_yaw_success_threshold,
            ],
            device=self.device,
        )
        return super().reset(env_mask=env_mask)

    def _update_metrics(self):
        wp.launch(
            kernel=_accumulate_velocity_metrics,
            dim=self.num_envs,
            inputs=[
                self._vel_command_b_wp,
                self.robot.data.root_lin_vel_b.warp,
                self.robot.data.root_ang_vel_b.warp,
                self._error_xy_sum_wp,
                self._error_yaw_sum_wp,
                self._step_count_wp,
            ],
            device=self.device,
        )

    def _resample_command(self, env_mask: wp.array):
        heading_range = self.cfg.ranges.heading if self.cfg.ranges.heading is not None else (0.0, 0.0)
        wp.launch(
            kernel=_resample_velocity_command,
            dim=self.num_envs,
            inputs=[
                env_mask,
                self._env.rng_state_wp,
                self._vel_command_b_wp,
                self._heading_target_wp,
                self._is_heading_env_wp,
                self._is_standing_env_wp,
                self.cfg.ranges.lin_vel_x[0],
                self.cfg.ranges.lin_vel_x[1],
                self.cfg.ranges.lin_vel_y[0],
                self.cfg.ranges.lin_vel_y[1],
                self.cfg.ranges.ang_vel_z[0],
                self.cfg.ranges.ang_vel_z[1],
                heading_range[0],
                heading_range[1],
                self.cfg.heading_command,
                self.cfg.rel_heading_envs,
                self.cfg.rel_standing_envs,
            ],
            device=self.device,
        )

    def _update_command(self):
        wp.launch(
            kernel=_update_velocity_command,
            dim=self.num_envs,
            inputs=[
                self._vel_command_b_wp,
                self._heading_target_wp,
                self._is_heading_env_wp,
                self._is_standing_env_wp,
                self.robot.data.heading_w.warp,
                self.cfg.heading_command,
                self.cfg.heading_control_stiffness,
                self.cfg.ranges.ang_vel_z[0],
                self.cfg.ranges.ang_vel_z[1],
            ],
            device=self.device,
        )
