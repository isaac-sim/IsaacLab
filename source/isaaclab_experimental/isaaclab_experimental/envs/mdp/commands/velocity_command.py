# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first velocity command generator."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.kernels.state_kernels import body_ang_vel_from_root, body_lin_vel_from_root

from isaaclab.envs.mdp.commands._debug_vis import _VelocityCommandDebugVis
from isaaclab.utils.seed import WarpRng

from isaaclab_experimental.managers import CommandTerm
from isaaclab_experimental.utils.warp import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

logger = logging.getLogger(__name__)


@wp.kernel
def _accumulate_velocity_errors(
    command: wp.array(dtype=wp.float32, ndim=2),
    root_pose_w: wp.array(dtype=wp.transformf),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    error_xy_sum: wp.array(dtype=wp.float32),
    error_yaw_sum: wp.array(dtype=wp.float32),
    step_count: wp.array(dtype=wp.float32),
):
    """Add this step's planar velocity error [m/s] and yaw-rate error [rad/s] to the episode sums."""
    env_id = wp.tid()
    lin_vel_b = body_lin_vel_from_root(root_pose_w[env_id], root_vel_w[env_id])
    ang_vel_b = body_ang_vel_from_root(root_pose_w[env_id], root_vel_w[env_id])
    dx = command[env_id, 0] - lin_vel_b[0]
    dy = command[env_id, 1] - lin_vel_b[1]
    error_xy_sum[env_id] += wp.sqrt(dx * dx + dy * dy)
    error_yaw_sum[env_id] += wp.abs(command[env_id, 2] - ang_vel_b[2])
    step_count[env_id] += 1.0


@wp.kernel
def _finalize_velocity_metrics(
    env_mask: wp.array(dtype=wp.bool),
    error_xy_sum: wp.array(dtype=wp.float32),
    error_yaw_sum: wp.array(dtype=wp.float32),
    step_count: wp.array(dtype=wp.float32),
    xy_threshold: float,
    yaw_threshold: float,
    error_xy: wp.array(dtype=wp.float32),
    error_yaw: wp.array(dtype=wp.float32),
    success_rate: wp.array(dtype=wp.float32),
):
    """Turn the ending episode's error sums into means and a success flag, then clear the sums."""
    env_id = wp.tid()
    if env_mask[env_id]:
        denominator = wp.max(step_count[env_id], 1.0)
        mean_error_xy = error_xy_sum[env_id] / denominator
        mean_error_yaw = error_yaw_sum[env_id] / denominator
        error_xy[env_id] = mean_error_xy
        error_yaw[env_id] = mean_error_yaw
        success_rate[env_id] = wp.where((mean_error_xy < xy_threshold) and (mean_error_yaw < yaw_threshold), 1.0, 0.0)
        error_xy_sum[env_id] = 0.0
        error_yaw_sum[env_id] = 0.0
        step_count[env_id] = 0.0


@wp.kernel
def _resample_velocity_command(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    lin_vel_x: wp.vec2f,
    lin_vel_y: wp.vec2f,
    ang_vel_z: wp.vec2f,
    heading: wp.vec2f,
    heading_command: bool,
    rel_heading_envs: float,
    rel_standing_envs: float,
    command: wp.array(dtype=wp.float32, ndim=2),
    heading_target: wp.array(dtype=wp.float32),
    is_heading_env: wp.array(dtype=wp.bool),
    is_standing_env: wp.array(dtype=wp.bool),
):
    """Sample a new base velocity command, heading target and heading/standing roles for the masked envs."""
    env_id = wp.tid()
    if env_mask[env_id]:
        state = rng_state[env_id]
        command[env_id, 0] = wp.randf(state, lin_vel_x[0], lin_vel_x[1])
        command[env_id, 1] = wp.randf(state, lin_vel_y[0], lin_vel_y[1])
        command[env_id, 2] = wp.randf(state, ang_vel_z[0], ang_vel_z[1])
        if heading_command:
            heading_target[env_id] = wp.randf(state, heading[0], heading[1])
            is_heading_env[env_id] = wp.randf(state, 0.0, 1.0) <= rel_heading_envs
        is_standing_env[env_id] = wp.randf(state, 0.0, 1.0) <= rel_standing_envs
        rng_state[env_id] = state


@wp.kernel
def _update_velocity_command(
    root_pose_w: wp.array(dtype=wp.transformf),
    heading_target: wp.array(dtype=wp.float32),
    is_heading_env: wp.array(dtype=wp.bool),
    is_standing_env: wp.array(dtype=wp.bool),
    heading_command: bool,
    heading_control_stiffness: float,
    ang_vel_z: wp.vec2f,
    command: wp.array(dtype=wp.float32, ndim=2),
):
    """Track the heading target with a clipped yaw-rate command and zero the command of standing envs."""
    env_id = wp.tid()
    if heading_command and is_heading_env[env_id]:
        # heading of the base x-axis in the world frame
        forward_w = wp.quat_rotate(wp.transform_get_rotation(root_pose_w[env_id]), wp.vec3f(1.0, 0.0, 0.0))
        heading_error = wrap_to_pi(heading_target[env_id] - wp.atan2(forward_w[1], forward_w[0]))
        command[env_id, 2] = wp.clamp(heading_control_stiffness * heading_error, ang_vel_z[0], ang_vel_z[1])
    if is_standing_env[env_id]:
        command[env_id, 0] = 0.0
        command[env_id, 1] = 0.0
        command[env_id, 2] = 0.0


class UniformVelocityCommand(_VelocityCommandDebugVis, CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    Warp-first twin of :class:`isaaclab.envs.mdp.commands.UniformVelocityCommand`. The command comprises a
    linear velocity in x and y direction and an angular velocity around the z-axis, in the robot's base frame.
    Commands are sampled from the per-environment Warp random state, so they follow the stable distribution
    but not its random sequence.

    The command buffer is a Warp array; :attr:`vel_command_b` is its zero-copy Torch view.
    """

    cfg: UniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            logger.warning(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # -- command: x vel, y vel, yaw vel
        self._vel_command_b_wp = wp.zeros((self.num_envs, 3), dtype=wp.float32, device=self.device)
        self.vel_command_b = wp.to_torch(self._vel_command_b_wp)
        self._heading_target_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._is_heading_env_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        self._is_standing_env_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        # -- metrics: finalized per-episode means/rates, written at reset() and read by the base class
        self.metrics["error_vel_xy"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.metrics["error_vel_yaw"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.metrics["success_rate"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        # -- per-episode running sums (cleared at episode reset)
        self._error_xy_sum = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._error_yaw_sum = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._step_count = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

        # adds (optional) cmd kind and element names for leapp export
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

    """
    Properties
    """

    @property
    def command(self) -> wp.array:
        """The desired base velocity command in the base frame [m/s, m/s, rad/s]. Shape is (num_envs, 3)."""
        return self._vel_command_b_wp

    """
    Operations.
    """

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        env_mask: wp.array | None = None,
    ) -> dict[str, torch.Tensor]:
        """Finalize the ending episode's metrics, then reset the command of the selected environments.

        Args:
            env_ids: The specific environment indices to reset. If None, all environments are considered.
            env_mask: Boolean Warp mask of shape (num_envs,) selecting the environments to reset.
                Takes precedence over ``env_ids``.

        Returns:
            Persistent views of the metric means over the reset environments.
        """
        env_mask = self._env.resolve_env_mask(env_ids=env_ids, env_mask=env_mask)
        wp.launch(
            kernel=_finalize_velocity_metrics,
            dim=self.num_envs,
            inputs=[
                env_mask,
                self._error_xy_sum,
                self._error_yaw_sum,
                self._step_count,
                self.cfg.vel_xy_success_threshold,
                self.cfg.vel_yaw_success_threshold,
                self.metrics["error_vel_xy"],
                self.metrics["error_vel_yaw"],
                self.metrics["success_rate"],
            ],
            device=self.device,
        )
        return super().reset(env_mask=env_mask)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # the per-episode mean is finalized in reset(), independent of episode length
        wp.launch(
            kernel=_accumulate_velocity_errors,
            dim=self.num_envs,
            inputs=[
                self._vel_command_b_wp,
                self.robot.data.root_link_pose_w.warp,
                self.robot.data.root_com_vel_w.warp,
                self._error_xy_sum,
                self._error_yaw_sum,
                self._step_count,
            ],
            device=self.device,
        )

    def _resample_command(self, env_mask: wp.array):
        ranges = self.cfg.ranges
        wp.launch(
            kernel=_resample_velocity_command,
            dim=self.num_envs,
            inputs=[
                env_mask,
                WarpRng.state,
                wp.vec2f(*ranges.lin_vel_x),
                wp.vec2f(*ranges.lin_vel_y),
                wp.vec2f(*ranges.ang_vel_z),
                wp.vec2f(*(ranges.heading or (0.0, 0.0))),
                self.cfg.heading_command,
                self.cfg.rel_heading_envs,
                self.cfg.rel_standing_envs,
                self._vel_command_b_wp,
                self._heading_target_wp,
                self._is_heading_env_wp,
                self._is_standing_env_wp,
            ],
            device=self.device,
        )

    def _update_command(self):
        wp.launch(
            kernel=_update_velocity_command,
            dim=self.num_envs,
            inputs=[
                self.robot.data.root_link_pose_w.warp,
                self._heading_target_wp,
                self._is_heading_env_wp,
                self._is_standing_env_wp,
                self.cfg.heading_command,
                self.cfg.heading_control_stiffness,
                wp.vec2f(*self.cfg.ranges.ang_vel_z),
                self._vel_command_b_wp,
            ],
            device=self.device,
        )
