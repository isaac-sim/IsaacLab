# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-first pose command generator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.commands._debug_vis import _PoseCommandDebugVis
from isaaclab.utils.leapp import POSE7_ELEMENT_NAMES

from isaaclab_experimental.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg


@wp.kernel
def _resample_pose_command(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    pos_x: wp.vec2f,
    pos_y: wp.vec2f,
    pos_z: wp.vec2f,
    roll: wp.vec2f,
    pitch: wp.vec2f,
    yaw: wp.vec2f,
    make_quat_unique: bool,
    command: wp.array(dtype=wp.float32, ndim=2),
):
    """Sample a new root-frame pose command: uniform position [m] and roll-pitch-yaw [rad] as (x, y, z, w)."""
    env_id = wp.tid()
    if env_mask[env_id]:
        state = rng_state[env_id]
        command[env_id, 0] = wp.randf(state, pos_x[0], pos_x[1])
        command[env_id, 1] = wp.randf(state, pos_y[0], pos_y[1])
        command[env_id, 2] = wp.randf(state, pos_z[0], pos_z[1])
        # extrinsic XYZ Euler angles, matching ``quat_from_euler_xyz``
        q_roll = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), wp.randf(state, roll[0], roll[1]))
        q_pitch = wp.quat_from_axis_angle(wp.vec3f(0.0, 1.0, 0.0), wp.randf(state, pitch[0], pitch[1]))
        q_yaw = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), wp.randf(state, yaw[0], yaw[1]))
        quat = q_yaw * q_pitch * q_roll
        if make_quat_unique and quat[3] < 0.0:
            quat = -quat
        command[env_id, 3] = quat[0]
        command[env_id, 4] = quat[1]
        command[env_id, 5] = quat[2]
        command[env_id, 6] = quat[3]
        rng_state[env_id] = state


@wp.kernel
def _update_pose_metrics(
    command_b: wp.array(dtype=wp.float32, ndim=2),
    root_pos_w: wp.array(dtype=wp.vec3f),
    root_quat_w: wp.array(dtype=wp.quatf),
    body_pos_w: wp.array(dtype=wp.vec3f, ndim=2),
    body_quat_w: wp.array(dtype=wp.quatf, ndim=2),
    body_idx: int,
    check_position: bool,
    position_threshold: float,
    check_orientation: bool,
    orientation_threshold: float,
    command_w: wp.array(dtype=wp.float32, ndim=2),
    position_error: wp.array(dtype=wp.float32),
    orientation_error: wp.array(dtype=wp.float32),
    succeeded: wp.array(dtype=wp.bool),
):
    """Express the command in the world frame, compute the tracking errors and update the sticky success flag.

    The orientation error is the axis-angle magnitude of the body-to-goal rotation, ``2 atan2(|xyz|, |w|)``, as in
    :func:`~isaaclab.utils.math.compute_pose_error`. Each configured threshold is an accept predicate
    (``error < threshold``); with no threshold configured, the success flag is left untouched.
    """
    env_id = wp.tid()
    root_quat = root_quat_w[env_id]
    goal_pos_b = wp.vec3f(command_b[env_id, 0], command_b[env_id, 1], command_b[env_id, 2])
    goal_quat_b = wp.quatf(command_b[env_id, 3], command_b[env_id, 4], command_b[env_id, 5], command_b[env_id, 6])
    goal_pos_w = root_pos_w[env_id] + wp.quat_rotate(root_quat, goal_pos_b)
    goal_quat_w = root_quat * goal_quat_b
    command_w[env_id, 0] = goal_pos_w[0]
    command_w[env_id, 1] = goal_pos_w[1]
    command_w[env_id, 2] = goal_pos_w[2]
    command_w[env_id, 3] = goal_quat_w[0]
    command_w[env_id, 4] = goal_quat_w[1]
    command_w[env_id, 5] = goal_quat_w[2]
    command_w[env_id, 6] = goal_quat_w[3]

    pos_error = wp.length(body_pos_w[env_id, body_idx] - goal_pos_w)
    quat_error = body_quat_w[env_id, body_idx] * wp.quat_inverse(goal_quat_w)
    rot_error = 2.0 * wp.atan2(wp.length(wp.vec3f(quat_error[0], quat_error[1], quat_error[2])), wp.abs(quat_error[3]))
    position_error[env_id] = pos_error
    orientation_error[env_id] = rot_error

    if check_position or check_orientation:
        success = bool(True)
        if check_position:
            success = success and (pos_error < position_threshold)
        if check_orientation:
            success = success and (rot_error < orientation_threshold)
        if success:
            succeeded[env_id] = True


@wp.kernel
def _finalize_pose_success(
    env_mask: wp.array(dtype=wp.bool),
    succeeded: wp.array(dtype=wp.bool),
    success_rate: wp.array(dtype=wp.float32),
):
    """Move the episode's sticky success flag into the success-rate metric and clear it for the next episode."""
    env_id = wp.tid()
    if env_mask[env_id]:
        success_rate[env_id] = wp.where(succeeded[env_id], 1.0, 0.0)
        succeeded[env_id] = False


class UniformPoseCommand(_PoseCommandDebugVis, CommandTerm):
    """Command generator for generating pose commands uniformly.

    Warp-first twin of :class:`isaaclab.envs.mdp.commands.UniformPoseCommand`. Positions are sampled uniformly
    within the configured ranges and orientations from uniform roll-pitch-yaw angles, in the robot's root frame.
    Commands are sampled from the per-environment Warp random state, so they follow the stable distribution
    but not its random sequence.

    When a success threshold is configured, the per-episode "ever successful" flag is kept in :attr:`_succeeded`
    (shared with the ``pose_command_success`` termination) and logged under ``Metrics/success_rate``.

    The command buffers are Warp arrays; :attr:`pose_command_b`, :attr:`pose_command_w` and :attr:`_succeeded`
    are their zero-copy Torch views.
    """

    cfg: UniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # -- commands: (x, y, z, qx, qy, qz, qw) in root frame and in world frame
        self._pose_command_b_wp = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)
        self._pose_command_w_wp = wp.zeros_like(self._pose_command_b_wp)
        self.pose_command_b = wp.to_torch(self._pose_command_b_wp)
        self.pose_command_w = wp.to_torch(self._pose_command_w_wp)
        self.pose_command_b[:, 6] = 1.0
        # -- metrics
        self.metrics["position_error"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.metrics["orientation_error"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        # -- per-episode sticky success bit (only used when at least one success threshold is set)
        self._track_success = (
            cfg.position_success_threshold is not None or cfg.orientation_success_threshold is not None
        )
        self._succeeded_wp = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        if self._track_success:
            self._succeeded = wp.to_torch(self._succeeded_wp)
            self.metrics["success_rate"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

        # adds (optional) cmd kind and element names for leapp export
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/pose"
        self.cfg.element_names = self.cfg.element_names or POSE7_ELEMENT_NAMES

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> wp.array:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (x, y, z, w).
        """
        return self._pose_command_b_wp

    """
    Operations.
    """

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        env_mask: wp.array | None = None,
    ) -> dict[str, torch.Tensor]:
        """Log the ending episode's metrics, then reset the command of the selected environments.

        Args:
            env_ids: The specific environment indices to reset. If None, all environments are considered.
            env_mask: Boolean Warp mask of shape (num_envs,) selecting the environments to reset.
                Takes precedence over ``env_ids``.

        Returns:
            Persistent views of the metric means over the reset environments.
        """
        env_mask = self._env.resolve_env_mask(env_ids=env_ids, env_mask=env_mask)
        if self._track_success:
            wp.launch(
                kernel=_finalize_pose_success,
                dim=self.num_envs,
                inputs=[env_mask, self._succeeded_wp, self.metrics["success_rate"]],
                device=self.device,
            )
        return super().reset(env_mask=env_mask)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        position_threshold = self.cfg.position_success_threshold
        orientation_threshold = self.cfg.orientation_success_threshold
        wp.launch(
            kernel=_update_pose_metrics,
            dim=self.num_envs,
            inputs=[
                self._pose_command_b_wp,
                self.robot.data.root_pos_w.warp,
                self.robot.data.root_quat_w.warp,
                self.robot.data.body_pos_w.warp,
                self.robot.data.body_quat_w.warp,
                self.body_idx,
                position_threshold is not None,
                float(position_threshold or 0.0),
                orientation_threshold is not None,
                float(orientation_threshold or 0.0),
                self._pose_command_w_wp,
                self.metrics["position_error"],
                self.metrics["orientation_error"],
                self._succeeded_wp,
            ],
            device=self.device,
        )

    def _resample_command(self, env_mask: wp.array):
        ranges = self.cfg.ranges
        wp.launch(
            kernel=_resample_pose_command,
            dim=self.num_envs,
            inputs=[
                env_mask,
                self._env.rng_state_wp,
                wp.vec2f(*ranges.pos_x),
                wp.vec2f(*ranges.pos_y),
                wp.vec2f(*ranges.pos_z),
                wp.vec2f(*ranges.roll),
                wp.vec2f(*ranges.pitch),
                wp.vec2f(*ranges.yaw),
                self.cfg.make_quat_unique,
                self._pose_command_b_wp,
            ],
            device=self.device,
        )

    def _update_command(self):
        pass
