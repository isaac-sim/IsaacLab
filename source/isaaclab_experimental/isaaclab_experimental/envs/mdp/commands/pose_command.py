# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native pose command generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands._debug_vis import _PoseCommandDebugVis
from isaaclab.utils.leapp import POSE7_ELEMENT_NAMES

from isaaclab_experimental.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformPoseCommandCfg


@wp.kernel
def _resample_pose_command(
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    command: wp.array(dtype=wp.float32, ndim=2),
    pos_x_min: float,
    pos_x_max: float,
    pos_y_min: float,
    pos_y_max: float,
    pos_z_min: float,
    pos_z_max: float,
    roll_min: float,
    roll_max: float,
    pitch_min: float,
    pitch_max: float,
    yaw_min: float,
    yaw_max: float,
    make_quat_unique: bool,
):
    """Draw a new body-frame pose command for selected envs: uniform position [m]

    and roll/pitch/yaw [rad] composed into a quaternion (optionally
    sign-canonicalized to the w >= 0 hemisphere). RNG state is written back so
    the sequence advances across replays.
    """
    env_id = wp.tid()
    if env_mask[env_id]:
        state = rng_state[env_id]
        command[env_id, 0] = wp.randf(state, pos_x_min, pos_x_max)
        command[env_id, 1] = wp.randf(state, pos_y_min, pos_y_max)
        command[env_id, 2] = wp.randf(state, pos_z_min, pos_z_max)

        roll = wp.randf(state, roll_min, roll_max)
        pitch = wp.randf(state, pitch_min, pitch_max)
        yaw = wp.randf(state, yaw_min, yaw_max)
        qx = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), roll)
        qy = wp.quat_from_axis_angle(wp.vec3f(0.0, 1.0, 0.0), pitch)
        qz = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), yaw)
        quat = wp.mul(wp.mul(qz, qy), qx)
        if make_quat_unique and quat[3] < 0.0:
            quat = wp.quatf(-quat[0], -quat[1], -quat[2], -quat[3])
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
    command_w: wp.array(dtype=wp.float32, ndim=2),
    position_error: wp.array(dtype=wp.float32),
    orientation_error: wp.array(dtype=wp.float32),
    success_rate: wp.array(dtype=wp.float32),
    track_success: bool,
    position_success_threshold: float,
):
    """Refresh the world-frame goal pose (root pose composed with the body-frame

    command) and the tracking metrics: end-effector position [m] / orientation
    [rad] errors vs the goal, and a position-threshold success flag when success
    tracking is enabled.
    """
    env_id = wp.tid()
    position_b = wp.vec3f(command_b[env_id, 0], command_b[env_id, 1], command_b[env_id, 2])
    orientation_b = wp.quatf(command_b[env_id, 3], command_b[env_id, 4], command_b[env_id, 5], command_b[env_id, 6])
    desired_position_w = root_pos_w[env_id] + wp.quat_rotate(root_quat_w[env_id], position_b)
    desired_orientation_w = root_quat_w[env_id] * orientation_b

    command_w[env_id, 0] = desired_position_w[0]
    command_w[env_id, 1] = desired_position_w[1]
    command_w[env_id, 2] = desired_position_w[2]
    command_w[env_id, 3] = desired_orientation_w[0]
    command_w[env_id, 4] = desired_orientation_w[1]
    command_w[env_id, 5] = desired_orientation_w[2]
    command_w[env_id, 6] = desired_orientation_w[3]

    position_delta = body_pos_w[env_id, body_idx] - desired_position_w
    position_error_value = wp.length(position_delta)
    orientation_delta = body_quat_w[env_id, body_idx] * wp.quat_inverse(desired_orientation_w)
    orientation_error_value = 2.0 * wp.acos(wp.clamp(wp.abs(orientation_delta[3]), 0.0, 1.0))
    position_error[env_id] = position_error_value
    orientation_error[env_id] = orientation_error_value
    if track_success and position_error_value < position_success_threshold:
        success_rate[env_id] = 1.0


class UniformPoseCommand(_PoseCommandDebugVis, CommandTerm):
    """Generate a pose command from uniform position and Euler-angle distributions."""

    cfg: UniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: Configuration for the command generator.
            env: Environment containing the commanded articulation.
        """
        super().__init__(cfg, env)

        # -- robot / body resolution
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # -- command buffers (Warp-native, pointer-stable)
        self.pose_command_b = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)
        self.pose_command_w = wp.zeros((self.num_envs, 7), dtype=wp.float32, device=self.device)

        # -- metrics buffers
        self.metrics["position_error"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self.metrics["orientation_error"] = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._success_rate = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._track_success = cfg.position_success_threshold is not None
        if self._track_success:
            self.metrics["success_rate"] = self._success_rate

        # adds (optional) cmd kind and element names for leapp export
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/pose"
        self.cfg.element_names = self.cfg.element_names or POSE7_ELEMENT_NAMES

        # -- Torch views (zero-copy aliases for stable consumers)
        self.pose_command_b_torch = wp.to_torch(self.pose_command_b)
        self.pose_command_w_torch = wp.to_torch(self.pose_command_w)
        self.pose_command_b_torch[:, 6] = 1.0

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Desired pose command [m, m, m, quaternion xyzw], shape ``(num_envs, 7)``."""
        return self.pose_command_b_torch

    @property
    def command_wp(self) -> wp.array(dtype=wp.float32, ndim=2):
        """Pointer-stable desired pose command [m, m, m, quaternion xyzw]."""
        return self.pose_command_b

    def _update_metrics(self):
        wp.launch(
            kernel=_update_pose_metrics,
            dim=self.num_envs,
            inputs=[
                self.pose_command_b,
                self.robot.data.root_pos_w.warp,
                self.robot.data.root_quat_w.warp,
                self.robot.data.body_pos_w.warp,
                self.robot.data.body_quat_w.warp,
                self.body_idx,
                self.pose_command_w,
                self.metrics["position_error"],
                self.metrics["orientation_error"],
                self._success_rate,
                self._track_success,
                self.cfg.position_success_threshold or 0.0,
            ],
            device=self.device,
        )

    def _resample_command(self, env_mask: wp.array):
        wp.launch(
            kernel=_resample_pose_command,
            dim=self.num_envs,
            inputs=[
                env_mask,
                self._env.rng_state_wp,
                self.pose_command_b,
                self.cfg.ranges.pos_x[0],
                self.cfg.ranges.pos_x[1],
                self.cfg.ranges.pos_y[0],
                self.cfg.ranges.pos_y[1],
                self.cfg.ranges.pos_z[0],
                self.cfg.ranges.pos_z[1],
                self.cfg.ranges.roll[0],
                self.cfg.ranges.roll[1],
                self.cfg.ranges.pitch[0],
                self.cfg.ranges.pitch[1],
                self.cfg.ranges.yaw[0],
                self.cfg.ranges.yaw[1],
                self.cfg.make_quat_unique,
            ],
            device=self.device,
        )

    def _update_command(self):
        pass

    def _debug_pose_command_w(self) -> torch.Tensor:
        """Return the Torch alias of the pointer-stable world-frame pose command buffer."""
        return self.pose_command_w_torch
