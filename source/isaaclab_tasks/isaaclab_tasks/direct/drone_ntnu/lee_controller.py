# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

from .utils import aggregate_inertia_about_robot_com, torch_rand_float_tensor

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from .controller_cfg import LeeControllerCfg


class BaseLeeController:
    """
    This class will operate as the base class for all controllers.
    It will be inherited by the specific controller classes.
    """

    cfg: LeeControllerCfg

    def __init__(self, cfg: LeeControllerCfg, env):
        self.cfg = cfg
        self.env = env
        self.robot: Articulation = env.scene["robot"]
        root_quat_exp = self.robot.data.root_link_quat_w.unsqueeze(1).expand(env.num_envs, self.robot.num_bodies, 4)
        body_link_pos_delta = self.robot.data.body_link_pos_w - self.robot.data.root_pos_w.unsqueeze(1)
        self.mass, self.robot_inertia, _ = aggregate_inertia_about_robot_com(
            self.robot.root_physx_view.get_inertias().to(env.device),
            self.robot.root_physx_view.get_inv_masses().to(env.device),
            self.robot.data.body_com_pos_b,
            self.robot.data.body_com_quat_b,
            math_utils.quat_apply_inverse(root_quat_exp, body_link_pos_delta),
            math_utils.quat_mul(math_utils.quat_inv(root_quat_exp), self.robot.data.body_link_quat_w),
        )
        self.gravity = torch.tensor(self.cfg.gravity, device=self.env.device).expand(self.env.num_envs, -1)

        # Read from config and set the values for controller parameters
        self.K_pos_max = torch.tensor(self.cfg.K_pos_max, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_pos_min = torch.tensor(self.cfg.K_pos_min, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_linvel_max = torch.tensor(self.cfg.K_vel_max, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_linvel_min = torch.tensor(self.cfg.K_vel_min, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_rot_max = torch.tensor(self.cfg.K_rot_max, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_rot_min = torch.tensor(self.cfg.K_rot_min, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_angvel_max = torch.tensor(self.cfg.K_angvel_max, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_angvel_min = torch.tensor(self.cfg.K_angvel_min, device=self.env.device).expand(self.env.num_envs, -1)

        # Set the current values of the controller parameters
        self.K_pos_current = (self.K_pos_max + self.K_pos_min) / 2.0
        self.K_linvel_current = (self.K_linvel_max + self.K_linvel_min) / 2.0
        self.K_rot_current = (self.K_rot_max + self.K_rot_min) / 2.0
        self.K_angvel_current = (self.K_angvel_max + self.K_angvel_min) / 2.0

        # tensors that are needed later in the controller are predefined here
        self.accel = torch.zeros((self.env.num_envs, 3), device=self.env.device)
        self.wrench_command_b = torch.zeros((self.env.num_envs, 6), device=self.env.device)  # [fx, fy, fz, tx, ty, tz]
        # tensors that are needed later in the controller are predefined here
        self.desired_body_angvel_w = torch.zeros_like(self.robot.data.root_ang_vel_b)
        self.euler_angle_rates = torch.zeros_like(self.robot.data.root_ang_vel_b)

        # buffer tensor to be used by torch.jit functions for various purposes
        self.buffer_tensor = torch.zeros((self.env.num_envs, 3, 3), device=self.env.device)

    def __call__(self, command):
        robot_euler_w = torch.stack(math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w), dim=-1)
        robot_euler_w = math_utils.wrap_to_pi(robot_euler_w)
        self.wrench_command_b[:] = 0.0
        self.wrench_command_b[:, 2] = (command[:, 0] + 1.0) * self.mass * torch.norm(self.gravity, dim=1)
        self.euler_angle_rates[:, :2] = 0.0
        self.euler_angle_rates[:, 2] = command[:, 3]
        self.desired_body_angvel_w[:] = euler_to_body_rate(robot_euler_w, self.euler_angle_rates, self.buffer_tensor)

        # quaternion desired
        # desired euler angle is equal to commanded roll, commanded pitch, and current yaw
        quat_w_desired = math_utils.quat_from_euler_xyz(command[:, 1], command[:, 2], robot_euler_w[:, 2])
        self.wrench_command_b[:, 3:6] = self.compute_body_torque(quat_w_desired, self.desired_body_angvel_w)

        return self.wrench_command_b

    def reset(self):
        self.reset_idx(env_ids=None)

    def reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs)
        self.randomize_params(env_ids)

    def randomize_params(self, env_ids):
        if self.cfg.randomize_params == False:
            return
        self.K_pos_current[env_ids] = torch_rand_float_tensor(self.K_pos_min[env_ids], self.K_pos_max[env_ids])
        self.K_linvel_current[env_ids] = torch_rand_float_tensor(self.K_linvel_min[env_ids], self.K_linvel_max[env_ids])
        self.K_rot_current[env_ids] = torch_rand_float_tensor(self.K_rot_min[env_ids], self.K_rot_max[env_ids])
        self.K_angvel_current[env_ids] = torch_rand_float_tensor(self.K_angvel_min[env_ids], self.K_angvel_max[env_ids])

    def compute_acceleration(self, setpoint_position, setpoint_velocity):
        position_error_world_frame = setpoint_position - self.robot.data.root_pos_w
        setpoint_velocity_world_frame = math_utils.quat_apply(
            math_utils.yaw_quat(self.robot.data.root_quat_w), setpoint_velocity
        )
        velocity_error = setpoint_velocity_world_frame - self.robot.data.root_lin_vel_w
        accel_command = self.K_pos_current * position_error_world_frame + self.K_linvel_current * velocity_error
        return accel_command

    def compute_body_torque(self, setpoint_orientation, setpoint_angvel):
        setpoint_angvel[:, 2] = torch.clamp(setpoint_angvel[:, 2], -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        RT_Rd_quat = math_utils.quat_mul(
            math_utils.quat_inv(self.robot.data.root_quat_w), setpoint_orientation
        )  # (N,4) wxyz
        R_err = math_utils.matrix_from_quat(RT_Rd_quat)
        skew_matrix = R_err.transpose(-1, -2) - R_err
        rotation_error = 0.5 * torch.stack([-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
        angvel_error = self.robot.data.root_ang_vel_b - math_utils.quat_apply(RT_Rd_quat, setpoint_angvel)
        feed_forward_body_rates = torch.cross(
            self.robot.data.root_ang_vel_b,
            torch.bmm(self.robot_inertia, self.robot.data.root_ang_vel_b.unsqueeze(2)).squeeze(2),
            dim=1,
        )
        torque = -self.K_rot_current * rotation_error - self.K_angvel_current * angvel_error + feed_forward_body_rates
        return torque


@torch.jit.script
def euler_to_body_rate(curr_euler_rate, des_euler_rate, mat_euler_to_body_rate):
    s_pitch = torch.sin(curr_euler_rate[:, 1])
    c_pitch = torch.cos(curr_euler_rate[:, 1])

    s_roll = torch.sin(curr_euler_rate[:, 0])
    c_roll = torch.cos(curr_euler_rate[:, 0])

    mat_euler_to_body_rate[:, 0, 0] = 1.0
    mat_euler_to_body_rate[:, 1, 1] = c_roll
    mat_euler_to_body_rate[:, 0, 2] = -s_pitch
    mat_euler_to_body_rate[:, 2, 1] = -s_roll
    mat_euler_to_body_rate[:, 1, 2] = s_roll * c_pitch
    mat_euler_to_body_rate[:, 2, 2] = c_roll * c_pitch

    return torch.bmm(mat_euler_to_body_rate, des_euler_rate.unsqueeze(2)).squeeze(2)
