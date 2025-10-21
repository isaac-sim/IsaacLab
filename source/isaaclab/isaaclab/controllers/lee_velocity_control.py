# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Multirotor
    from .lee_velocity_control_cfg import LeeVelControllerCfg


class LeeVelController:
    """Implementation of a Lee-style geometric controller (SE(3)).

    Computes a body-frame wrench command ``[Fx, Fy, Fz, Tx, Ty, Tz]`` from a compact action:
    collective thrust scale, roll command, pitch command, and yaw-rate command.
    Gains may be randomized per environment if enabled in the configuration.
    """

    cfg: LeeVelControllerCfg

    def __init__(self, cfg: LeeVelControllerCfg, asset: Multirotor, num_envs: int, device: str):
        """Initialize controller buffers and pre-compute aggregate inertias.

        Args:
            cfg: Controller configuration.
            env: Owning environment (must provide ``scene['robot']``, ``num_envs``, and ``device``).
        """
        self.cfg = cfg
        self.robot: Multirotor = asset

        # Aggregate mass and inertia about the robot COM for all bodies
        root_quat_exp = self.robot.data.root_link_quat_w.unsqueeze(1).expand(num_envs, self.robot.num_bodies, 4)
        body_link_pos_delta = self.robot.data.body_link_pos_w - self.robot.data.root_pos_w.unsqueeze(1)
        self.mass, self.robot_inertia, _ = math_utils.aggregate_inertia_about_robot_com(
            self.robot.root_physx_view.get_inertias().to(device),
            self.robot.root_physx_view.get_inv_masses().to(device),
            self.robot.data.body_com_pos_b,
            self.robot.data.body_com_quat_b,
            math_utils.quat_apply_inverse(root_quat_exp, body_link_pos_delta),
            math_utils.quat_mul(math_utils.quat_inv(root_quat_exp), self.robot.data.body_link_quat_w),
        )
        self.gravity = torch.tensor(self.cfg.gravity, device=device).expand(num_envs, -1)

        # Gain ranges (single tensor each): shape (num_envs, 2, 3); [:,0]=min, [:,1]=max
        self.K_vel_range = torch.tensor(self.cfg.K_vel_range, device=device).repeat(num_envs, 1, 1)
        self.K_rot_range = torch.tensor(self.cfg.K_rot_range, device=device).repeat(num_envs, 1, 1)
        self.K_angvel_range = torch.tensor(self.cfg.K_angvel_range, device=device).repeat(num_envs, 1, 1)

        # Current (possibly randomized) gains
        self.K_vel_current = self.K_vel_range.mean(dim=1)
        self.K_rot_current = self.K_rot_range.mean(dim=1)
        self.K_angvel_current = self.K_angvel_range.mean(dim=1)

        # Buffers (all shapes use num_envs in the first dimension)
        self.accel = torch.zeros((num_envs, 3), device=device)
        self.wrench_command_b = torch.zeros((num_envs, 6), device=device)  # [fx, fy, fz, tx, ty, tz]
        self.desired_body_angvel_w = torch.zeros_like(self.robot.data.root_ang_vel_b)
        self.desired_quat = torch.zeros_like(self.robot.data.root_quat_w)
        self.euler_angle_rates_w = torch.zeros_like(self.robot.data.root_ang_vel_b)
        self.buffer_tensor = torch.zeros((num_envs, 3, 3), device=device)
    
    def compute(self, command):
        
        self.wrench_command_b[:] = 0.0
        acc = self.compute_acceleration(setpoint_velocity=command[:, 0:3])
        forces = (acc - self.gravity) * self.mass.view(-1,1)
        self.wrench_command_b[:, 2] = torch.sum(
            forces * math_utils.matrix_from_quat(self.robot.data.root_quat_w)[:, :, 2], dim=1
        )
        
        robot_euler_w = torch.stack(math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w), dim=-1)
        robot_euler_w = math_utils.wrap_to_pi(robot_euler_w) 
        
        self.desired_quat = calculate_desired_orientation_for_position_velocity_control(forces, robot_euler_w[:, 2],self.buffer_tensor)
        self.euler_angle_rates_w[:, :2] = 0.0
        self.euler_angle_rates_w[:, 2] = command[:, 3]
        self.desired_body_angvel_w[:] = euler_to_body_rate(robot_euler_w, self.euler_angle_rates_w, self.buffer_tensor)
        self.wrench_command_b[:, 3:6] = self.compute_body_torque(self.desired_quat, self.desired_body_angvel_w)
        
        return self.wrench_command_b
        
    def reset(self):
        """Reset controller state for all environments."""
        self.reset_idx(env_ids=None)

    def reset_idx(self, env_ids: torch.Tensor | None):
        """Reset controller state (and optionally randomize gains) for selected environments.

        Args:
            env_ids: Tensor of environment indices, or ``None`` for all.
        """
        if env_ids is None:
            env_ids = slice(None)
        self.randomize_params(env_ids)

    def randomize_params(self, env_ids: slice | torch.Tensor):
        """Randomize controller gains for the given environments if enabled."""
        if not self.cfg.randomize_params:
            return
        self.K_vel_current[env_ids] = math_utils.rand_range(self.K_vel_range[env_ids, 0], self.K_vel_range[env_ids, 1])
        self.K_rot_current[env_ids] = math_utils.rand_range(self.K_rot_range[env_ids, 0], self.K_rot_range[env_ids, 1])
        self.K_angvel_current[env_ids] = math_utils.rand_range(self.K_angvel_range[env_ids, 0], self.K_angvel_range[env_ids, 1])

    def compute_acceleration(self, setpoint_velocity):
        robot_vehicle_orientation = vehicle_frame_quat_from_quat(self.robot.data.root_quat_w)
        setpoint_velocity_world_frame = math_utils.quat_apply(
            robot_vehicle_orientation, setpoint_velocity
        )
        velocity_error = setpoint_velocity_world_frame - self.robot.data.root_lin_vel_w

        accel_command = (
            self.K_vel_current * velocity_error
        )
        return accel_command

    def compute_body_torque(self, setpoint_orientation, setpoint_angvel):
        """PD attitude control in body frame with feedforward Coriolis term.

        Args:
            setpoint_orientation: (num_envs, 4) desired orientation quaternion (wxyz) in world frame.
            setpoint_angvel: (num_envs, 3) desired angular velocity in world frame [rad/s].

        Returns:
            (num_envs, 3) body torque command [N·m].
        """
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

def calculate_desired_orientation_for_position_velocity_control(
    forces_command, yaw_setpoint, rotation_matrix_desired
):
    b3_c = torch.div(forces_command, torch.norm(forces_command, dim=1).unsqueeze(1))
    temp_dir = torch.zeros_like(forces_command)
    temp_dir[:, 0] = torch.cos(yaw_setpoint)
    temp_dir[:, 1] = torch.sin(yaw_setpoint)

    b2_c = torch.cross(b3_c, temp_dir, dim=1)
    b2_c = torch.div(b2_c, torch.norm(b2_c, dim=1).unsqueeze(1))
    b1_c = torch.cross(b2_c, b3_c, dim=1)

    rotation_matrix_desired[:, :, 0] = b1_c
    rotation_matrix_desired[:, :, 1] = b2_c
    rotation_matrix_desired[:, :, 2] = b3_c
    q = math_utils.quat_from_matrix(rotation_matrix_desired)
    quat_desired = torch.stack((q[:, 0], q[:, 1], q[:, 2], q[:, 3]), dim=1)

    return quat_desired

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

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz_tensor(q):
    qx, qy, qz, qw = 1, 2, 3, 0
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(torch.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack(
        [roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)], dim=-1
    )

@torch.jit.script
def quat_from_euler_xyz_tensor(euler_xyz_tensor: torch.Tensor) -> torch.Tensor:
    roll = euler_xyz_tensor[..., 0]
    pitch = euler_xyz_tensor[..., 1]
    yaw = euler_xyz_tensor[..., 2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def vehicle_frame_quat_from_quat(body_quat: torch.Tensor) -> torch.Tensor:
    body_euler = get_euler_xyz_tensor(body_quat) * torch.tensor(
        [0.0, 0.0, 1.0], device=body_quat.device
    )
    return quat_from_euler_xyz_tensor(body_euler)