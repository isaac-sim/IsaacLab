# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the physical-grasp, two-cup Franka pour task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


def _canonical_pose(pose: torch.Tensor) -> torch.Tensor:
    """Return a finite pose with the quaternion mapped to its unique hemisphere."""
    pose = torch.nan_to_num(pose)
    return torch.cat((pose[:, :3], math_utils.quat_unique(pose[:, 3:7])), dim=-1)


def tcp_pose_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Tool-centre pose in the robot-base frame with a canonical XYZW quaternion."""
    return _canonical_pose(env.tcp_pose_e())


def ee_pose_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Backward-compatible hand-pose observation used by older diagnostic scripts."""
    return torch.nan_to_num(env.ee_pose_e())


def cup_pose_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Source-cup pose in the robot-base frame with a canonical XYZW quaternion."""
    return _canonical_pose(env.cup_pose_e())


def target_pose_obs(env: FrankaPourEnv) -> torch.Tensor:
    return _canonical_pose(env.target_pose_e())


def tcp_to_grasp_position_c_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Desired grasp point minus TCP position, expressed in the source-cup frame [m]."""
    cup_pose = env.cup_pose_e()
    tcp_position_c = math_utils.quat_apply_inverse(
        cup_pose[:, 3:7],
        env.tcp_pos_e() - cup_pose[:, :3],
    )
    desired_position_c = torch.zeros_like(tcp_position_c)
    desired_position_c[:, 2] = float(env.cfg.cup_grasp_height)
    return torch.nan_to_num(desired_position_c - tcp_position_c)


def grasp_to_tcp_quat_obs(env: FrankaPourEnv) -> torch.Tensor:
    """TCP orientation relative to the desired source-cup grasp frame as a canonical quaternion."""
    cup_quat = env.cup_pose_e()[:, 3:7]
    desired_quat = math_utils.quat_mul(cup_quat, env.desired_grasp_tcp_quat_c())
    tcp_quat = env.tcp_pose_e()[:, 3:7]
    error_quat = math_utils.quat_mul(math_utils.quat_conjugate(desired_quat), tcp_quat)
    return torch.nan_to_num(math_utils.quat_unique(error_quat))


def target_position_c_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Receiving-cup center expressed in the source-cup frame [m]."""
    cup_pose = env.cup_pose_e()
    target_offset = env.target_pose_e()[:, :3] - cup_pose[:, :3]
    return torch.nan_to_num(math_utils.quat_apply_inverse(cup_pose[:, 3:7], target_offset))


def tcp_to_grasp_obs(env: FrankaPourEnv) -> torch.Tensor:
    return torch.nan_to_num(env.cup_grasp_point_e() - env.tcp_pos_e())


def cup_to_target_obs(env: FrankaPourEnv) -> torch.Tensor:
    return torch.nan_to_num(env.target_pose_e()[:, :3] - env.cup_pose_e()[:, :3])


def gripper_width_obs(env: FrankaPourEnv) -> torch.Tensor:
    return torch.nan_to_num(env.gripper_width()).unsqueeze(-1)


def finger_position_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Individual finger-joint positions [m]."""
    return torch.nan_to_num(env.finger_joint_pos())


def finger_velocity_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Individual finger-joint velocities [m/s]."""
    return torch.nan_to_num(env.finger_joint_vel())


def arm_reference_phase_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Monotonic arm-reference progress in ``[0, 1]``."""
    action = env.action_manager.get_term("arm_action")
    phase = getattr(action, "reference_phase", None)
    if phase is None:
        phase = torch.zeros(env.num_envs, device=env.device)
    return torch.nan_to_num(phase).unsqueeze(-1)


def arm_reference_error_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Current applied trajectory target minus measured arm position [rad].

    The action term low-pass filters policy residuals, so its applied target contains persistent
    controller state that cannot be reconstructed from only the latest raw action. Exposing that
    target error keeps the policy observation Markov under randomized reset-bank references.
    """
    action = env.action_manager.get_term("arm_action")
    error = getattr(action, "reference_error", None)
    if error is None:
        error = action.processed_actions - env.arm_joint_pos()
    return torch.nan_to_num(error)


def trajectory_status_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Controller milestone, dwell, and capture state that affects future transitions."""
    arm = env.action_manager.get_term("arm_action")
    gripper = env.action_manager.get_term("gripper_action")
    arm_status = getattr(arm, "milestone_status", torch.zeros((env.num_envs, 6), device=env.device))
    capture_status = getattr(gripper, "capture_status", torch.ones((env.num_envs, 2), device=env.device))
    return torch.nan_to_num(torch.cat((arm_status, capture_status), dim=-1)).clamp_(0.0, 1.0)


def success_dwell_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Fraction of the stable-success dwell already satisfied."""
    dwell_steps = max(1, math.ceil(float(env.cfg.success_dwell_time_s) / max(float(env.step_dt), 1.0e-6)))
    return torch.clamp(env._success_dwell_count.float() / dwell_steps, 0.0, 1.0).unsqueeze(-1)


def time_remaining_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Normalized finite-horizon time remaining in ``[0, 1]``."""
    progress = env.episode_length_buf.float() / max(int(env.max_episode_length), 1)
    return torch.clamp(1.0 - progress, 0.0, 1.0).unsqueeze(-1)


def lost_grasp_dwell_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Fraction of the consecutive post-lift grasp-loss dwell already accumulated."""
    dwell_steps = max(
        1,
        math.ceil(float(env.cfg.lost_grasp_dwell_time_s) / max(float(env.step_dt), 1.0e-6)),
    )
    return torch.clamp(env._lost_grasp_dwell_count.float() / dwell_steps, 0.0, 1.0).unsqueeze(-1)


def pour_target_fraction_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Current episode's required held-delivery fraction in ``[0, 1]``."""
    return torch.nan_to_num(env.pour_target_frac).clamp_(0.0, 1.0).unsqueeze(-1)


def gripper_target_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Filtered symmetric per-finger target [m]."""
    return torch.nan_to_num(env.action_manager.get_term("gripper_action").commanded_position)


def gripper_contact_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Per-finger position-drive deflection caused by contact [m]."""
    gripper = env.action_manager.get_term("gripper_action")
    deflection = getattr(gripper, "contact_deflection", torch.zeros((env.num_envs, 2), device=env.device))
    return torch.nan_to_num(deflection)


def cup_velocity_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Source-cup world-frame linear and angular velocity [m/s, rad/s]."""
    return torch.nan_to_num(env.cup_velocity_w())


def particle_fractions_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Source, target, spill, and held-qualified target fractions."""
    scale = max(env.num_particles, 1)
    source = env.count_in_source() / scale
    target = env.count_in_target() / scale
    spilled = env.count_spilled() / scale
    held_target = env.current_held_delivered_mask().sum(dim=1).float() / scale
    return torch.stack((source, target, spilled, held_target), dim=-1)


def particle_transfer_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Airborne-media centroid/velocity relative to the receiver plus airborne fraction.

    Counts alone cannot tell the privileged critic whether a stream is moving toward or past the
    receiver. This compact summary supplies that value-estimation signal without exposing
    per-particle state to the actor.
    """
    source, target, spilled = env.particle_region_masks()
    airborne = ~(source | target | spilled)
    weight = airborne.unsqueeze(-1).to(dtype=torch.float32)
    count = weight.sum(dim=1)
    denominator = count.clamp_min(1.0)
    centroid = (env.particle_pos_e() * weight).sum(dim=1) / denominator
    velocity = (env.particle_vel_e() * weight).sum(dim=1) / denominator
    receiver = env.target_pose_e()[:, :3]
    centroid_relative = torch.where(count > 0.0, centroid - receiver, torch.zeros_like(centroid))
    velocity = torch.where(count > 0.0, velocity, torch.zeros_like(velocity))
    fraction = count / max(env.num_particles, 1)
    summary = torch.cat((centroid_relative / 0.30, velocity / 2.0, fraction), dim=-1)
    return torch.nan_to_num(summary).clamp_(-2.0, 2.0)


def held_delivery_history_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Fraction of particles that have entered the receiver during a qualified held pour."""
    fraction = env.held_delivered_mask().sum(dim=1).float() / max(env.num_particles, 1)
    return torch.nan_to_num(fraction).clamp_(0.0, 1.0).unsqueeze(-1)
