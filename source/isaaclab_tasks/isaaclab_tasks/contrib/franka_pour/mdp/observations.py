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


def _nearest_grasp_to_tcp_quat(env: FrankaPourEnv) -> torch.Tensor:
    """Return TCP error from the nearest of four equivalent cup-side grasp frames.

    The source cup is rotationally symmetric for grasping from its four horizontal sides. The
    offline reset generator deliberately balances those sides, so comparing the TCP only with the
    authored side-zero frame would report a 90, 180, or 270 degree error for an equally valid
    grasp. Selecting the nearest canonical frame keeps the observation and physical shaping
    invariant to that discrete symmetry without exposing the reset row's ``grasp_side`` label.
    """
    cup_quat = env.cup_pose_e()[:, 3:7]
    tcp_quat = env.tcp_pose_e()[:, 3:7]
    base_grasp = env.desired_grasp_tcp_quat_c()
    if base_grasp.ndim == 1:
        base_grasp = base_grasp.unsqueeze(0)
    if base_grasp.shape[0] == 1 and cup_quat.shape[0] != 1:
        base_grasp = base_grasp.expand(cup_quat.shape[0], -1)
    if base_grasp.shape != cup_quat.shape:
        raise ValueError(
            "desired_grasp_tcp_quat_c must return one XYZW quaternion per environment "
            f"or one broadcast quaternion, got {tuple(base_grasp.shape)} for {cup_quat.shape[0]} environments."
        )

    side_angles = torch.arange(4, device=cup_quat.device, dtype=cup_quat.dtype) * (0.5 * math.pi)
    side_axes = torch.zeros((4, 3), device=cup_quat.device, dtype=cup_quat.dtype)
    side_axes[:, 2] = 1.0
    side_rotations = math_utils.quat_from_angle_axis(side_angles, side_axes)
    side_rotations = side_rotations.unsqueeze(0).expand(cup_quat.shape[0], -1, -1)
    base_grasps = base_grasp.unsqueeze(1).expand_as(side_rotations)
    cup_quaternions = cup_quat.unsqueeze(1).expand_as(side_rotations)
    tcp_quaternions = tcp_quat.unsqueeze(1).expand_as(side_rotations)
    desired_quaternions = math_utils.quat_mul(
        cup_quaternions,
        math_utils.quat_mul(side_rotations, base_grasps),
    )
    error_quaternions = math_utils.quat_unique(
        math_utils.quat_mul(math_utils.quat_conjugate(desired_quaternions), tcp_quaternions)
    )
    error_magnitudes = torch.linalg.vector_norm(
        math_utils.axis_angle_from_quat(error_quaternions),
        dim=-1,
    )
    error_magnitudes = torch.nan_to_num(error_magnitudes, nan=torch.inf, posinf=torch.inf, neginf=torch.inf)
    nearest_side = torch.argmin(error_magnitudes, dim=1)
    nearest_error = torch.gather(
        error_quaternions,
        dim=1,
        index=nearest_side[:, None, None].expand(-1, 1, 4),
    ).squeeze(1)
    return torch.nan_to_num(math_utils.quat_unique(nearest_error))


def grasp_to_tcp_quat_obs(env: FrankaPourEnv) -> torch.Tensor:
    """TCP orientation relative to the nearest equivalent source-cup grasp frame."""
    return _nearest_grasp_to_tcp_quat(env)


def target_position_c_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Receiving-cup center expressed in the source-cup frame [m]."""
    cup_pose = env.cup_pose_e()
    target_offset = env.target_pose_e()[:, :3] - cup_pose[:, :3]
    return torch.nan_to_num(math_utils.quat_apply_inverse(cup_pose[:, 3:7], target_offset))


def finger_position_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Individual finger-joint positions [m]."""
    return torch.nan_to_num(env.finger_joint_pos())


def finger_velocity_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Individual finger-joint velocities [m/s]."""
    return torch.nan_to_num(env.finger_joint_vel())


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
    return torch.nan_to_num(gripper.contact_deflection)


def normalized_cup_velocity_obs(
    env: FrankaPourEnv,
    max_surface_speed: float = 0.5,
    surface_radius: float = 0.13,
) -> torch.Tensor:
    """Cup velocity normalized by a conservative maximum surface speed.

    Linear velocity is divided directly by ``max_surface_speed``. Angular velocity is first
    converted to tangential velocity at ``surface_radius``, so the sum of the two vector norms is
    the cup's normalized conservative surface speed. Non-finite simulation state is handled by the
    failure terms and is kept finite here so PPO can consume the terminal observation.
    """
    for name, value in (("max_surface_speed", max_surface_speed), ("surface_radius", surface_radius)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    velocity = torch.nan_to_num(env.cup_velocity_w(), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.cat(
        (
            velocity[:, :3] / float(max_surface_speed),
            velocity[:, 3:] * (float(surface_radius) / float(max_surface_speed)),
        ),
        dim=-1,
    )


def particle_fractions_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Source, target, spill, and legacy held-delivery fractions.

    The final checkpoint-facing channel is retained as zero for compatibility. The original held-
    delivery tracker was never updated by the task and therefore produced the same value.
    """
    scale = max(env.num_particles, 1)
    source = env.count_in_source() / scale
    target = env.count_in_target() / scale
    spilled = env.count_spilled() / scale
    held_target = torch.zeros_like(source)
    return torch.stack((source, target, spilled, held_target), dim=-1)


def particle_source_state_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Source-contained media centroid and relative velocity in the source-cup frame.

    The centroid is relative to the cup origin and normalized by 0.12 m. The velocity subtracts
    the rigid velocity of the corresponding cup point before rotating into the cup frame, then is
    normalized by 2 m/s. Both vectors are zero when the source cup contains no particles.
    """
    source = env.particle_region_masks()[0]
    weight = source.unsqueeze(-1).to(dtype=torch.float32)
    count = weight.sum(dim=1)
    denominator = count.clamp_min(1.0)
    centroid = (env.particle_pos_e() * weight).sum(dim=1) / denominator
    velocity = (env.particle_vel_e() * weight).sum(dim=1) / denominator

    cup_pose = env.cup_pose_e()
    centroid_offset = centroid - cup_pose[:, :3]
    cup_velocity = env.cup_velocity_w()
    rigid_point_velocity = cup_velocity[:, :3] + torch.linalg.cross(cup_velocity[:, 3:], centroid_offset, dim=-1)
    centroid_c = math_utils.quat_apply_inverse(cup_pose[:, 3:7], centroid_offset)
    velocity_c = math_utils.quat_apply_inverse(
        cup_pose[:, 3:7],
        velocity - rigid_point_velocity,
    )
    present = count > 0.0
    summary = torch.cat((centroid_c / 0.12, velocity_c / 2.0), dim=-1)
    summary = torch.where(present.expand_as(summary), summary, torch.zeros_like(summary))
    return torch.nan_to_num(summary).clamp_(-2.0, 2.0)


def particle_transfer_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Airborne-media centroid/velocity in the receiver frame plus airborne fraction.

    Counts alone cannot tell the privileged critic whether a stream is moving toward or past the
    receiver. This compact, permutation-invariant summary exposes that flow without requiring an
    ordered particle point cloud.
    """
    source, target, spilled = env.particle_region_masks()
    airborne = ~(source | target | spilled)
    weight = airborne.unsqueeze(-1).to(dtype=torch.float32)
    count = weight.sum(dim=1)
    denominator = count.clamp_min(1.0)
    centroid = (env.particle_pos_e() * weight).sum(dim=1) / denominator
    velocity = (env.particle_vel_e() * weight).sum(dim=1) / denominator
    receiver_pose = env.target_pose_e()
    centroid_relative = math_utils.quat_apply_inverse(
        receiver_pose[:, 3:7],
        centroid - receiver_pose[:, :3],
    )
    velocity = math_utils.quat_apply_inverse(receiver_pose[:, 3:7], velocity)
    centroid_relative = torch.where(count > 0.0, centroid_relative, torch.zeros_like(centroid))
    velocity = torch.where(count > 0.0, velocity, torch.zeros_like(velocity))
    fraction = count / max(env.num_particles, 1)
    summary = torch.cat((centroid_relative / 0.30, velocity / 2.0, fraction), dim=-1)
    return torch.nan_to_num(summary).clamp_(-2.0, 2.0)


def held_delivery_history_obs(env: FrankaPourEnv) -> torch.Tensor:
    """Return the legacy zero held-delivery channel retained for checkpoint compatibility."""
    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
