# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the pour task.

In addition to the standard timeout, instability guards reset non-finite states and finite media
that leave the task workspace before they can expand the allocating sparse grid without bound.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


_GRIPPER_POSITION_TOLERANCE = 1.0e-6


def _oriented_boxes_overlap(
    center_a: torch.Tensor,
    quaternion_a: torch.Tensor,
    half_extents_a: Sequence[float],
    center_b: torch.Tensor,
    quaternion_b: torch.Tensor,
    half_extents_b: Sequence[float],
    *,
    clearance: float = 0.0,
) -> torch.Tensor:
    """Return batched OBB intersection using the complete 15-axis separating-axis test."""
    if center_a.ndim != 2 or center_a.shape[1] != 3 or center_b.shape != center_a.shape:
        raise ValueError("Both center tensors must have shape (N, 3).")
    count = center_a.shape[0]
    if quaternion_a.shape != (count, 4) or quaternion_b.shape != (count, 4):
        raise ValueError("Both quaternion tensors must have shape (N, 4).")
    if clearance < 0.0:
        raise ValueError("clearance must be nonnegative.")

    rotation_a = math_utils.matrix_from_quat(quaternion_a)
    rotation_b = math_utils.matrix_from_quat(quaternion_b)
    relative_rotation = rotation_a.transpose(-1, -2) @ rotation_b
    absolute_rotation = relative_rotation.abs() + 1.0e-6
    translation = (rotation_a.transpose(-1, -2) @ (center_b - center_a).unsqueeze(-1)).squeeze(-1)
    half_a = torch.as_tensor(half_extents_a, device=center_a.device, dtype=center_a.dtype) + clearance * 0.5
    half_b = torch.as_tensor(half_extents_b, device=center_a.device, dtype=center_a.dtype) + clearance * 0.5

    separated = torch.zeros(count, device=center_a.device, dtype=torch.bool)
    for axis in range(3):
        radius_b = (absolute_rotation[:, axis, :] * half_b).sum(dim=-1)
        separated |= translation[:, axis].abs() > half_a[axis] + radius_b
    for axis in range(3):
        projection = (translation * relative_rotation[:, :, axis]).sum(dim=-1).abs()
        radius_a = (absolute_rotation[:, :, axis] * half_a).sum(dim=-1)
        separated |= projection > radius_a + half_b[axis]
    for axis_a in range(3):
        for axis_b in range(3):
            other_a = (axis_a + 1) % 3
            last_a = (axis_a + 2) % 3
            other_b = (axis_b + 1) % 3
            last_b = (axis_b + 2) % 3
            projection = (
                translation[:, last_a] * relative_rotation[:, other_a, axis_b]
                - translation[:, other_a] * relative_rotation[:, last_a, axis_b]
            ).abs()
            radius_a = (
                half_a[other_a] * absolute_rotation[:, last_a, axis_b]
                + half_a[last_a] * absolute_rotation[:, other_a, axis_b]
            )
            radius_b = (
                half_b[other_b] * absolute_rotation[:, axis_a, last_b]
                + half_b[last_b] * absolute_rotation[:, axis_a, other_b]
            )
            separated |= projection > radius_a + radius_b
    return ~separated


def nonfinite_failure(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate on non-finite simulation state (instability guard)."""
    return ~env.state_finite()


def extreme_rigid_state(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate finite rigid states before extreme values reach observation normalization."""
    return ~env.rigid_state_in_bounds()


def _source_receiver_envelope_overlap(
    source_pose: torch.Tensor,
    target_pose: torch.Tensor,
    *,
    source_half_extents: Sequence[float],
    target_half_extents: Sequence[float],
    clearance: float = 0.0,
) -> torch.Tensor:
    """Return conservative overlap between source and receiver outer envelopes.

    Both body origins lie at their outer base. The source box is projected into the receiver frame,
    then tested against the receiver's complete outer envelope. Treating the open cavity as part of
    that forbidden envelope deliberately prevents the policy from nesting the loaded source cup in
    the receiver instead of pouring from above. It remains rotation-aware and allows arbitrary
    receiver yaw.
    """
    if source_pose.ndim != 2 or source_pose.shape[1] != 7 or target_pose.shape != source_pose.shape:
        raise ValueError("source_pose and target_pose must have matching shape (N, 7).")
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance must be finite and nonnegative.")
    if len(source_half_extents) != 3 or len(target_half_extents) != 3:
        raise ValueError("source and target half extents must each contain three values.")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in source_half_extents):
        raise ValueError("source half extents must be finite and positive.")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in target_half_extents):
        raise ValueError("target half extents must be finite and positive.")

    source_half = source_pose.new_tensor(source_half_extents)
    target_half = target_pose.new_tensor(target_half_extents)
    source_center_offset = torch.zeros((source_pose.shape[0], 3), device=source_pose.device, dtype=source_pose.dtype)
    source_center_offset[:, 2] = source_half[2]
    target_center_offset = torch.zeros_like(source_center_offset)
    target_center_offset[:, 2] = target_half[2]
    source_center = source_pose[:, :3] + math_utils.quat_apply(source_pose[:, 3:7], source_center_offset)
    target_center = target_pose[:, :3] + math_utils.quat_apply(target_pose[:, 3:7], target_center_offset)
    return _oriented_boxes_overlap(
        source_center,
        source_pose[:, 3:7],
        source_half_extents,
        target_center,
        target_pose[:, 3:7],
        target_half_extents,
        clearance=clearance,
    )


def source_receiver_overlap(env: FrankaPourEnv, clearance: float = 0.001) -> torch.Tensor:
    """Terminate when the source cup enters the receiver's forbidden outer envelope."""
    source_half_extents = (
        0.5 * float(env.cfg.source_cup_inner_width) + float(env.cfg.source_cup_wall_thickness),
        0.5 * float(env.cfg.source_cup_inner_depth) + float(env.cfg.source_cup_wall_thickness),
        0.5 * (float(env.cfg.source_cup_cavity_depth) + float(env.cfg.source_cup_bottom_thickness)),
    )
    target_half_extents = (
        0.5 * float(env.cfg.target_cup_inner_width) + float(env.cfg.target_cup_wall_thickness),
        0.5 * float(env.cfg.target_cup_inner_depth) + float(env.cfg.target_cup_wall_thickness),
        0.5 * (float(env.cfg.target_cup_cavity_depth) + float(env.cfg.target_cup_bottom_thickness)),
    )
    return _source_receiver_envelope_overlap(
        env.cup_pose_e(),
        env.target_pose_e(),
        source_half_extents=source_half_extents,
        target_half_extents=target_half_extents,
        clearance=clearance,
    )


def particle_out_of_bounds(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate an environment when any media particle escapes its workspace."""
    return ~env.particles_in_workspace()


def excessive_spill(env: FrankaPourEnv, terminate: bool = True) -> torch.Tensor:
    """Track when strictly more than the allowed media fraction is spilled."""
    if not isinstance(terminate, bool):
        raise TypeError("terminate must be a bool.")
    spilled = env.spilled_fraction() > float(env.cfg.max_spill_fraction)
    return spilled if terminate else torch.zeros_like(spilled)


def unsuccessful_time_out(env: FrankaPourEnv) -> torch.Tensor:
    """Truncate at the finite-horizon deadline unless success fired on the same step."""
    deadline = env.episode_length_buf >= env.max_episode_length
    return deadline & ~env.episode_succeeded


def lost_lifted_grasp(
    env: FrankaPourEnv,
    dwell_time_s: float = 0.05,
    max_tcp_distance: float = 0.015,
    max_gripper_width_error: float = 0.012,
    max_gripper_command: float = 0.025,
    terminate: bool = True,
) -> torch.Tensor:
    """Track when a demonstrated lift loses the physical cup grasp continuously.

    A short dwell rejects isolated contact-deflection flicker from the coupled rigid solver while
    still allowing the reverse curriculum to terminate a genuinely dropped cup promptly. Static
    reset-dataset training can disable termination while retaining the state monitor for metrics.
    """
    if not math.isfinite(dwell_time_s) or dwell_time_s <= 0.0:
        raise ValueError(f"dwell_time_s must be finite and positive, got {dwell_time_s}.")
    if not isinstance(terminate, bool):
        raise TypeError("terminate must be a bool.")
    cup_lift = env.cup_pose_e()[:, 2] - float(env.cup_reset_height)
    tcp_distance = torch.linalg.vector_norm(env.tcp_pos_e() - env.cup_grasp_point_e(), dim=-1)
    gripper_width_error = torch.abs(env.gripper_width() - float(env.gripper_grasp_width))
    gripper = env.action_manager.get_term("gripper_action")
    preloaded_grasp = (
        (tcp_distance <= float(max_tcp_distance))
        & (gripper_width_error <= float(max_gripper_width_error))
        & (gripper.commanded_position[:, 0] <= float(max_gripper_command) + _GRIPPER_POSITION_TOLERANCE)
        & gripper.bilateral_contact
    )
    lifted_grasp = preloaded_grasp & (cup_lift >= max(float(env.cfg.success_min_lift_height), 1.0e-6))
    env._lifted_grasp_seen |= lifted_grasp
    lost = env._lifted_grasp_seen & ~preloaded_grasp
    dwell_steps = max(1, math.ceil(float(dwell_time_s) / max(float(env.step_dt), 1.0e-6)))
    env._lost_grasp_dwell_count[:] = torch.where(
        lost,
        torch.clamp(env._lost_grasp_dwell_count + 1, max=dwell_steps),
        0,
    )
    dwell_qualified_loss = lost & (env._lost_grasp_dwell_count >= dwell_steps)
    return dwell_qualified_loss if terminate else torch.zeros_like(dwell_qualified_loss)


def immediate_pour_success_mask(
    target_fraction: torch.Tensor,
    threshold: float | torch.Tensor,
    failure_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the immediate-success predicate shared by runtime and offline validation."""
    return (target_fraction >= threshold) & ~failure_mask


def immediate_pour_success(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate immediately when the current target-bowl fraction reaches its threshold.

    Reset-cache particles always start in the source cup, so current target occupancy directly
    measures task progress without also requiring a retained grasp, lift milestone, delivery
    history, or dwell interval. Failure terms run before success and therefore retain precedence.
    """
    target_fraction = env.count_in_target() / max(env.num_particles, 1)
    success = immediate_pour_success_mask(
        target_fraction,
        env.pour_target_frac,
        env.termination_manager.terminated,
    )
    env._success_dwell_count[:] = success.to(dtype=env._success_dwell_count.dtype)
    env.episode_succeeded |= success
    return success
