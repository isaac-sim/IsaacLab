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
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from ..geometry import box_center_from_base_pose, oriented_boxes_overlap

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


_GRIPPER_POSITION_TOLERANCE = 1.0e-6


def nonfinite_failure(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate on non-finite simulation state (instability guard)."""
    cup_velocity = env._source_cup.data.root_link_vel_w.torch
    finite = (
        torch.isfinite(env._robot.data.joint_pos.torch).all(dim=-1)
        & torch.isfinite(env._robot.data.joint_vel.torch).all(dim=-1)
        & torch.isfinite(env._robot.data.body_link_pose_w.torch[:, env._tcp_body_idx]).all(dim=-1)
        & torch.isfinite(env._source_cup.data.root_link_pose_w.torch).all(dim=-1)
        & torch.isfinite(cup_velocity).all(dim=-1)
        & torch.isfinite(env._media.data.particle_pos_w.torch).all(dim=(1, 2))
    )
    return ~finite


def extreme_rigid_state(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate finite rigid states before extreme values reach observation normalization."""
    joint_position = env._robot.data.joint_pos.torch
    joint_velocity = env._robot.data.joint_vel.torch
    joint_lower = env._joint_pos_limits_t[..., 0] - float(env.cfg.state_bound_joint_position_margin)
    joint_upper = env._joint_pos_limits_t[..., 1] + float(env.cfg.state_bound_joint_position_margin)
    joint_ok = ((joint_position >= joint_lower) & (joint_position <= joint_upper)).all(dim=-1)
    joint_ok &= (joint_velocity.abs() <= float(env.cfg.state_bound_max_joint_velocity)).all(dim=-1)

    hand_pose = env._robot.data.body_link_pose_w.torch[:, env._tcp_body_idx]
    tcp_position, tcp_quaternion = math_utils.combine_frame_transforms(
        hand_pose[:, :3], hand_pose[:, 3:7], env._tcp_offset_pos, env._tcp_offset_quat
    )
    tcp_pose = torch.cat((tcp_position, tcp_quaternion), dim=-1)
    cup_pose = env._source_cup.data.root_link_pose_w.torch
    lower, upper = env._particle_workspace_lower_t, env._particle_workspace_upper_t
    tcp_position_e = tcp_pose[:, :3] - env.env_origins
    cup_position_e = cup_pose[:, :3] - env.env_origins
    pose_ok = (
        ((tcp_position_e >= lower) & (tcp_position_e <= upper)).all(dim=-1)
        & ((cup_position_e >= lower) & (cup_position_e <= upper)).all(dim=-1)
        & torch.isfinite(tcp_pose).all(dim=-1)
        & torch.isfinite(cup_pose).all(dim=-1)
        & ((torch.linalg.vector_norm(tcp_pose[:, 3:7], dim=-1) - 1.0).abs() <= 0.1)
        & ((torch.linalg.vector_norm(cup_pose[:, 3:7], dim=-1) - 1.0).abs() <= 0.1)
    )
    cup_velocity = env._source_cup.data.root_link_vel_w.torch
    velocity_ok = torch.linalg.vector_norm(cup_velocity[:, :3], dim=-1) <= float(
        env.cfg.state_bound_max_cup_linear_velocity
    )
    velocity_ok &= torch.linalg.vector_norm(cup_velocity[:, 3:], dim=-1) <= float(
        env.cfg.state_bound_max_cup_angular_velocity
    )
    return ~(joint_ok & pose_ok & velocity_ok)


def source_receiver_overlap(env: FrankaPourEnv, clearance: float = 0.001) -> torch.Tensor:
    """Terminate when the source cup enters the receiver's forbidden outer envelope."""
    source_pose, target_pose = env.cup_pose_e(), env.target_pose_e()
    return oriented_boxes_overlap(
        box_center_from_base_pose(source_pose, env._source_outer_half_t),
        source_pose[:, 3:7],
        env._source_outer_half_t,
        box_center_from_base_pose(target_pose, env._target_outer_half_t),
        target_pose[:, 3:7],
        env._target_outer_half_t,
        clearance,
    )


def particle_out_of_bounds(env: FrankaPourEnv) -> torch.Tensor:
    """Terminate an environment when any media particle escapes its workspace."""
    particles = env.particle_pos_e()
    return ~((particles >= env._particle_workspace_lower_t) & (particles <= env._particle_workspace_upper_t)).all(
        dim=(1, 2)
    )


def excessive_spill(env: FrankaPourEnv, terminate: bool = True) -> torch.Tensor:
    """Track when strictly more than the allowed media fraction is spilled."""
    spilled = env._particle_region_masks()[2].sum(dim=1) / max(env._num_particles, 1)
    spilled = spilled > float(env.cfg.max_spill_fraction)
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
    cup_lift = env.cup_pose_e()[:, 2] - env._cup_reset_height
    tcp_distance = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - env.cup_grasp_point_e(), dim=-1)
    gripper_width_error = torch.abs(env.gripper_width() - env._gripper_grasp_width)
    gripper = env.action_manager.get_term("gripper_action")
    reached_grasp = tcp_distance <= float(max_tcp_distance)
    bilateral_grasp = (
        reached_grasp
        & (gripper_width_error <= float(max_gripper_width_error))
        & (gripper.commanded_position[:, 0] <= float(max_gripper_command) + _GRIPPER_POSITION_TOLERANCE)
        & gripper.bilateral_contact
    )
    lifted_grasp = bilateral_grasp & (cup_lift >= max(float(env.cfg.success_min_lift_height), 1.0e-6))
    env._lifted_grasp_seen |= lifted_grasp
    lost = env._lifted_grasp_seen & ~bilateral_grasp
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
    target_fraction = env._particle_region_masks()[1].sum(dim=1) / max(env._num_particles, 1)
    success = immediate_pour_success_mask(
        target_fraction,
        env.pour_target_frac,
        env.termination_manager.terminated,
    )
    env._success_dwell_count[:] = success.to(dtype=env._success_dwell_count.dtype)
    env.episode_succeeded |= success
    return success
