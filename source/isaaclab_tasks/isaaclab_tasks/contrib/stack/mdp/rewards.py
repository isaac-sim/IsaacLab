# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse objectives and task-state helpers for cube stacking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .robot_state import end_effector_pose
from .runtime_state import get_stack_reset_runtime_state
from .terminations import _cube_pair_is_aligned

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def action_term_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Penalize the magnitude of one named raw action term."""
    action = env.action_manager.get_term(action_name).raw_actions
    return torch.sum(torch.square(action), dim=1)


def finite_joint_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    maximum_velocity: float = 3.0,
) -> torch.Tensor:
    """Penalize bounded joint velocity without propagating solver divergence.

    Nonfinite robot states are terminated separately. Sanitizing this
    regularizer keeps the reward finite on the same transition so one diverged
    environment cannot poison the complete PPO rollout before it is reset.
    """
    if maximum_velocity <= 0.0:
        raise ValueError("maximum_velocity must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    velocity = torch.nan_to_num(
        velocity,
        nan=0.0,
        posinf=maximum_velocity,
        neginf=-maximum_velocity,
    )
    velocity = torch.clamp(velocity, min=-maximum_velocity, max=maximum_velocity)
    return torch.sum(torch.square(velocity), dim=1)


def irrecoverable_stack_failure(
    env: ManagerBasedRLEnv,
    success_termination_name: str = "success",
) -> torch.Tensor:
    """Return a unit-integral pulse for a non-timeout terminal failure.

    Intermediate reset rows deliberately include states far from the final
    stack. Treating an ordinary horizon timeout as a failure overwhelms rare
    forward-transition samples and teaches the policy to stop. Drops,
    out-of-workspace states, and numerical divergence remain penalized.
    """
    succeeded = env.termination_manager.get_term(success_termination_name)
    failed = env.reset_terminated & ~succeeded
    # RewardManager integrates rates by multiplying every term by step_dt.
    # Dividing here makes the configured weight the exact episode impulse.
    return failed.float() / max(float(env.step_dt), 1.0e-6)


def _resolve_finger_joint_ids(env: ManagerBasedRLEnv, robot: Articulation) -> list[int]:
    """Resolve both fingers of the Franka parallel gripper."""
    joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(joint_ids) != 2:
        raise ValueError("Franka stack gripper thresholds require a two-finger parallel gripper.")
    return joint_ids


_CUBE_PERMUTATIONS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def _cube_positions(
    env: ManagerBasedRLEnv,
    cube_cfgs: tuple[SceneEntityCfg, ...],
    env_ids: Sequence[int] | torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Read selected world-frame cube positions [m]."""
    return torch.stack([env.scene[cfg.name].data.root_pos_w.torch[env_ids] for cfg in cube_cfgs], dim=1)


def order_invariant_stack_progress(
    env: ManagerBasedRLEnv,
    cube_cfgs: tuple[SceneEntityCfg, ...] = (
        SceneEntityCfg("cube_1"),
        SceneEntityCfg("cube_2"),
        SceneEntityCfg("cube_3"),
    ),
    xy_threshold: float = 0.025,
    height_threshold: float = 0.012,
    cube_height: float = 0.04,
) -> torch.Tensor:
    """Return zero, one, or two stacked pairs without assigning cube roles."""
    return _stack_progress_from_positions(_cube_positions(env, cube_cfgs), xy_threshold, height_threshold, cube_height)


def _stack_progress_from_positions(
    positions: torch.Tensor, xy_threshold: float, height_threshold: float, cube_height: float
) -> torch.Tensor:
    delta = positions[:, :, None, :] - positions[:, None, :, :]
    pair_is_stacked = _cube_pair_is_aligned(delta, xy_threshold, cube_height, height_threshold, True)
    pair_exists = pair_is_stacked.flatten(start_dim=1).any(dim=1)

    complete = torch.zeros_like(pair_exists)
    for base_id, middle_id, top_id in _CUBE_PERMUTATIONS:
        complete |= pair_is_stacked[:, middle_id, base_id] & pair_is_stacked[:, top_id, middle_id]
    return pair_exists.float() + complete.float()


def _role_plan_potential(
    base_position: torch.Tensor,
    middle_position: torch.Tensor,
    top_position: torch.Tensor,
    *,
    ee_position: torch.Tensor,
    gripper_release_progress: torch.Tensor,
    gripper_released: torch.Tensor,
    gripper_closed: torch.Tensor,
    reach_std: float,
    align_std: float,
    place_std: float,
    grasp_distance: float,
    lift_height: float,
    lifted_fraction: float,
    align_distance: float,
    xy_threshold: float,
    height_threshold: float,
    cube_height: float,
) -> torch.Tensor:
    """Return continuous progress through one fixed-base side-cube order."""
    middle_distance = torch.linalg.vector_norm(middle_position - ee_position, dim=1)
    top_distance = torch.linalg.vector_norm(top_position - ee_position, dim=1)
    middle_xy_distance = torch.linalg.vector_norm(middle_position[:, :2] - base_position[:, :2], dim=1)
    top_xy_distance = torch.linalg.vector_norm(top_position[:, :2] - middle_position[:, :2], dim=1)
    middle_place_error = torch.linalg.vector_norm(
        torch.cat(
            (
                middle_position[:, :2] - base_position[:, :2],
                (middle_position[:, 2] - base_position[:, 2] - cube_height).unsqueeze(1),
            ),
            dim=1,
        ),
        dim=1,
    )
    top_place_error = torch.linalg.vector_norm(
        torch.cat(
            (
                top_position[:, :2] - middle_position[:, :2],
                (top_position[:, 2] - middle_position[:, 2] - cube_height).unsqueeze(1),
            ),
            dim=1,
        ),
        dim=1,
    )

    middle_stacked = _cube_pair_is_aligned(
        middle_position - base_position, xy_threshold, cube_height, height_threshold, True
    )
    top_stacked = middle_stacked & _cube_pair_is_aligned(
        top_position - middle_position, xy_threshold, cube_height, height_threshold, True
    )
    middle_near = middle_distance < grasp_distance
    top_near = top_distance < grasp_distance
    middle_grasped = middle_near & gripper_closed & ~middle_stacked
    top_grasped = top_near & gripper_closed & ~top_stacked
    middle_released = middle_stacked & (~middle_near | gripper_released)

    middle_reach = 1.0 - torch.tanh(middle_distance / reach_std)
    top_reach = 1.0 - torch.tanh(top_distance / reach_std)
    middle_lift = torch.clamp(
        (middle_position[:, 2] - base_position[:, 2]) / lift_height,
        min=0.0,
        max=1.0,
    )
    top_lift = torch.clamp(
        (top_position[:, 2] - base_position[:, 2]) / lift_height,
        min=0.0,
        max=1.0,
    )
    middle_align = 1.0 - torch.tanh(middle_xy_distance / align_std)
    top_align = 1.0 - torch.tanh(top_xy_distance / align_std)
    middle_place = 1.0 - torch.tanh(middle_place_error / place_std)
    top_place = 1.0 - torch.tanh(top_place_error / place_std)

    potential = middle_reach
    potential = torch.where(middle_grasped, torch.maximum(potential, 1.0 + middle_lift), potential)
    middle_lifted = middle_grasped & (middle_lift > lifted_fraction)
    potential = torch.where(middle_lifted, torch.maximum(potential, 2.0 + middle_align), potential)
    middle_aligned = middle_lifted & (middle_xy_distance < align_distance)
    potential = torch.where(middle_aligned, torch.maximum(potential, 3.0 + middle_place), potential)
    potential = torch.where(
        middle_stacked,
        torch.maximum(potential, 4.0 + gripper_release_progress),
        potential,
    )
    potential = torch.where(middle_released, torch.maximum(potential, 5.0 + top_reach), potential)
    potential = torch.where(
        middle_released & top_grasped,
        torch.maximum(potential, 6.0 + top_lift),
        potential,
    )
    top_lifted = middle_released & top_grasped & (top_lift > lifted_fraction)
    potential = torch.where(top_lifted, torch.maximum(potential, 7.0 + top_align), potential)
    top_aligned = top_lifted & (top_xy_distance < align_distance)
    potential = torch.where(top_aligned, torch.maximum(potential, 8.0 + top_place), potential)
    potential = torch.where(
        top_stacked,
        torch.maximum(potential, 9.0 + gripper_release_progress),
        potential,
    )
    return potential


def role_conditioned_stack_potential(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tool_body_name: str = "panda_hand",
    tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
    cube_cfgs: tuple[SceneEntityCfg, ...] = (
        SceneEntityCfg("cube_1"),
        SceneEntityCfg("cube_2"),
        SceneEntityCfg("cube_3"),
    ),
    reach_std: float = 0.10,
    align_std: float = 0.08,
    place_std: float = 0.05,
    grasp_distance: float = 0.045,
    maximum_grasp_finger_position: float = 0.03,
    lift_height: float = 0.10,
    lifted_fraction: float = 0.45,
    align_distance: float = 0.06,
    xy_threshold: float = 0.025,
    height_threshold: float = 0.012,
    cube_height: float = 0.04,
    *,
    finger_joint_ids: Sequence[int] | None = None,
    env_ids: Sequence[int] | torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """Return progress with the reset-assigned base and either movable cube order.

    Reset-time color permutations are gathered into stable physical roles.
    Role zero remains the base while the maximum over ``1→2`` and ``2→1``
    makes the two movable cubes interchangeable.
    """
    reset_state = get_stack_reset_runtime_state(env)

    positions = _cube_positions(env, cube_cfgs, env_ids)
    role_to_cube = reset_state.role_to_cube[env_ids]
    gather_index = role_to_cube.unsqueeze(-1).expand(-1, -1, 3)
    role_positions = torch.gather(positions, 1, gather_index)
    robot: Articulation = env.scene[robot_cfg.name]
    ee_position, _ = end_effector_pose(env, robot_cfg, tool_body_name, tool_offset, env_ids=env_ids)
    if finger_joint_ids is None:
        finger_joint_ids = _resolve_finger_joint_ids(env, robot)
    finger_positions = robot.data.joint_pos.torch[env_ids][:, finger_joint_ids]
    gripper_release_progress = ((finger_positions.amin(dim=1) - 0.020) / (0.024 - 0.020)).clamp(0.0, 1.0)
    gripper_released = (finger_positions > 0.023).all(dim=1)
    gripper_closed = (finger_positions < maximum_grasp_finger_position).all(dim=1)
    common = {
        "ee_position": ee_position,
        "gripper_release_progress": gripper_release_progress,
        "gripper_released": gripper_released,
        "gripper_closed": gripper_closed,
        "reach_std": reach_std,
        "align_std": align_std,
        "place_std": place_std,
        "grasp_distance": grasp_distance,
        "lift_height": lift_height,
        "lifted_fraction": lifted_fraction,
        "align_distance": align_distance,
        "xy_threshold": xy_threshold,
        "height_threshold": height_threshold,
        "cube_height": cube_height,
    }
    first_then_second = _role_plan_potential(
        role_positions[:, 0],
        role_positions[:, 1],
        role_positions[:, 2],
        **common,
    )
    second_then_first = _role_plan_potential(
        role_positions[:, 0],
        role_positions[:, 2],
        role_positions[:, 1],
        **common,
    )
    return torch.nan_to_num(
        torch.maximum(first_then_second, second_then_first),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def stack_success_pulse(
    env: ManagerBasedRLEnv,
    context_term_name: str = "progress_context",
) -> torch.Tensor:
    """Return a unit-integral pulse for the first stable stack event."""
    context = env.termination_manager.get_term_cfg(context_term_name).func
    return context.new_success.float() / max(float(env.step_dt), 1.0e-6)
