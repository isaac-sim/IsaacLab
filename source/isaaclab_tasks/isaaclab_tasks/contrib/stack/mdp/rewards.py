# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse objectives and task-state helpers for cube stacking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .robot_state import (
    end_effector_pose,
    grasp_pair_end_effector_pose,
    grasp_pair_posture_closure,
    two_finger_posture_closure,
)
from .runtime_state import get_stack_reset_runtime_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


STACK_COM_XY_THRESHOLD = 0.02
"""Maximum lateral COM separation for a stacked pair [m]."""

STACK_COM_HEIGHT_THRESHOLD = 0.01
"""Maximum error from one cube-height of vertical COM separation [m]."""


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


def cube_com_pair_aligned(
    upper_com: torch.Tensor,
    lower_com: torch.Tensor,
    xy_threshold: float = STACK_COM_XY_THRESHOLD,
    height_threshold: float = STACK_COM_HEIGHT_THRESHOLD,
    cube_height: float = 0.04,
) -> torch.Tensor:
    """Return whether two cube COMs satisfy the geometric stack threshold."""
    position_delta = upper_com - lower_com
    xy_distance = torch.linalg.vector_norm(position_delta[:, :2], dim=1)
    height_error = torch.abs(position_delta[:, 2] - cube_height)
    return (xy_distance < xy_threshold) & (height_error < height_threshold)


def _pair_is_stacked(
    upper_object: RigidObject,
    lower_object: RigidObject,
    xy_threshold: float,
    height_threshold: float,
    cube_height: float,
) -> torch.Tensor:
    """Return whether an upper and lower cube have aligned centers of mass."""
    return cube_com_pair_aligned(
        upper_object.data.root_pos_w.torch,
        lower_object.data.root_pos_w.torch,
        xy_threshold=xy_threshold,
        height_threshold=height_threshold,
        cube_height=cube_height,
    )


def _gripper_joint_positions(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    gripper_cfg: SceneEntityCfg | None,
) -> torch.Tensor:
    """Return the selected gripper joint positions."""
    if gripper_cfg is not None:
        return robot.data.joint_pos.torch[:, gripper_cfg.joint_ids]

    joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(joint_ids) != 2:
        raise ValueError("Franka stack gripper thresholds require a two-finger parallel gripper.")
    return robot.data.joint_pos.torch[:, joint_ids]


def _gripper_posture_closure(
    joint_positions: torch.Tensor,
    open_joint_positions: tuple[float, ...],
    closed_joint_positions: tuple[float, ...],
    finger_joint_counts: tuple[int, int],
) -> torch.Tensor:
    """Return the weaker finger's least-squares closure projection."""
    return two_finger_posture_closure(
        joint_positions,
        open_joint_positions,
        closed_joint_positions,
        finger_joint_counts,
    ).amin(dim=1)


def _grasp_pair_closure(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg | None,
    grasp_pair_joint_names: tuple[tuple[str, ...], ...],
    grasp_pair_open_joint_positions: tuple[tuple[float, ...], ...],
    grasp_pair_closed_joint_positions: tuple[tuple[float, ...], ...],
    finger_joint_counts: tuple[int, int],
) -> torch.Tensor:
    """Return the weaker active finger's pair-conditioned closure."""
    robot_cfg = SceneEntityCfg(gripper_cfg.name if gripper_cfg is not None else "robot")
    return grasp_pair_posture_closure(
        env,
        robot_cfg=robot_cfg,
        joint_names_by_pair=grasp_pair_joint_names,
        open_joint_positions_by_pair=grasp_pair_open_joint_positions,
        closed_joint_positions_by_pair=grasp_pair_closed_joint_positions,
        finger_joint_counts=finger_joint_counts,
    ).amin(dim=1)


def _gripper_is_released(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    minimum_finger_position: float = 0.023,
    gripper_cfg: SceneEntityCfg | None = None,
    open_joint_positions: tuple[float, ...] | None = None,
    closed_joint_positions: tuple[float, ...] | None = None,
    finger_joint_counts: tuple[int, int] = (4, 4),
    maximum_gripper_closure: float = 0.2,
    grasp_pair_joint_names: tuple[tuple[str, ...], ...] | None = None,
    grasp_pair_open_joint_positions: tuple[tuple[float, ...], ...] | None = None,
    grasp_pair_closed_joint_positions: tuple[tuple[float, ...], ...] | None = None,
) -> torch.Tensor:
    """Return whether the configured gripper has cleared a supported cube.

    Supplying open and closed posture vectors enables arbitrary joint
    directions and counts. Omitting them preserves the Franka threshold.
    """
    pair_mode = (
        grasp_pair_joint_names is not None
        or grasp_pair_open_joint_positions is not None
        or grasp_pair_closed_joint_positions is not None
    )
    if pair_mode:
        if (
            grasp_pair_joint_names is None
            or grasp_pair_open_joint_positions is None
            or grasp_pair_closed_joint_positions is None
        ):
            raise ValueError("Pair-conditioned release requires joint names and open/closed postures for every pair.")
        if not 0.0 <= maximum_gripper_closure <= 1.0:
            raise ValueError("maximum_gripper_closure must be in [0, 1].")
        closure = _grasp_pair_closure(
            env,
            gripper_cfg,
            grasp_pair_joint_names,
            grasp_pair_open_joint_positions,
            grasp_pair_closed_joint_positions,
            finger_joint_counts,
        )
        return closure <= maximum_gripper_closure

    posture_mode = gripper_cfg is not None or open_joint_positions is not None or closed_joint_positions is not None
    if posture_mode:
        if open_joint_positions is None or closed_joint_positions is None or gripper_cfg is None:
            raise ValueError(
                "Posture-based release requires gripper_cfg, open_joint_positions, and closed_joint_positions."
            )
        if not 0.0 <= maximum_gripper_closure <= 1.0:
            raise ValueError("maximum_gripper_closure must be in [0, 1].")
        joint_positions = _gripper_joint_positions(env, robot, gripper_cfg)
        closure = _gripper_posture_closure(
            joint_positions,
            open_joint_positions,
            closed_joint_positions,
            finger_joint_counts,
        )
        return closure <= maximum_gripper_closure

    finger_positions = _gripper_joint_positions(env, robot, None)
    return torch.all(finger_positions > minimum_finger_position, dim=1)


def _gripper_release_progress(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    contact_finger_position: float = 0.020,
    clear_finger_position: float = 0.024,
    gripper_cfg: SceneEntityCfg | None = None,
    open_joint_positions: tuple[float, ...] | None = None,
    closed_joint_positions: tuple[float, ...] | None = None,
    finger_joint_counts: tuple[int, int] = (4, 4),
    minimum_gripper_closure: float = 0.8,
    maximum_gripper_closure: float = 0.2,
    grasp_pair_joint_names: tuple[tuple[str, ...], ...] | None = None,
    grasp_pair_open_joint_positions: tuple[tuple[float, ...], ...] | None = None,
    grasp_pair_closed_joint_positions: tuple[tuple[float, ...], ...] | None = None,
) -> torch.Tensor:
    """Return continuous progress from contact closure to object clearance."""
    pair_mode = (
        grasp_pair_joint_names is not None
        or grasp_pair_open_joint_positions is not None
        or grasp_pair_closed_joint_positions is not None
    )
    if pair_mode:
        if (
            grasp_pair_joint_names is None
            or grasp_pair_open_joint_positions is None
            or grasp_pair_closed_joint_positions is None
        ):
            raise ValueError(
                "Pair-conditioned release progress requires joint names and open/closed postures for every pair."
            )
        if not 0.0 <= maximum_gripper_closure < minimum_gripper_closure <= 1.0:
            raise ValueError("Posture closure thresholds must satisfy 0 <= maximum < minimum <= 1.")
        closure = _grasp_pair_closure(
            env,
            gripper_cfg,
            grasp_pair_joint_names,
            grasp_pair_open_joint_positions,
            grasp_pair_closed_joint_positions,
            finger_joint_counts,
        )
        return ((minimum_gripper_closure - closure) / (minimum_gripper_closure - maximum_gripper_closure)).clamp(
            0.0, 1.0
        )

    posture_mode = gripper_cfg is not None or open_joint_positions is not None or closed_joint_positions is not None
    if posture_mode:
        if open_joint_positions is None or closed_joint_positions is None or gripper_cfg is None:
            raise ValueError(
                "Posture-based release progress requires gripper_cfg, open_joint_positions, and closed_joint_positions."
            )
        if not 0.0 <= maximum_gripper_closure < minimum_gripper_closure <= 1.0:
            raise ValueError("Posture closure thresholds must satisfy 0 <= maximum < minimum <= 1.")
        joint_positions = _gripper_joint_positions(env, robot, gripper_cfg)
        closure = _gripper_posture_closure(
            joint_positions,
            open_joint_positions,
            closed_joint_positions,
            finger_joint_counts,
        )
        return ((minimum_gripper_closure - closure) / (minimum_gripper_closure - maximum_gripper_closure)).clamp(
            0.0, 1.0
        )

    if clear_finger_position <= contact_finger_position:
        raise ValueError("clear_finger_position must exceed contact_finger_position.")
    finger_position = torch.amin(_gripper_joint_positions(env, robot, None), dim=1)
    return torch.clamp(
        (finger_position - contact_finger_position) / (clear_finger_position - contact_finger_position),
        min=0.0,
        max=1.0,
    )


def _gripper_is_closed(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    maximum_finger_position: float,
    gripper_cfg: SceneEntityCfg | None = None,
    open_joint_positions: tuple[float, ...] | None = None,
    closed_joint_positions: tuple[float, ...] | None = None,
    finger_joint_counts: tuple[int, int] = (4, 4),
    minimum_gripper_closure: float = 0.8,
    grasp_pair_joint_names: tuple[tuple[str, ...], ...] | None = None,
    grasp_pair_open_joint_positions: tuple[tuple[float, ...], ...] | None = None,
    grasp_pair_closed_joint_positions: tuple[tuple[float, ...], ...] | None = None,
) -> torch.Tensor:
    """Return whether the configured gripper has closed around the grasp region."""
    pair_mode = (
        grasp_pair_joint_names is not None
        or grasp_pair_open_joint_positions is not None
        or grasp_pair_closed_joint_positions is not None
    )
    if pair_mode:
        if (
            grasp_pair_joint_names is None
            or grasp_pair_open_joint_positions is None
            or grasp_pair_closed_joint_positions is None
        ):
            raise ValueError("Pair-conditioned closure requires joint names and open/closed postures for every pair.")
        if not 0.0 <= minimum_gripper_closure <= 1.0:
            raise ValueError("minimum_gripper_closure must be in [0, 1].")
        closure = _grasp_pair_closure(
            env,
            gripper_cfg,
            grasp_pair_joint_names,
            grasp_pair_open_joint_positions,
            grasp_pair_closed_joint_positions,
            finger_joint_counts,
        )
        return closure >= minimum_gripper_closure

    posture_mode = gripper_cfg is not None or open_joint_positions is not None or closed_joint_positions is not None
    if posture_mode:
        if open_joint_positions is None or closed_joint_positions is None or gripper_cfg is None:
            raise ValueError(
                "Posture-based closure requires gripper_cfg, open_joint_positions, and closed_joint_positions."
            )
        if not 0.0 <= minimum_gripper_closure <= 1.0:
            raise ValueError("minimum_gripper_closure must be in [0, 1].")
        joint_positions = _gripper_joint_positions(env, robot, gripper_cfg)
        closure = _gripper_posture_closure(
            joint_positions,
            open_joint_positions,
            closed_joint_positions,
            finger_joint_counts,
        )
        return closure >= minimum_gripper_closure

    finger_positions = _gripper_joint_positions(env, robot, None)
    return torch.all(finger_positions < maximum_finger_position, dim=1)


_CUBE_PERMUTATIONS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def _order_invariant_cube_state(
    env: ManagerBasedRLEnv,
    cube_cfgs: tuple[SceneEntityCfg, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cube COM positions [m] and spatial velocities [m/s, rad/s]."""
    cubes: tuple[RigidObject, ...] = tuple(env.scene[cfg.name] for cfg in cube_cfgs)
    positions = torch.stack(tuple(cube.data.root_pos_w.torch for cube in cubes), dim=1)
    velocities = torch.stack(tuple(cube.data.root_vel_w.torch for cube in cubes), dim=1)
    return positions, velocities


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
    positions, _ = _order_invariant_cube_state(env, cube_cfgs)
    delta = positions[:, :, None, :] - positions[:, None, :, :]
    xy_distance = torch.linalg.vector_norm(delta[..., :2], dim=-1)
    height_error = torch.abs(delta[..., 2] - cube_height)
    pair_is_stacked = (xy_distance < xy_threshold) & (height_error < height_threshold)
    pair_exists = pair_is_stacked.flatten(start_dim=1).any(dim=1)

    complete = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
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

    middle_stacked = (middle_xy_distance < xy_threshold) & (
        torch.abs(middle_position[:, 2] - base_position[:, 2] - cube_height) < height_threshold
    )
    top_stacked = (
        middle_stacked
        & (top_xy_distance < xy_threshold)
        & (torch.abs(top_position[:, 2] - middle_position[:, 2] - cube_height) < height_threshold)
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
    grasp_pair_tool_offsets: tuple[tuple[float, float, float], ...] | None = None,
    gripper_cfg: SceneEntityCfg | None = None,
    open_gripper_joint_positions: tuple[float, ...] | None = None,
    closed_gripper_joint_positions: tuple[float, ...] | None = None,
    gripper_finger_joint_counts: tuple[int, int] = (4, 4),
    grasp_pair_joint_names: tuple[tuple[str, ...], ...] | None = None,
    grasp_pair_open_joint_positions: tuple[tuple[float, ...], ...] | None = None,
    grasp_pair_closed_joint_positions: tuple[tuple[float, ...], ...] | None = None,
    minimum_gripper_closure: float = 0.8,
    maximum_gripper_closure: float = 0.2,
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
) -> torch.Tensor:
    """Return fixed-center-base progress with either side-cube order.

    Reset-time color permutations are gathered into stable physical roles.
    Role zero remains the base while the maximum over ``1→2`` and ``2→1``
    makes the two movable cubes interchangeable.
    """
    reset_state = get_stack_reset_runtime_state(env)

    positions, _ = _order_invariant_cube_state(env, cube_cfgs)
    role_to_cube = reset_state.role_to_cube.long()
    gather_index = role_to_cube.unsqueeze(-1).expand(-1, -1, 3)
    role_positions = torch.gather(positions, 1, gather_index)
    robot: Articulation = env.scene[robot_cfg.name]
    if grasp_pair_tool_offsets is None:
        ee_position, _ = end_effector_pose(
            env,
            robot_cfg=robot_cfg,
            body_name=tool_body_name,
            body_offset=tool_offset,
        )
    else:
        ee_position, _ = grasp_pair_end_effector_pose(
            env,
            robot_cfg=robot_cfg,
            body_name=tool_body_name,
            tool_offsets_by_pair=grasp_pair_tool_offsets,
        )
    gripper_kwargs = {
        "gripper_cfg": gripper_cfg,
        "open_joint_positions": open_gripper_joint_positions,
        "closed_joint_positions": closed_gripper_joint_positions,
        "finger_joint_counts": gripper_finger_joint_counts,
    }
    if (
        grasp_pair_joint_names is not None
        or grasp_pair_open_joint_positions is not None
        or grasp_pair_closed_joint_positions is not None
    ):
        gripper_kwargs.update(
            {
                "grasp_pair_joint_names": grasp_pair_joint_names,
                "grasp_pair_open_joint_positions": grasp_pair_open_joint_positions,
                "grasp_pair_closed_joint_positions": grasp_pair_closed_joint_positions,
            }
        )
    gripper_release_progress = _gripper_release_progress(
        env,
        robot,
        minimum_gripper_closure=minimum_gripper_closure,
        maximum_gripper_closure=maximum_gripper_closure,
        **gripper_kwargs,
    )
    gripper_released = _gripper_is_released(
        env,
        robot,
        maximum_gripper_closure=maximum_gripper_closure,
        **gripper_kwargs,
    )
    gripper_closed = _gripper_is_closed(
        env,
        robot,
        maximum_grasp_finger_position,
        minimum_gripper_closure=minimum_gripper_closure,
        **gripper_kwargs,
    )
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
