# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-independent end-effector state helpers for stack tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .runtime_state import get_stack_reset_runtime_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def _active_grasp_pair_ids(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the episode-long active grasp-pair indices."""
    return get_stack_reset_runtime_state(env).grasp_pair_ids.long()


def _resolve_grasp_pair_entity_ids(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    names_by_pair: tuple[tuple[str, ...], ...],
) -> tuple[Articulation, torch.Tensor]:
    """Resolve and cache an equally sized joint selection per pair."""
    robot: Articulation = env.scene[robot_cfg.name]
    cache = getattr(env, "_stack_grasp_pair_entity_cache", None)
    if cache is None:
        cache = {}
        env._stack_grasp_pair_entity_cache = cache
    cache_key = (robot_cfg.name, names_by_pair)
    entity_ids = cache.get(cache_key)
    if entity_ids is None:
        if len(names_by_pair) < 1:
            raise ValueError("At least one grasp pair must be configured.")
        resolved: list[list[int]] = []
        for names in names_by_pair:
            ids, _ = robot.find_joints(names, preserve_order=True)
            if len(ids) != len(names):
                raise ValueError(f"Grasp pair {names} resolved to {len(ids)} joints; expected exactly {len(names)}.")
            resolved.append(ids)
        widths = {len(ids) for ids in resolved}
        if len(widths) != 1:
            raise ValueError("Every grasp pair must select the same number of entities.")
        entity_ids = torch.tensor(resolved, dtype=torch.long, device=env.device)
        cache[cache_key] = entity_ids
    return robot, entity_ids


def _gather_active_pair_rows(values: torch.Tensor, pair_ids: torch.Tensor) -> torch.Tensor:
    """Gather one configured pair row for each parallel environment."""
    return values[pair_ids]


def _grasp_pair_value_table(
    env: ManagerBasedRLEnv,
    name: str,
    values: tuple[tuple[float, ...], ...],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Materialize and cache one immutable grasp-pair value table."""
    cache = getattr(env, "_stack_grasp_pair_value_cache", None)
    if cache is None:
        cache = {}
        env._stack_grasp_pair_value_cache = cache
    cache_key = (name, values, reference.dtype)
    table = cache.get(cache_key)
    if table is None:
        table = reference.new_tensor(values)
        cache[cache_key] = table
    return table


def grasp_pair_joint_positions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    *,
    joint_names_by_pair: tuple[tuple[str, ...], ...],
) -> torch.Tensor:
    """Return active two-finger joint positions in canonical pair-role order.

    The first four entries describe the selected opposing finger and the last
    four describe the thumb. Pair identity is episode-local reset metadata,
    while this semantic ordering keeps the policy interface fixed at eight
    entries as the physical opposing finger changes.
    """
    robot, joint_ids_by_pair = _resolve_grasp_pair_entity_ids(env, robot_cfg, joint_names_by_pair)
    joint_ids = _gather_active_pair_rows(joint_ids_by_pair, _active_grasp_pair_ids(env))
    return torch.gather(robot.data.joint_pos.torch, 1, joint_ids)


def two_finger_posture_closure(
    joint_positions: torch.Tensor,
    open_joint_positions: tuple[float, ...],
    closed_joint_positions: tuple[float, ...],
    finger_joint_counts: tuple[int, int] = (4, 4),
) -> torch.Tensor:
    """Project two multi-joint fingers onto an open-to-closed posture synergy.

    Each feature is the least-squares scalar projection onto that finger's
    configured posture direction. This weights joints by their actual motion
    range instead of letting a small-range joint dominate the closure estimate.

    Args:
        joint_positions: Selected finger positions, shape ``(num_envs, joints)``.
        open_joint_positions: Open posture [rad], in selected-joint order.
        closed_joint_positions: Closed posture [rad], in selected-joint order.
        finger_joint_counts: Number of selected joints in each finger.

    Returns:
        The two normalized closure projections, shape ``(num_envs, 2)``.
    """
    if len(finger_joint_counts) != 2 or any(count < 1 for count in finger_joint_counts):
        raise ValueError("finger_joint_counts must contain two positive counts.")
    joint_count = joint_positions.shape[1]
    if sum(finger_joint_counts) != joint_count:
        raise ValueError(
            f"finger_joint_counts selects {sum(finger_joint_counts)} joints, but joint_positions has {joint_count}."
        )
    if len(open_joint_positions) != joint_count or len(closed_joint_positions) != joint_count:
        raise ValueError("Open and closed postures must contain one value per selected joint.")

    open_positions = joint_positions.new_tensor(open_joint_positions)
    posture_delta = joint_positions.new_tensor(closed_joint_positions) - open_positions
    features: list[torch.Tensor] = []
    start = 0
    for count in finger_joint_counts:
        stop = start + count
        squared_range = sum(
            (closed - open_) ** 2
            for open_, closed in zip(
                open_joint_positions[start:stop],
                closed_joint_positions[start:stop],
                strict=True,
            )
        )
        if squared_range <= 1.0e-12:
            raise ValueError("Each finger's open and closed postures must differ.")
        displacement = joint_positions[:, start:stop] - open_positions[start:stop]
        features.append((displacement * posture_delta[start:stop]).sum(dim=1) / squared_range)
        start = stop
    return torch.stack(features, dim=1).clamp(0.0, 1.0)


def grasp_pair_posture_closure(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    *,
    joint_names_by_pair: tuple[tuple[str, ...], ...],
    open_joint_positions_by_pair: tuple[tuple[float, ...], ...],
    closed_joint_positions_by_pair: tuple[tuple[float, ...], ...],
    finger_joint_counts: tuple[int, int] = (4, 4),
) -> torch.Tensor:
    """Return the two closure projections for each environment's active pair."""
    joint_positions = grasp_pair_joint_positions(
        env,
        robot_cfg=robot_cfg,
        joint_names_by_pair=joint_names_by_pair,
    )
    pair_ids = _active_grasp_pair_ids(env)
    open_positions = _gather_active_pair_rows(
        _grasp_pair_value_table(env, "open_joint_positions", open_joint_positions_by_pair, joint_positions),
        pair_ids,
    )
    closed_positions = _gather_active_pair_rows(
        _grasp_pair_value_table(env, "closed_joint_positions", closed_joint_positions_by_pair, joint_positions),
        pair_ids,
    )
    if len(finger_joint_counts) != 2 or any(count < 1 for count in finger_joint_counts):
        raise ValueError("finger_joint_counts must contain two positive counts.")
    if sum(finger_joint_counts) != joint_positions.shape[1]:
        raise ValueError("finger_joint_counts must cover every active grasp-pair joint.")

    posture_delta = closed_positions - open_positions
    features: list[torch.Tensor] = []
    start = 0
    for count in finger_joint_counts:
        stop = start + count
        finger_delta = posture_delta[:, start:stop]
        squared_range = torch.sum(torch.square(finger_delta), dim=1).clamp_min(1.0e-12)
        displacement = joint_positions[:, start:stop] - open_positions[:, start:stop]
        features.append(torch.sum(displacement * finger_delta, dim=1) / squared_range)
        start = stop
    return torch.stack(features, dim=1).clamp(0.0, 1.0)


def _end_effector_cache_entry(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    body_name: str,
    body_offset: tuple[float, float, float],
) -> tuple[Articulation, int, torch.Tensor]:
    """Resolve and cache one articulation body and its tool-frame offset."""
    robot: Articulation = env.scene[robot_cfg.name]
    cache = getattr(env, "_stack_end_effector_cache", None)
    if cache is None:
        cache = {}
        env._stack_end_effector_cache = cache

    cache_key = (robot_cfg.name, body_name, body_offset)
    entry = cache.get(cache_key)
    if entry is None:
        body_ids, _ = robot.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one end-effector body matching '{body_name}', found {len(body_ids)}.")
        entry = (
            body_ids[0],
            torch.tensor(body_offset, dtype=torch.float32, device=env.device),
        )
        cache[cache_key] = entry
    return robot, entry[0], entry[1]


def end_effector_pose(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_hand",
    body_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a configured tool-center position [m] and orientation."""
    robot, body_id, tool_offset = _end_effector_cache_entry(env, robot_cfg, body_name, body_offset)

    body_position = robot.data.body_pos_w.torch[:, body_id]
    body_orientation = robot.data.body_quat_w.torch[:, body_id]
    tool_position = body_position + math_utils.quat_apply(body_orientation, tool_offset.expand(env.num_envs, -1))
    return tool_position, body_orientation


def end_effector_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_hand",
    body_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    """Return the tool-center linear/angular velocity in world coordinates."""
    robot, body_id, tool_offset = _end_effector_cache_entry(env, robot_cfg, body_name, body_offset)

    body_orientation = robot.data.body_quat_w.torch[:, body_id]
    body_velocity = robot.data.body_vel_w.torch[:, body_id]
    tool_offset_world = math_utils.quat_apply(body_orientation, tool_offset.expand(env.num_envs, -1))
    tool_linear_velocity = body_velocity[:, :3] + torch.linalg.cross(
        body_velocity[:, 3:],
        tool_offset_world,
    )
    return torch.cat((tool_linear_velocity, body_velocity[:, 3:]), dim=1)


def grasp_pair_end_effector_pose(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "palm_link",
    *,
    tool_offsets_by_pair: tuple[tuple[float, float, float], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the active grasp pair's tool-center pose.

    Pair-specific offsets account for the different opposing-finger geometry
    without changing the six-dimensional palm orientation representation.
    """
    robot, body_id, _ = _end_effector_cache_entry(env, robot_cfg, body_name, (0.0, 0.0, 0.0))
    body_position = robot.data.body_pos_w.torch[:, body_id]
    body_orientation = robot.data.body_quat_w.torch[:, body_id]
    offsets = _gather_active_pair_rows(
        _grasp_pair_value_table(env, "tool_offsets", tool_offsets_by_pair, body_position),
        _active_grasp_pair_ids(env),
    )
    tool_position = body_position + math_utils.quat_apply(body_orientation, offsets)
    return tool_position, body_orientation


def grasp_pair_end_effector_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "palm_link",
    *,
    tool_offsets_by_pair: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
    """Return active grasp-pair tool-center linear/angular velocity."""
    robot, body_id, _ = _end_effector_cache_entry(env, robot_cfg, body_name, (0.0, 0.0, 0.0))
    body_orientation = robot.data.body_quat_w.torch[:, body_id]
    body_velocity = robot.data.body_vel_w.torch[:, body_id]
    offsets = _gather_active_pair_rows(
        _grasp_pair_value_table(env, "tool_offsets", tool_offsets_by_pair, body_velocity),
        _active_grasp_pair_ids(env),
    )
    offset_world = math_utils.quat_apply(body_orientation, offsets)
    linear_velocity = body_velocity[:, :3] + torch.linalg.cross(body_velocity[:, 3:], offset_world)
    return torch.cat((linear_velocity, body_velocity[:, 3:]), dim=1)
