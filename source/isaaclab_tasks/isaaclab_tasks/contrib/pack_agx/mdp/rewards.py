# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Monotonic stage tracking and sparse rewards for Pack-AGX RL."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from .terminations import agx_in_box_from_pose, agx_is_horizontal_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)


def get_task_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current per-environment Pack-AGX stage."""
    return get_pack_agx_state(env).task_stage


def _advance_with_hold(
    stage: torch.Tensor,
    hold_counter: torch.Tensor,
    *,
    current: int,
    condition: torch.Tensor,
    hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one stage after ``condition`` holds for consecutive steps."""
    in_stage = stage == current
    active_counter = torch.where(
        condition,
        hold_counter + 1,
        torch.zeros_like(hold_counter),
    )
    hold_counter = torch.where(in_stage, active_counter, hold_counter)
    advance = in_stage & (hold_counter >= hold_steps)
    stage = torch.where(advance, torch.full_like(stage, current + 1), stage)
    hold_counter = torch.where(advance, torch.zeros_like(hold_counter), hold_counter)
    return stage, hold_counter


def update_task_stage(
    env: ManagerBasedRLEnv,
    agx_orin_cfg: SceneEntityCfg,
    protective_box_cfg: SceneEntityCfg,
    lift_z: float,
    align_target_z_offset: float,
    align_xy: float,
    rotation_tolerance: float,
    lift_hold_steps: int,
    align_hold_steps: int,
    seat_hold_steps: int,
    print_log: bool,
) -> torch.Tensor:
    """Advance the lift -> align -> seat -> release stage machine.

    Position targets are calibrated from replay episode 21. Requiring consecutive
    simulation steps prevents a one-frame contact impulse from earning a stage
    reward. Release requires the AGX to remain seated, the right thumb to open,
    and the right wrist to retreat.
    """
    state = get_pack_agx_state(env)
    stage = state.task_stage
    old_stage = stage.clone()

    agx_orin = env.scene[agx_orin_cfg.name]
    protective_box = env.scene[protective_box_cfg.name]
    robot = env.scene["robot"]
    agx_pos = agx_orin.data.root_pos_w.torch
    agx_quat = agx_orin.data.root_quat_w.torch
    box_pos = protective_box.data.root_pos_w.torch
    right_wrist_id = list(robot.body_names).index("right_wrist_yaw_link")
    right_thumb_id = list(robot.joint_names).index("right_thumb_MCP_FE")
    right_wrist_pos = robot.data.body_pos_w.torch[:, right_wrist_id]
    right_thumb_pos = robot.data.joint_pos.torch[:, right_thumb_id]

    align_target_distance = _position_distance(
        agx_pos,
        box_pos,
        align_target_z_offset,
    )

    horizontal = agx_is_horizontal_from_quat(
        agx_quat,
        rotation_tolerance=rotation_tolerance,
    )
    lifted = (agx_pos[:, 2] > (state.initial_agx_z + lift_z)) & horizontal
    stage, state.stage_hold_counter = _advance_with_hold(
        stage,
        state.stage_hold_counter,
        current=0,
        condition=lifted,
        hold_steps=lift_hold_steps,
    )

    aligned_while_lifted = (old_stage == 1) & lifted & (align_target_distance <= align_xy)
    stage, state.stage_hold_counter = _advance_with_hold(
        stage,
        state.stage_hold_counter,
        current=1,
        condition=aligned_while_lifted,
        hold_steps=align_hold_steps,
    )

    in_box = agx_in_box_from_pose(
        agx_pos,
        agx_quat,
        box_pos,
    )
    seated = (old_stage == 2) & in_box
    stage, state.stage_hold_counter = _advance_with_hold(
        stage,
        state.stage_hold_counter,
        current=2,
        condition=seated,
        hold_steps=seat_hold_steps,
    )

    right_wrist_distance = torch.linalg.vector_norm(
        right_wrist_pos - agx_pos,
        dim=-1,
    )
    released = (
        (old_stage == 3)
        & in_box
        & (right_thumb_pos <= 0.261799)  # 15 degrees
        & (right_wrist_distance >= 0.32)
    )
    stage, state.stage_hold_counter = _advance_with_hold(
        stage,
        state.stage_hold_counter,
        current=3,
        condition=released,
        hold_steps=seat_hold_steps,
    )

    if print_log and (stage != old_stage).any():
        for env_id in torch.nonzero(stage != old_stage, as_tuple=False).flatten():
            logger.info(
                "Pack-AGX env %d advanced stage %d -> %d",
                int(env_id),
                int(old_stage[env_id]),
                int(stage[env_id]),
            )

    state.task_stage = stage
    return stage


def _sparse_stage_reward(
    env: ManagerBasedRLEnv,
    stage: torch.Tensor,
    previous_attr: str,
    from_stage: int,
) -> torch.Tensor:
    state = get_pack_agx_state(env)
    previous = getattr(state, previous_attr)
    completed = (previous == from_stage) & (stage >= from_stage + 1)
    reward = torch.where(
        completed,
        torch.ones(env.num_envs, device=env.device) / env.step_dt,
        torch.zeros(env.num_envs, device=env.device),
    )
    setattr(state, previous_attr, stage.clone())
    return reward


@dataclass
class PackAgxState:
    """Per-environment progress through lift, alignment, seating, and release."""

    task_stage: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_stage_lift: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_stage_align: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_stage_seat: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    initial_agx_z: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    stage_hold_counter: torch.Tensor = field(default_factory=lambda: torch.empty(0))


def get_pack_agx_state(env: ManagerBasedRLEnv) -> PackAgxState:
    """Return the lazily-created task state stored on ``env``."""
    if not hasattr(env, "pack_agx_state"):
        num_envs = env.num_envs
        device = env.device
        env.pack_agx_state = PackAgxState(
            task_stage=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_stage_lift=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_stage_align=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_stage_seat=torch.zeros(num_envs, dtype=torch.long, device=device),
            initial_agx_z=torch.zeros(num_envs, device=device),
            stage_hold_counter=torch.zeros(num_envs, dtype=torch.long, device=device),
        )
    return env.pack_agx_state


def _position_distance(
    agx_pos: torch.Tensor,
    box_pos: torch.Tensor,
    z_offset: float,
) -> torch.Tensor:
    displacement = agx_pos - box_pos
    displacement[:, 2] -= z_offset
    return torch.linalg.vector_norm(displacement, dim=-1)


def lift_agx_reward(
    env: ManagerBasedRLEnv,
    agx_orin_cfg: SceneEntityCfg,
    protective_box_cfg: SceneEntityCfg,
    lift_z: float,
    align_target_z_offset: float,
    align_xy: float,
    rotation_tolerance: float,
    lift_hold_steps: int,
    align_hold_steps: int,
    seat_hold_steps: int,
    print_log: bool,
) -> torch.Tensor:
    """Reward the first stable lift and update all task stages."""
    stage = update_task_stage(
        env,
        agx_orin_cfg=agx_orin_cfg,
        protective_box_cfg=protective_box_cfg,
        lift_z=lift_z,
        align_target_z_offset=align_target_z_offset,
        align_xy=align_xy,
        rotation_tolerance=rotation_tolerance,
        lift_hold_steps=lift_hold_steps,
        align_hold_steps=align_hold_steps,
        seat_hold_steps=seat_hold_steps,
        print_log=print_log,
    )
    return _sparse_stage_reward(env, stage, "prev_stage_lift", from_stage=0)


def align_agx_reward(
    env: ManagerBasedRLEnv,
    agx_orin_cfg: SceneEntityCfg = SceneEntityCfg("agx_orin"),
    protective_box_cfg: SceneEntityCfg = SceneEntityCfg("protective_box"),
    distance_offset: float = 0.25,
) -> torch.Tensor:
    """Reward alignment plus positive progress shaping during Stage 1.

    The continuous term is ``distance_offset - distance``, zeroed at the typical
    stage-entry distance (~0.25 m from the randomized initial poses) so it is
    positive whenever the AGX is closer to the align target than at entry.
    """
    state = get_pack_agx_state(env)
    reward = _sparse_stage_reward(
        env,
        state.task_stage,
        "prev_stage_align",
        from_stage=1,
    )
    distance = _position_distance(
        env.scene[agx_orin_cfg.name].data.root_pos_w.torch,
        env.scene[protective_box_cfg.name].data.root_pos_w.torch,
        0.10,
    )
    return reward + torch.where(
        state.task_stage == 1,
        distance_offset - distance,
        torch.zeros_like(distance),
    )


def seat_agx_reward(
    env: ManagerBasedRLEnv,
    agx_orin_cfg: SceneEntityCfg = SceneEntityCfg("agx_orin"),
    protective_box_cfg: SceneEntityCfg = SceneEntityCfg("protective_box"),
    target_offset: float = 0.25,
) -> torch.Tensor:
    """Reward seating/release plus positive progress shaping during Stage 2.

    The continuous term is ``target_offset - (distance + yaw_distance)``, zeroed
    at the typical stage-entry error (~0.12 m position + ~0.1 rad yaw) so it is
    positive whenever the AGX is closer to the seated pose than at entry.
    """
    state = get_pack_agx_state(env)
    previous = state.prev_stage_seat
    completed = ((previous == 2) & (state.task_stage >= 3)) | ((previous == 3) & (state.task_stage >= 4))
    reward = torch.where(
        completed,
        torch.ones(env.num_envs, device=env.device) / env.step_dt,
        torch.zeros(env.num_envs, device=env.device),
    )
    state.prev_stage_seat = state.task_stage.clone()
    agx_orin = env.scene[agx_orin_cfg.name]
    protective_box = env.scene[protective_box_cfg.name]
    distance = _position_distance(
        agx_orin.data.root_pos_w.torch,
        protective_box.data.root_pos_w.torch,
        -0.01945,
    )
    _, _, agx_yaw = euler_xyz_from_quat(agx_orin.data.root_quat_w.torch)
    _, _, box_yaw = euler_xyz_from_quat(protective_box.data.root_quat_w.torch)
    yaw_delta = agx_yaw - box_yaw
    yaw_distance = torch.abs(torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)))
    return reward + torch.where(
        state.task_stage == 2,
        target_offset - (distance + yaw_distance),
        torch.zeros_like(distance),
    )
