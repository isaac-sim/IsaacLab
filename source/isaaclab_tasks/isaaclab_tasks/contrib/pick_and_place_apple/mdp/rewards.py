# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage machine and sparse rewards for H2 pick-and-place apple RL."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)

__all__ = [
    "PnpAppleState",
    "get_pnp_apple_state",
    "get_task_stage",
    "update_task_stage",
    "left_grasp_lift_reward",
    "handover_to_right_reward",
    "place_on_plate_reward",
    "release_on_plate_reward",
]


@dataclass
class PnpAppleState:
    """Per-env task state for the pick-and-place apple environment.

    Stage semantics:
        0 - Reach & left grasp + lift
        1 - Right hand catches apple while left releases
        2 - Place apple on plate
        3 - Release apple while it remains on plate
        4 - Task complete
    """

    task_stage: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_stage_left_grasp: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_stage_handover: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_stage_place: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_stage_release: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    initial_apple_z: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    left_dist_at_grasp: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    stage_hold_counter: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    last_debug_print_step: int = -1


def get_pnp_apple_state(env: ManagerBasedRLEnv) -> PnpAppleState:
    """Get or lazily initialise :class:`PnpAppleState` on *env*."""
    if not hasattr(env, "pnp_apple_state"):
        device = env.device
        n = env.num_envs
        env.pnp_apple_state = PnpAppleState(
            task_stage=torch.zeros(n, dtype=torch.long, device=device),
            prev_stage_left_grasp=torch.zeros(n, dtype=torch.long, device=device),
            prev_stage_handover=torch.zeros(n, dtype=torch.long, device=device),
            prev_stage_place=torch.zeros(n, dtype=torch.long, device=device),
            prev_stage_release=torch.zeros(n, dtype=torch.long, device=device),
            initial_apple_z=torch.zeros(n, device=device),
            left_dist_at_grasp=torch.full((n,), float("inf"), device=device),
            stage_hold_counter=torch.zeros(n, dtype=torch.long, device=device),
        )
    return env.pnp_apple_state


def get_task_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current per-env task stage tensor."""
    return get_pnp_apple_state(env).task_stage


def _wrist_positions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene[robot_cfg.name]
    body_names = list(robot.body_names)
    left_bid = body_names.index("left_wrist_yaw_link")
    right_bid = body_names.index("right_wrist_yaw_link")
    body_pos = robot.data.body_pos_w.torch
    return body_pos[:, left_bid], body_pos[:, right_bid]


def _right_thumb_open(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Classify the right thumb as open using demo-calibrated joint thresholds."""
    robot = env.scene[robot_cfg.name]
    thumb_joint_names = (
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
    )
    joint_names = list(robot.joint_names)
    joint_ids = torch.tensor(
        [joint_names.index(name) for name in thumb_joint_names],
        dtype=torch.long,
        device=env.device,
    )
    joint_pos = robot.data.joint_pos.torch
    thumb_pos = joint_pos.index_select(dim=1, index=joint_ids)
    # 1.5x the demo open-pose means, in degrees:
    # CMC_FE 2.55, CMC_AA 10.35, MCP_AA 23.4, IP 18.75.
    thresholds = torch.tensor(
        [2.55, 10.35, 23.4, 18.75],
        dtype=thumb_pos.dtype,
        device=env.device,
    ) * (torch.pi / 180.0)
    open_votes = thumb_pos < thresholds
    return open_votes.sum(dim=1) >= 3


def _apple_plate_positions(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    apple_pos = env.scene[apple_cfg.name].data.root_pos_w.torch
    plate_pos = env.scene[plate_cfg.name].data.root_pos_w.torch
    return apple_pos, plate_pos


def _apple_on_plate(
    apple_pos: torch.Tensor,
    plate_pos: torch.Tensor,
    xy_radius: float,
    z_above: float,
    z_window: float,
) -> torch.Tensor:
    dx = apple_pos[:, 0] - plate_pos[:, 0]
    dy = apple_pos[:, 1] - plate_pos[:, 1]
    horiz = torch.sqrt(dx * dx + dy * dy)
    in_xy = horiz < xy_radius
    dz = apple_pos[:, 2] - plate_pos[:, 2]
    in_z = (dz > z_above) & (dz < (z_above + z_window))
    return in_xy & in_z


def update_task_stage(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lift_z: float = 0.10,
    grasp_dist: float = 0.26,
    release_dist: float = 0.08,
    place_grasp_dist: float = 0.30,
    release_grasp_dist: float = 0.32,
    xy_radius: float = 0.10,
    z_above: float = 0.025,
    z_window: float = 0.075,
    hold_steps: int = 5,
    print_log: bool = False,
) -> torch.Tensor:
    """Advance the monotonic stage machine. Returns current stage tensor.

    Distance thresholds are calibrated from 11 annotated demos (wrist_yaw_link
    -> apple center): a holding hand sits ~0.22m from the apple center and the
    apple is lifted ~0.19m by the handover, so grasp_dist=0.26 and lift_z=0.10.
    At a genuine placement the right hand carries the apple down and stays
    ~0.23m from it (max 0.28), whereas an accidental drop from handover height
    leaves the hand ~0.38m away; ``place_grasp_dist=0.30`` rejects such drops.
    Stage 4 requires at least three of four discriminative right-thumb joints
    to indicate an open pose while the apple remains on the plate.
    ``release_grasp_dist`` is retained for config compatibility but is unused.
    """
    state = get_pnp_apple_state(env)
    stage = state.task_stage
    old_stage = stage.clone()

    apple_pos, plate_pos = _apple_plate_positions(env, apple_cfg, plate_cfg)
    left_wrist, right_wrist = _wrist_positions(env, robot_cfg)

    left_dist = torch.norm(apple_pos - left_wrist, dim=-1)
    right_dist = torch.norm(apple_pos - right_wrist, dim=-1)

    # Stage 0 -> 1: left grasp + lift
    lifted = apple_pos[:, 2] > (state.initial_apple_z + lift_z)
    left_close = left_dist < grasp_dist
    cond_0 = lifted & left_close
    stage, state.stage_hold_counter = _advance_with_hold(
        stage, state.stage_hold_counter, current=0, cond=cond_0, hold_steps=hold_steps
    )
    entering_stage_1 = (old_stage == 0) & (stage >= 1)
    state.left_dist_at_grasp = torch.where(entering_stage_1, left_dist, state.left_dist_at_grasp)

    # Stage 1 -> 2: right hand catches while left hand releases, with apple still lifted.
    right_close = right_dist < grasp_dist
    left_released = left_dist > (state.left_dist_at_grasp + release_dist)
    still_lifted = apple_pos[:, 2] > (state.initial_apple_z + lift_z)
    cond_1 = right_close & left_released & still_lifted
    stage, state.stage_hold_counter = _advance_with_hold(
        stage, state.stage_hold_counter, current=1, cond=cond_1, hold_steps=hold_steps
    )

    # Stage 2 -> 3: apple on plate AND still controlled by the right hand.
    # Requiring the right wrist to stay near the apple distinguishes a genuine
    # placement (hand carries the apple down, ~0.23m) from an accidental drop
    # during handover (apple free-falls onto the plate while the hand remains
    # up at handover height, ~0.38m away).
    on_plate = _apple_on_plate(apple_pos, plate_pos, xy_radius, z_above, z_window)
    right_controls = right_dist < place_grasp_dist
    cond_2 = on_plate & right_controls
    stage, state.stage_hold_counter = _advance_with_hold(
        stage, state.stage_hold_counter, current=2, cond=cond_2, hold_steps=hold_steps
    )

    # Stage 3 -> 4: right thumb opens while the apple remains on the plate.
    # Requiring old_stage == 3 prevents a one-frame 2 -> 3 -> 4 transition.
    right_released = _right_thumb_open(env, robot_cfg)
    cond_3 = (old_stage == 3) & on_plate & right_released
    stage, state.stage_hold_counter = _advance_with_hold(
        stage, state.stage_hold_counter, current=3, cond=cond_3, hold_steps=hold_steps
    )

    if print_log and (stage != old_stage).any():
        for env_id in range(env.num_envs):
            if stage[env_id] != old_stage[env_id]:
                logger.debug(
                    "Env %d: Stage %d -> %d",
                    env_id,
                    old_stage[env_id].item(),
                    stage[env_id].item(),
                )

    state.task_stage = stage
    return stage


def _advance_with_hold(
    stage: torch.Tensor,
    hold_counter: torch.Tensor,
    current: int,
    cond: torch.Tensor,
    hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count consecutive satisfied steps before advancing from *current*."""
    in_stage = stage == current
    # Only the active stage owns the shared counter.  Calls that check later
    # stages in the same update must preserve it; otherwise the stage-1/stage-2
    # checks reset stage 0's progress every step and no transition can ever
    # reach hold_steps.
    active_counter = torch.where(
        cond,
        hold_counter + 1,
        torch.zeros_like(hold_counter),
    )
    hold_counter = torch.where(in_stage, active_counter, hold_counter)
    advance = in_stage & (hold_counter >= hold_steps)
    next_stage = torch.full_like(stage, current + 1)
    stage = torch.where(advance, next_stage, stage)
    hold_counter = torch.where(advance, torch.zeros_like(hold_counter), hold_counter)
    return stage, hold_counter


def _sparse_stage_reward(
    env: ManagerBasedRLEnv,
    stage: torch.Tensor,
    prev_stage_attr: str,
    from_stage: int,
) -> torch.Tensor:
    state = get_pnp_apple_state(env)
    prev = getattr(state, prev_stage_attr)
    just_completed = (prev == from_stage) & (stage >= from_stage + 1)
    reward = torch.where(
        just_completed,
        torch.ones(env.num_envs, device=env.device) / env.step_dt,
        torch.zeros(env.num_envs, device=env.device),
    )
    setattr(state, prev_stage_attr, stage.clone())
    return reward


def left_grasp_lift_reward(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lift_z: float = 0.10,
    grasp_dist: float = 0.26,
    release_dist: float = 0.08,
    place_grasp_dist: float = 0.30,
    release_grasp_dist: float = 0.32,
    xy_radius: float = 0.10,
    z_above: float = 0.025,
    z_window: float = 0.075,
    hold_steps: int = 5,
    print_log: bool = False,
) -> torch.Tensor:
    """Sparse reward for stage 0 -> 1 (left grasp + lift). Also advances stages."""
    stage = update_task_stage(
        env,
        apple_cfg,
        plate_cfg,
        robot_cfg,
        lift_z=lift_z,
        grasp_dist=grasp_dist,
        release_dist=release_dist,
        place_grasp_dist=place_grasp_dist,
        release_grasp_dist=release_grasp_dist,
        xy_radius=xy_radius,
        z_above=z_above,
        z_window=z_window,
        hold_steps=hold_steps,
        print_log=print_log,
    )
    return _sparse_stage_reward(env, stage, "prev_stage_left_grasp", from_stage=0)


def handover_to_right_reward(
    env: ManagerBasedRLEnv,
    print_log: bool = False,
) -> torch.Tensor:
    """Sparse reward for stage 1 -> 2 (right handover)."""
    stage = get_task_stage(env)
    return _sparse_stage_reward(env, stage, "prev_stage_handover", from_stage=1)


def place_on_plate_reward(
    env: ManagerBasedRLEnv,
    print_log: bool = False,
) -> torch.Tensor:
    """Sparse reward for stage 2 -> 3 (apple placed on plate)."""
    stage = get_task_stage(env)
    return _sparse_stage_reward(env, stage, "prev_stage_place", from_stage=2)


def release_on_plate_reward(
    env: ManagerBasedRLEnv,
    print_log: bool = False,
) -> torch.Tensor:
    """Sparse reward for stage 3 -> 4 (right hand releases apple on plate)."""
    stage = get_task_stage(env)
    return _sparse_stage_reward(env, stage, "prev_stage_release", from_stage=3)
