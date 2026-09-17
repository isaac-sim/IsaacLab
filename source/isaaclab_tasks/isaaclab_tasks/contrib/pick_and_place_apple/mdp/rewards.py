# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase machine and sparse rewards for H2 pick-and-place apple RL."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, ClassVar

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)


def _advance_with_hold(
    phase: torch.Tensor,
    hold_counter: torch.Tensor,
    *,
    current: int,
    condition: torch.Tensor,
    hold_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one phase after ``condition`` holds for consecutive steps.

    Only the phase matching ``current`` owns the shared hold counter. Calls that
    check later phases in the same update leave it untouched; otherwise the
    phase-1/phase-2 checks would reset phase 0's progress every step and no
    transition could ever reach ``hold_steps``.
    """
    in_phase = phase == current
    active_counter = torch.where(
        condition,
        hold_counter + 1,
        torch.zeros_like(hold_counter),
    )
    hold_counter = torch.where(in_phase, active_counter, hold_counter)
    advance = in_phase & (hold_counter >= hold_steps)
    phase = torch.where(advance, torch.full_like(phase, current + 1), phase)
    hold_counter = torch.where(advance, torch.zeros_like(hold_counter), hold_counter)
    return phase, hold_counter


def update_task_phase(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lift_z: float = 0.10,
    grasp_dist: float = 0.26,
    release_dist: float = 0.08,
    place_grasp_dist: float = 0.30,
    xy_radius: float = 0.10,
    z_above: float = 0.025,
    z_window: float = 0.075,
    hold_steps: int = 5,
    print_log: bool = False,
) -> torch.Tensor:
    """Advance the monotonic phase machine. Returns current phase tensor.

    Distance thresholds are calibrated from 11 annotated demos (wrist_yaw_link
    -> apple center): a holding hand sits ~0.22m from the apple center and the
    apple is lifted ~0.19m by the handover, so grasp_dist=0.26 and lift_z=0.10.
    At a genuine placement the right hand carries the apple down and stays
    ~0.23m from it (max 0.28), whereas an accidental drop from handover height
    leaves the hand ~0.38m away; ``place_grasp_dist=0.30`` rejects such drops.
    phase 4 requires at least three of four discriminative right-thumb joints
    to indicate an open pose while the apple remains on the plate.
    """
    state = env.pnp_apple_state
    phase = state.task_phase
    old_phase = phase.clone()

    apple_pos, plate_pos = _apple_plate_positions(env, apple_cfg, plate_cfg)
    left_wrist, right_wrist = _wrist_positions(env, robot_cfg)

    left_dist = torch.norm(apple_pos - left_wrist, dim=-1)
    right_dist = torch.norm(apple_pos - right_wrist, dim=-1)

    # phase 0 -> 1: left grasp + lift
    lifted = apple_pos[:, 2] > (state.initial_apple_z + lift_z)
    left_close = left_dist < grasp_dist
    cond_0 = lifted & left_close
    phase, state.phase_hold_counter = _advance_with_hold(
        phase, state.phase_hold_counter, current=0, condition=cond_0, hold_steps=hold_steps
    )
    entering_phase_1 = (old_phase == 0) & (phase >= 1)
    state.left_dist_at_grasp = torch.where(entering_phase_1, left_dist, state.left_dist_at_grasp)

    # phase 1 -> 2: right hand catches while left hand releases, with apple still lifted.
    right_close = right_dist < grasp_dist
    left_released = left_dist > (state.left_dist_at_grasp + release_dist)
    still_lifted = apple_pos[:, 2] > (state.initial_apple_z + lift_z)
    cond_1 = right_close & left_released & still_lifted
    phase, state.phase_hold_counter = _advance_with_hold(
        phase, state.phase_hold_counter, current=1, condition=cond_1, hold_steps=hold_steps
    )

    # phase 2 -> 3: apple on plate AND still controlled by the right hand.
    # Requiring the right wrist to stay near the apple distinguishes a genuine
    # placement (hand carries the apple down, ~0.23m) from an accidental drop
    # during handover (apple free-falls onto the plate while the hand remains
    # up at handover height, ~0.38m away).
    on_plate = _apple_on_plate(apple_pos, plate_pos, xy_radius, z_above, z_window)
    right_controls = right_dist < place_grasp_dist
    cond_2 = on_plate & right_controls
    phase, state.phase_hold_counter = _advance_with_hold(
        phase, state.phase_hold_counter, current=2, condition=cond_2, hold_steps=hold_steps
    )

    # phase 3 -> 4: right thumb opens while the apple remains on the plate.
    # Requiring old_phase == 3 prevents a one-frame 2 -> 3 -> 4 transition.
    right_released = _right_thumb_open(env, robot_cfg)
    cond_3 = (old_phase == 3) & on_plate & right_released
    phase, state.phase_hold_counter = _advance_with_hold(
        phase, state.phase_hold_counter, current=3, condition=cond_3, hold_steps=hold_steps
    )

    if print_log and (phase != old_phase).any():
        for env_id in torch.nonzero(phase != old_phase, as_tuple=False).flatten():
            logger.info(
                "PNP env %d advanced phase %d -> %d",
                int(env_id),
                int(old_phase[env_id]),
                int(phase[env_id]),
            )

    state.task_phase = phase
    return phase


def _sparse_phase_reward(
    env: ManagerBasedRLEnv,
    phase: torch.Tensor,
    previous_attr: str,
    from_phase: int,
) -> torch.Tensor:
    state = env.pnp_apple_state
    previous = getattr(state, previous_attr)
    completed = (previous == from_phase) & (phase >= from_phase + 1)
    reward = torch.where(
        completed,
        torch.ones(env.num_envs, device=env.device) / env.step_dt,
        torch.zeros(env.num_envs, device=env.device),
    )
    setattr(state, previous_attr, phase.clone())
    return reward


@dataclass
class PnpAppleState:
    """Per-env task state for the pick-and-place apple environment.

    phase semantics:
        0 - Reach & left grasp + lift
        1 - Right hand catches apple while left releases
        2 - Place apple on plate
        3 - Release apple while it remains on plate
        4 - Task complete
    """

    task_phase: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_phase_left_grasp: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_phase_handover: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_phase_place: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_phase_release: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    initial_apple_z: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    left_dist_at_grasp: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    phase_hold_counter: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    @classmethod
    def create(cls, num_envs: int, device: str) -> PnpAppleState:
        """Allocate the per-environment buffers.

        Args:
            num_envs: Number of environments in the scene.
            device: Torch device the buffers live on.
        """
        return cls(
            task_phase=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_phase_left_grasp=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_phase_handover=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_phase_place=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_phase_release=torch.zeros(num_envs, dtype=torch.long, device=device),
            initial_apple_z=torch.zeros(num_envs, device=device),
            left_dist_at_grasp=torch.full((num_envs,), float("inf"), device=device),
            phase_hold_counter=torch.zeros(num_envs, dtype=torch.long, device=device),
        )

    # Start-of-episode value for every per-environment field. ``left_dist_at_grasp``
    # starts at infinity so the phase 1 -> 2 release check cannot fire before a grasp.
    EPISODE_RESET_VALUES: ClassVar[dict[str, float]] = {
        "task_phase": 0,
        "prev_phase_left_grasp": 0,
        "prev_phase_handover": 0,
        "prev_phase_place": 0,
        "prev_phase_release": 0,
        "left_dist_at_grasp": float("inf"),
        "phase_hold_counter": 0,
    }
    # Refreshed from the scene by ``reset_task_phase`` once the apple pose is randomized.
    EXTERNALLY_RESET_FIELDS: ClassVar[frozenset[str]] = frozenset({"initial_apple_z"})

    def reset(self, env_ids: torch.Tensor) -> None:
        """Restore per-episode phase bookkeeping for ``env_ids``.

        Args:
            env_ids: Indices of the environments being reset.

        Raises:
            RuntimeError: If a field was added without declaring how it resets.
        """
        declared = set(self.EPISODE_RESET_VALUES) | self.EXTERNALLY_RESET_FIELDS
        undeclared = {f.name for f in fields(self)} - declared
        if undeclared:
            raise RuntimeError(
                f"{type(self).__name__} fields {sorted(undeclared)} are missing from"
                " EPISODE_RESET_VALUES or EXTERNALLY_RESET_FIELDS; they would leak across episodes."
            )
        for name, value in self.EPISODE_RESET_VALUES.items():
            getattr(self, name)[env_ids] = value


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


def left_grasp_lift_reward(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lift_z: float = 0.10,
    grasp_dist: float = 0.26,
    release_dist: float = 0.08,
    place_grasp_dist: float = 0.30,
    xy_radius: float = 0.10,
    z_above: float = 0.025,
    z_window: float = 0.075,
    hold_steps: int = 5,
    print_log: bool = False,
) -> torch.Tensor:
    """Sparse reward for phase 0 -> 1 (left grasp + lift). Also advances phases."""
    phase = update_task_phase(
        env,
        apple_cfg,
        plate_cfg,
        robot_cfg,
        lift_z=lift_z,
        grasp_dist=grasp_dist,
        release_dist=release_dist,
        place_grasp_dist=place_grasp_dist,
        xy_radius=xy_radius,
        z_above=z_above,
        z_window=z_window,
        hold_steps=hold_steps,
        print_log=print_log,
    )
    return _sparse_phase_reward(env, phase, "prev_phase_left_grasp", from_phase=0)


def handover_to_right_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse reward for phase 1 -> 2 (right handover)."""
    phase = env.pnp_apple_state.task_phase
    return _sparse_phase_reward(env, phase, "prev_phase_handover", from_phase=1)


def place_on_plate_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse reward for phase 2 -> 3 (apple placed on plate)."""
    phase = env.pnp_apple_state.task_phase
    return _sparse_phase_reward(env, phase, "prev_phase_place", from_phase=2)


def release_on_plate_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse reward for phase 3 -> 4 (right hand releases apple on plate)."""
    phase = env.pnp_apple_state.task_phase
    return _sparse_phase_reward(env, phase, "prev_phase_release", from_phase=3)
