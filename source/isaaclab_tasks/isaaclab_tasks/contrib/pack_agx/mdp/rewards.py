# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Monotonic phase tracking and sparse rewards for Pack-AGX RL."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, ClassVar

import torch

from isaaclab.managers import SceneEntityCfg

from .terminations import agx_in_box_from_pose, agx_is_horizontal_from_quat

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
    """Advance the lift -> align -> seat -> release phase machine.

    Position targets are calibrated from replay episode 21. Requiring consecutive
    simulation steps prevents a one-frame contact impulse from earning a phase
    reward. Release requires the AGX to remain seated, the right thumb to open,
    and the right wrist to retreat.
    """
    state = env.pack_agx_state
    phase = state.task_phase
    old_phase = phase.clone()

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
    phase, state.phase_hold_counter = _advance_with_hold(
        phase,
        state.phase_hold_counter,
        current=0,
        condition=lifted,
        hold_steps=lift_hold_steps,
    )

    aligned_while_lifted = (old_phase == 1) & lifted & (align_target_distance <= align_xy)
    phase, state.phase_hold_counter = _advance_with_hold(
        phase,
        state.phase_hold_counter,
        current=1,
        condition=aligned_while_lifted,
        hold_steps=align_hold_steps,
    )

    in_box = agx_in_box_from_pose(
        agx_pos,
        agx_quat,
        box_pos,
    )
    seated = (old_phase == 2) & in_box
    phase, state.phase_hold_counter = _advance_with_hold(
        phase,
        state.phase_hold_counter,
        current=2,
        condition=seated,
        hold_steps=seat_hold_steps,
    )

    right_wrist_distance = torch.linalg.vector_norm(
        right_wrist_pos - agx_pos,
        dim=-1,
    )
    released = (
        (old_phase == 3)
        & in_box
        & (right_thumb_pos <= 0.261799)  # 15 degrees
        & (right_wrist_distance >= 0.32)
    )
    phase, state.phase_hold_counter = _advance_with_hold(
        phase,
        state.phase_hold_counter,
        current=3,
        condition=released,
        hold_steps=seat_hold_steps,
    )

    if print_log and (phase != old_phase).any():
        for env_id in torch.nonzero(phase != old_phase, as_tuple=False).flatten():
            logger.info(
                "Pack-AGX env %d advanced phase %d -> %d",
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
    state = env.pack_agx_state
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
class PackAgxState:
    """Per-env task state for the Pack-AGX environment.

    phase semantics:
        0 - Lift the AGX off the table while keeping it horizontal
        1 - Align the AGX over the protective box
        2 - Seat the AGX inside the box
        3 - Release: right thumb opens and the wrist retreats
        4 - Task complete
    """

    task_phase: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_phase_lift: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_phase_align: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_phase_seat: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    prev_phase_release: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    initial_agx_z: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    phase_hold_counter: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    @classmethod
    def create(cls, num_envs: int, device: str) -> PackAgxState:
        """Allocate the per-environment buffers.

        Args:
            num_envs: Number of environments in the scene.
            device: Torch device the buffers live on.
        """
        return cls(
            task_phase=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_phase_lift=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_phase_align=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_phase_seat=torch.zeros(num_envs, dtype=torch.long, device=device),
            prev_phase_release=torch.zeros(num_envs, dtype=torch.long, device=device),
            initial_agx_z=torch.zeros(num_envs, device=device),
            phase_hold_counter=torch.zeros(num_envs, dtype=torch.long, device=device),
        )

    # Start-of-episode value for every per-environment field.
    EPISODE_RESET_VALUES: ClassVar[dict[str, float]] = {
        "task_phase": 0,
        "prev_phase_lift": 0,
        "prev_phase_align": 0,
        "prev_phase_seat": 0,
        "prev_phase_release": 0,
        "phase_hold_counter": 0,
    }
    # Refreshed from the scene by ``reset_task_phase`` once the AGX pose is randomized.
    EXTERNALLY_RESET_FIELDS: ClassVar[frozenset[str]] = frozenset({"initial_agx_z"})

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
    """Reward the first stable lift and update all task phases."""
    phase = update_task_phase(
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
    return _sparse_phase_reward(env, phase, "prev_phase_lift", from_phase=0)


def align_agx_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse reward for phase 1 -> 2 (AGX aligned over the protective box)."""
    phase = env.pack_agx_state.task_phase
    return _sparse_phase_reward(env, phase, "prev_phase_align", from_phase=1)

    # Dense progress shaping during phase 1, disabled in favour of pure sparse
    # phase rewards. The continuous term was ``distance_offset - distance``,
    # zeroed at the typical phase-entry distance (~0.25 m from the randomized
    # initial poses) so it stayed positive whenever the AGX was closer to the
    # align target than at entry.
    #
    # def align_agx_reward(
    #     env: ManagerBasedRLEnv,
    #     agx_orin_cfg: SceneEntityCfg = SceneEntityCfg("agx_orin"),
    #     protective_box_cfg: SceneEntityCfg = SceneEntityCfg("protective_box"),
    #     distance_offset: float = 0.25,
    # ) -> torch.Tensor:
    #     state = env.pack_agx_state
    #     reward = _sparse_phase_reward(env, state.task_phase, "prev_phase_align", from_phase=1)
    #     distance = _position_distance(
    #         env.scene[agx_orin_cfg.name].data.root_pos_w.torch,
    #         env.scene[protective_box_cfg.name].data.root_pos_w.torch,
    #         0.10,
    #     )
    #     return reward + torch.where(
    #         state.task_phase == 1,
    #         distance_offset - distance,
    #         torch.zeros_like(distance),
    #     )


def seat_agx_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse reward for phase 2 -> 3 (AGX seated in the protective box)."""
    phase = env.pack_agx_state.task_phase
    return _sparse_phase_reward(env, phase, "prev_phase_seat", from_phase=2)

    # Dense progress shaping during phase 2, disabled in favour of pure sparse
    # phase rewards. The continuous term was ``target_offset - (distance +
    # yaw_distance)``, zeroed at the typical phase-entry error (~0.12 m position
    # + ~0.1 rad yaw). This function also used to award the 3 -> 4 release
    # transition, which release_agx_reward now covers as its own term.
    #
    # def seat_agx_reward(
    #     env: ManagerBasedRLEnv,
    #     agx_orin_cfg: SceneEntityCfg = SceneEntityCfg("agx_orin"),
    #     protective_box_cfg: SceneEntityCfg = SceneEntityCfg("protective_box"),
    #     target_offset: float = 0.25,
    # ) -> torch.Tensor:
    #     state = env.pack_agx_state
    #     previous = state.prev_phase_seat
    #     completed = ((previous == 2) & (state.task_phase >= 3)) | ((previous == 3) & (state.task_phase >= 4))
    #     reward = torch.where(
    #         completed,
    #         torch.ones(env.num_envs, device=env.device) / env.step_dt,
    #         torch.zeros(env.num_envs, device=env.device),
    #     )
    #     state.prev_phase_seat = state.task_phase.clone()
    #     agx_orin = env.scene[agx_orin_cfg.name]
    #     protective_box = env.scene[protective_box_cfg.name]
    #     distance = _position_distance(
    #         agx_orin.data.root_pos_w.torch,
    #         protective_box.data.root_pos_w.torch,
    #         -0.01945,
    #     )
    #     _, _, agx_yaw = euler_xyz_from_quat(agx_orin.data.root_quat_w.torch)
    #     _, _, box_yaw = euler_xyz_from_quat(protective_box.data.root_quat_w.torch)
    #     yaw_delta = agx_yaw - box_yaw
    #     yaw_distance = torch.abs(torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)))
    #     return reward + torch.where(
    #         state.task_phase == 2,
    #         target_offset - (distance + yaw_distance),
    #         torch.zeros_like(distance),
    #     )


def release_agx_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse reward for phase 3 -> 4 (right hand releases the seated AGX)."""
    phase = env.pack_agx_state.task_phase
    return _sparse_phase_reward(env, phase, "prev_phase_release", from_phase=3)
