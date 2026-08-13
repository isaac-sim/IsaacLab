# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-dataset progress and outcome rewards for pouring MPM media."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg
from isaaclab.utils import math as math_utils

from .observations import _nearest_grasp_to_tcp_quat

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


def terminal_failure(env: FrankaPourEnv, include_time_out: bool = True) -> torch.Tensor:
    """Return one unit-integral pulse for an unsuccessful terminal transition."""
    success = _success_terminal(env)
    completed = env.termination_manager.dones if include_time_out else env.termination_manager.terminated
    failed = completed & ~success
    # RewardManager multiplies terms by step_dt. Dividing here makes the configured weight the
    # exact one-time episode penalty, independent of the control frequency.
    return failed.float() / max(float(env.step_dt), 1.0e-6)


def _open_quality(env: FrankaPourEnv) -> torch.Tensor:
    travel = max(env._gripper_open_width - env._gripper_grasp_width, 1.0e-4)
    return torch.clamp((env.gripper_width() - env._gripper_grasp_width) / travel, 0.0, 1.0)


def tcp_cup_grasp_pose_tanh(
    env: FrankaPourEnv,
    position_std: float = 0.20,
    orientation_std: float = 0.75,
    open_hand_fraction: float = 0.35,
) -> torch.Tensor:
    """Return persistent grasp-pose quality with recoverable open-hand coordination.

    The nearest of the source cup's four equivalent horizontal side grasps supplies the
    orientation error. The broad position kernel remains informative throughout the workspace,
    while the open-hand bonus vanishes at the grasp pose so closing on the cup is never penalized.

    Args:
        env: Franka pouring environment.
        position_std: Tanh scale for TCP-to-grasp position error [m].
        orientation_std: Tanh scale for grasp orientation error [rad].
        open_hand_fraction: Maximum open-hand contribution away from the grasp pose.
    """
    distance = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - env.cup_grasp_point_e(), dim=-1)
    position_quality = 1.0 - torch.tanh(distance / float(position_std))

    error_quat = _nearest_grasp_to_tcp_quat(env)
    orientation_error = torch.linalg.vector_norm(math_utils.axis_angle_from_quat(error_quat), dim=-1)
    orientation_quality = 1.0 - torch.tanh(orientation_error / float(orientation_std))

    # Retain half of the position contribution when orientation quality is zero.
    pose_quality = position_quality * (0.5 + 0.5 * orientation_quality)
    # Weight the open-hand term by 1 - pose_quality so it vanishes at the grasp pose.
    potential = pose_quality + float(open_hand_fraction) * (1.0 - pose_quality) * _open_quality(env)
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _grasp_width_quality(env: FrankaPourEnv, preload_position: float) -> torch.Tensor:
    """Return grasp quality requiring commanded preload and non-empty physical closure."""
    width = env.gripper_width()
    open_width = env._gripper_open_width
    grasp_width = env._gripper_grasp_width
    travel = max(open_width - grasp_width, 1.0e-4)
    close_progress = torch.clamp((open_width - width) / travel, 0.0, 1.0)
    open_position = 0.5 * open_width
    preload_travel = max(open_position - float(preload_position), 1.0e-4)
    gripper = env.action_manager.get_term("gripper_action")
    command = gripper.commanded_position[:, 0]
    commanded_preload = torch.clamp((open_position - command) / preload_travel, 0.0, 1.0)
    # The minimum requires both physical closure and commanded preload; either zero input yields zero.
    return torch.minimum(close_progress, commanded_preload) * gripper.contact_quality


def _grasp_lift_potential(
    env: FrankaPourEnv,
    target_height: float,
    grasp_reach_std: float,
    grasp_preload_position: float,
    grasp_fraction: float,
) -> torch.Tensor:
    """Return bounded contact-qualified grasp and lift completion in ``[0, 1]``."""
    distance = torch.linalg.vector_norm(env.tcp_pose_e()[:, :3] - env.cup_grasp_point_e(), dim=-1)
    proximity = torch.clamp(1.0 - distance / float(grasp_reach_std), 0.0, 1.0)
    # Smoothstep has compact support: proximity is zero outside grasp_reach_std.
    proximity = proximity.square() * (3.0 - 2.0 * proximity)
    held = proximity * _grasp_width_quality(env, grasp_preload_position)
    height = torch.clamp(
        (env.cup_pose_e()[:, 2] - env._cup_reset_height) / float(target_height),
        0.0,
        1.0,
    )
    potential = held * (float(grasp_fraction) + (1.0 - float(grasp_fraction)) * height)
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _reset_dataset_physical_potential(
    env: FrankaPourEnv,
    approach_position_std: float,
    approach_orientation_std: float,
    approach_open_hand_fraction: float,
    grasp_target_height: float,
    grasp_reach_std: float,
    grasp_preload_position: float,
    grasp_fraction: float,
    source_mouth_height: float,
    target_rim_height: float,
    target_clearance_height: float,
    alignment_std: float,
    tilt_alignment_radius: float,
    target_tilt: float,
) -> torch.Tensor:
    """Return one label-free physical potential spanning the reset dataset."""
    approach = tcp_cup_grasp_pose_tanh(
        env,
        position_std=approach_position_std,
        orientation_std=approach_orientation_std,
        open_hand_fraction=approach_open_hand_fraction,
    )
    grasp_lift = _grasp_lift_potential(
        env,
        target_height=grasp_target_height,
        grasp_reach_std=grasp_reach_std,
        grasp_preload_position=grasp_preload_position,
        grasp_fraction=grasp_fraction,
    )
    source_pose = env.cup_pose_e()
    target_pose = env.target_pose_e()
    local_mouth = torch.zeros_like(source_pose[:, :3])
    local_mouth[:, 2] = float(source_mouth_height)
    mouth_position = source_pose[:, :3] + math_utils.quat_apply(source_pose[:, 3:7], local_mouth)
    target_rim_position = target_pose[:, :3].clone()
    target_rim_position[:, 2] += float(target_rim_height)
    target_offset_xy = target_rim_position[:, :2] - mouth_position[:, :2]
    target_distance_xy = torch.linalg.vector_norm(target_offset_xy, dim=-1)
    clearance = mouth_position[:, 2] - target_rim_position[:, 2]
    clearance_quality = torch.clamp(clearance / float(target_clearance_height), 0.0, 1.0)
    alignment_quality = 1.0 - torch.tanh(target_distance_xy / float(alignment_std))
    alignment = grasp_lift * clearance_quality * alignment_quality

    local_open_axis = torch.zeros_like(source_pose[:, :3])
    local_open_axis[:, 2] = 1.0
    open_axis = math_utils.quat_apply(source_pose[:, 3:7], local_open_axis)
    tilt_angle = torch.acos(torch.clamp(open_axis[:, 2], -1.0, 1.0))
    tilt_quality = torch.clamp(tilt_angle / float(target_tilt), 0.0, 1.0)
    proximity = torch.clamp(1.0 - target_distance_xy / float(tilt_alignment_radius), 0.0, 1.0)
    tilt_alignment = proximity.square() * (3.0 - 2.0 * proximity)
    tilt = grasp_lift * clearance_quality * tilt_alignment * tilt_quality

    # These coefficients form one bounded potential rather than four independently collectable
    # state rewards. Discounted differences therefore repay reversals and cannot be accumulated by
    # holding or cycling through a locally favorable pose.
    potential = 0.20 * approach + 0.30 * grasp_lift + 0.25 * alignment + 0.25 * tilt
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


class PourResetLearningProgress(ManagerTermBase):
    """Track meaningful local progress without contributing policy reward.

    The flat cache's ``difficulty`` coordinate identifies the next semantic milestone along the
    reset manifold. A separate coordinate measured from live physical state verifies that the policy
    advances by at least ``minimum_progress`` from the state that was actually restored. The semantic
    target therefore cannot turn simulator settling or a row authored just below a milestone into
    free curriculum success.

    Grasp progress is contact-qualified: commanding the hand closed near the cup is insufficient
    without bilateral finger contact. A final-milestone target can only be satisfied by real
    particle-transfer success. The sticky result is curriculum evidence only; this termination term
    always returns false.
    """

    _milestones = (0.20, 0.32, 0.50, 0.75, 0.95, 1.00)

    def __init__(self, cfg: TerminationTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        self._target_potential = torch.ones(env.num_envs, device=env.device)
        self._requires_terminal_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.is_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.new_success = torch.zeros_like(self.is_success)
        self.ever_success = torch.zeros_like(self.is_success)
        self._no_termination = torch.zeros_like(self.is_success)
        # The curriculum runs before manager reset, so it observes the completed episode's sticky
        # value. Manager reset then clears this tensor after the next row has been restored.

    @staticmethod
    def _potential_params(params: Mapping[str, object]) -> dict[str, object]:
        params = dict(params)
        params.pop("minimum_progress", None)
        params.pop("minimum_episode_steps", None)
        # Ignore legacy capture-intent keys; progress has always required physical contact.
        params.pop("capture_orientation_tolerance", None)
        params.pop("capture_intent_gain", None)
        return params

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Baseline newly restored worlds and choose a local, non-free advancement target."""
        if env_ids is None:
            env_ids = slice(None)
        params = dict(self.cfg.params or {})
        minimum_progress = float(params.get("minimum_progress", 0.08))
        potential_params = params.get("potential_params")

        initial = _reset_dataset_physical_potential(
            self._env,
            **self._potential_params(potential_params),
        )[env_ids]
        rows = self._env.reset_dataset_row_id[env_ids]
        row_progress = self._env._reset_dataset_states["difficulty"][rows]
        milestones = initial.new_tensor(self._milestones)
        # ``difficulty`` is the analytic, globally ordered reset-manifold coordinate. The first
        # strictly later bin supplies the semantic goal, while the measured-state floor guarantees
        # a real minimum advancement even when analytic and physical coordinates do not coincide.
        next_milestone_index = torch.searchsorted(milestones, row_progress + 1.0e-6)
        next_milestone_index.clamp_(max=milestones.numel() - 1)
        next_milestone = milestones[next_milestone_index]
        target = torch.maximum(initial + minimum_progress, next_milestone).clamp_(max=1.0)

        self._target_potential[env_ids] = target
        self._requires_terminal_success[env_ids] = target >= 1.0 - 1.0e-6
        self.is_success[env_ids] = False
        self.new_success[env_ids] = False
        self.ever_success[env_ids] = False

    def __call__(
        self,
        env: FrankaPourEnv,
        potential_params: Mapping[str, object],
        minimum_progress: float = 0.08,
        minimum_episode_steps: int = 3,
    ) -> torch.Tensor:
        """Latch local advancement and return a non-terminating mask."""
        current = _reset_dataset_physical_potential(env, **self._potential_params(potential_params))
        reached_potential = (
            (current >= self._target_potential)
            & ~self._requires_terminal_success
            & (env.episode_length_buf >= minimum_episode_steps)
        )
        # Failure terms run before this context. A transition that reaches a milestone while also
        # spilling, leaving bounds, colliding, or becoming non-finite is not useful curriculum
        # evidence. Terminal particle transfer remains valid because the success term explicitly
        # masks earlier failures and sets ``episode_succeeded`` before this term runs.
        unsafe_terminal = env.termination_manager.terminated & ~env.episode_succeeded
        reached_potential &= ~unsafe_terminal
        # A real transfer can terminate on the first post-reset physics step. The cache contract
        # forbids success at reset time, so delaying this signal would mislabel the easiest terminal
        # rows as failures before they can reach ``minimum_episode_steps``.
        reached = reached_potential | env.episode_succeeded
        self.is_success.copy_(reached)
        self.new_success.copy_(reached & ~self.ever_success)
        self.ever_success |= reached
        return self._no_termination


def _success_terminal(env: FrankaPourEnv) -> torch.Tensor:
    """Return this step's managed success, or false when a replay tool removed that term."""
    if "success" not in env.termination_manager.active_terms:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return env.termination_manager.get_term("success")


def pour_success_bonus(env: FrankaPourEnv) -> torch.Tensor:
    """Reward exactly the managed success termination."""
    success = _success_terminal(env)
    return success.float() / max(float(env.step_dt), 1.0e-6)
