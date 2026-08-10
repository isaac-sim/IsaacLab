# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for conveyor transfer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from ..conveyor_geometry import (
    BELT_CENTER_X,
    BELT_HALF_STRAIGHT,
    BELT_TURN_RADIUS,
    BELT_WIDTH,
    GUARD_THICKNESS,
)
from .kinematics import end_effector_pose
from .reset_events import CUBE_COUNT, CUBE_SIZE, ConveyorResetRecipe, side_inner_y
from .rewards import current_transfer_potential, physical_cube_acquisition_mask

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


_TRACK_X_CLEARANCE = BELT_TURN_RADIUS + 0.5 * BELT_WIDTH + GUARD_THICKNESS + CUBE_SIZE


def transfer_success_mask(
    cube_positions: torch.Tensor,
    cube_linear_velocities: torch.Tensor,
    tool_positions: torch.Tensor,
    finger_positions: torch.Tensor,
    target_side_ids: torch.Tensor,
    lateral_tolerance: float = 0.055,
    maximum_cube_speed: float = 0.65,
    minimum_finger_position: float = 0.027,
    minimum_tool_clearance: float = 0.055,
) -> torch.Tensor:
    """Return whether the active cube is released on its destination belt."""
    target_y = side_inner_y(target_side_ids)
    on_straight = torch.abs(cube_positions[:, 0] - BELT_CENTER_X) < BELT_HALF_STRAIGHT
    on_lane = torch.abs(cube_positions[:, 1] - target_y) < lateral_tolerance
    supported_height = (cube_positions[:, 2] > 0.045) & (cube_positions[:, 2] < 0.095)
    moving_safely = torch.linalg.vector_norm(cube_linear_velocities, dim=1) < maximum_cube_speed
    released = torch.amin(finger_positions, dim=1) > minimum_finger_position
    hand_clear = torch.linalg.vector_norm(tool_positions - cube_positions, dim=1) > minimum_tool_clearance
    return on_straight & on_lane & supported_height & moving_safely & released & hand_clear


class StableConveyorTransfer(ManagerTermBase):
    """Track stable released placements without terminating the episode."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.is_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.new_success = torch.zeros_like(self.is_success)
        self.pending_success = torch.zeros_like(self.is_success)
        self.ever_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._no_termination = torch.zeros_like(self.is_success)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimum_episode_steps: int = 2,
        hold_steps: int = 3,
        lateral_tolerance: float = 0.055,
        maximum_cube_speed: float = 0.65,
        minimum_finger_position: float = 0.027,
        minimum_tool_clearance: float = 0.055,
    ) -> torch.Tensor:
        """Update current, edge-triggered, and sticky transfer success state."""
        if minimum_episode_steps < 0 or hold_steps < 1:
            raise ValueError("minimum_episode_steps must be non-negative and hold_steps must be positive.")
        state = env.conveyor_transfer_state
        cubes: tuple[RigidObject, ...] = tuple(env.scene[f"cube_{cube_id}"] for cube_id in range(CUBE_COUNT))
        positions = torch.stack(tuple(cube.data.root_pos_w.torch for cube in cubes), dim=1)
        velocities = torch.stack(tuple(cube.data.root_lin_vel_w.torch for cube in cubes), dim=1)
        index = state.target_cube_ids.view(env.num_envs, 1, 1).expand(-1, 1, 3)
        active_position = torch.gather(positions, 1, index).squeeze(1) - env.scene.env_origins
        active_velocity = torch.gather(velocities, 1, index).squeeze(1)
        tool_position, _ = end_effector_pose(env)
        tool_position = tool_position - env.scene.env_origins
        robot: Articulation = env.scene["robot"]
        finger_ids, _ = robot.find_joints("panda_finger_joint[1-2]", preserve_order=True)
        finger_positions = robot.data.joint_pos.torch[:, finger_ids]
        successful = transfer_success_mask(
            active_position,
            active_velocity,
            tool_position,
            finger_positions,
            1 - state.source_side_ids,
            lateral_tolerance=lateral_tolerance,
            maximum_cube_speed=maximum_cube_speed,
            minimum_finger_position=minimum_finger_position,
            minimum_tool_clearance=minimum_tool_clearance,
        )
        subgoal_steps = env.episode_length_buf - state.subgoal_start_steps
        successful &= subgoal_steps >= minimum_episode_steps
        self._stable_steps = torch.where(successful, self._stable_steps + 1, torch.zeros_like(self._stable_steps))
        stable = self._stable_steps >= hold_steps
        self.new_success.copy_(stable & ~self.is_success & ~self.pending_success)
        self.is_success.copy_(stable)
        self.pending_success |= self.new_success
        self.ever_success |= self.new_success
        env.extras["successes"] = self.ever_success
        return self._no_termination

    def consume_success(self, env_ids: Sequence[int]) -> None:
        """Clear per-subgoal success state after the goal-transition event."""
        self._stable_steps[env_ids] = 0
        self.is_success[env_ids] = False
        self.new_success[env_ids] = False
        self.pending_success[env_ids] = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Log episode success and clear history for selected environments."""
        successes = self.ever_success if env_ids is None else self.ever_success[env_ids]
        if successes.numel() > 0:
            self._env.extras.setdefault("log", {})["Metrics/success_rate"] = successes.float().mean().item()

        reset_ids = slice(None) if env_ids is None else env_ids
        self._stable_steps[reset_ids] = 0
        self.is_success[reset_ids] = False
        self.new_success[reset_ids] = False
        self.pending_success[reset_ids] = False
        self.ever_success[reset_ids] = False


def subgoal_time_out(env: ManagerBasedRLEnv, timeout_s: float = 20.0) -> torch.Tensor:
    """Truncate environments that make no transfer within one subgoal timeout [s]."""
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive.")
    timeout_steps = math.ceil(timeout_s / env.step_dt)
    state = env.conveyor_transfer_state
    return env.episode_length_buf - state.subgoal_start_steps >= timeout_steps


def transfer_sequence_time_out(env: ManagerBasedRLEnv, maximum_transfers: int = 8) -> torch.Tensor:
    """Truncate long successful sequences so reset coverage remains fresh."""
    if maximum_transfers < 1:
        raise ValueError("maximum_transfers must be positive.")
    return env.conveyor_transfer_state.transfer_counts >= maximum_transfers


class ConveyorResetLearningProgress(ManagerTermBase):
    """Track row-relative progress without terminating the episode.

    The adaptive reset sampler needs useful evidence before complete transfers
    are common. Each reset row therefore asks the policy to increase the same
    transfer potential used for dense shaping by a fixed amount. Episodes keep
    running toward strict released placement; this context only records which
    rows have advanced meaningfully.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._target_potential = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self.is_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.new_success = torch.zeros_like(self.is_success)
        self.ever_success = torch.zeros_like(self.is_success)
        self._no_termination = torch.zeros_like(self.is_success)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimum_episode_steps: int = 3,
        minimum_progress: float = 0.35,
        maximum_target_potential: float = 5.0,
        minimum_acquisition_lift: float = 0.025,
        maximum_acquisition_tool_distance: float = 0.075,
        maximum_acquisition_finger_position: float = 0.030,
    ) -> torch.Tensor:
        """Update sticky row-progress evidence and return an all-false mask."""
        if (
            minimum_episode_steps < 0
            or minimum_progress <= 0.0
            or maximum_target_potential <= 0.0
            or minimum_acquisition_lift <= 0.0
            or maximum_acquisition_tool_distance <= 0.0
            or maximum_acquisition_finger_position <= 0.0
        ):
            raise ValueError("Invalid conveyor reset-learning progress thresholds.")
        current = current_transfer_potential(env)
        reached = (current >= self._target_potential) & (env.episode_length_buf >= minimum_episode_steps)
        state = env.conveyor_transfer_state
        acquisition_recipe = (
            (state.recipe_ids == int(ConveyorResetRecipe.GRASP))
            | (state.recipe_ids == int(ConveyorResetRecipe.PREGRASP))
            | (state.recipe_ids == int(ConveyorResetRecipe.BELT))
        )
        physically_acquired = physical_cube_acquisition_mask(
            env,
            minimum_lift=minimum_acquisition_lift,
            maximum_tool_distance=maximum_acquisition_tool_distance,
            maximum_finger_position=maximum_acquisition_finger_position,
        )
        reached &= ~acquisition_recipe | physically_acquired
        self.is_success.copy_(reached)
        self.new_success.copy_(reached & ~self.ever_success)
        self.ever_success |= reached
        return self._no_termination

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Set a meaningful potential target from each newly sampled row."""
        if env_ids is None:
            env_ids = slice(None)
        initial = current_transfer_potential(self._env)
        minimum_progress = float(self.cfg.params.get("minimum_progress", 0.35))
        maximum_target = float(self.cfg.params.get("maximum_target_potential", 5.0))
        self._target_potential[env_ids] = torch.clamp_max(initial[env_ids] + minimum_progress, maximum_target)
        self.is_success[env_ids] = False
        self.new_success[env_ids] = False
        self.ever_success[env_ids] = False


def cube_out_of_workspace(
    env: ManagerBasedRLEnv,
    minimum: tuple[float, float, float] = (
        BELT_CENTER_X - BELT_HALF_STRAIGHT - _TRACK_X_CLEARANCE,
        -1.05,
        -0.05,
    ),
    maximum: tuple[float, float, float] = (
        BELT_CENTER_X + BELT_HALF_STRAIGHT + _TRACK_X_CLEARANCE,
        1.05,
        0.80,
    ),
) -> torch.Tensor:
    """Terminate when any cube leaves the complete guarded racetrack workspace."""
    cubes: tuple[RigidObject, ...] = tuple(env.scene[f"cube_{cube_id}"] for cube_id in range(CUBE_COUNT))
    positions = torch.stack(tuple(cube.data.root_pos_w.torch for cube in cubes), dim=1)
    positions -= env.scene.env_origins.unsqueeze(1)
    lower = positions.new_tensor(minimum)
    upper = positions.new_tensor(maximum)
    return torch.any((positions < lower) | (positions > upper), dim=(1, 2))


def nonfinite_scene_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate environments containing nonfinite robot or cube state."""
    robot: Articulation = env.scene[robot_cfg.name]
    invalid = ~torch.all(torch.isfinite(robot.data.joint_pos.torch), dim=1)
    invalid |= ~torch.all(torch.isfinite(robot.data.joint_vel.torch), dim=1)
    for cube_id in range(CUBE_COUNT):
        cube: RigidObject = env.scene[f"cube_{cube_id}"]
        invalid |= ~torch.all(torch.isfinite(cube.data.root_state_w.torch), dim=1)
    return invalid
