# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful success contexts shared by stack rewards, curricula, and terminations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import math as math_utils

from .rewards import (
    _gripper_is_released,
    _order_invariant_cube_state,
    order_invariant_stack_progress,
    role_conditioned_stack_potential,
)
from .runtime_state import get_stack_reset_runtime_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


class _StackSuccessContext(ManagerTermBase):
    """Own common current/new/episode success state for context terms."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.is_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.new_success = torch.zeros_like(self.is_success)
        self.ever_success = torch.zeros_like(self.is_success)
        self._no_termination = torch.zeros_like(self.is_success)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear success state for reset environments."""
        if env_ids is None:
            env_ids = slice(None)
        self.is_success[env_ids] = False
        self.new_success[env_ids] = False
        self.ever_success[env_ids] = False

    def _update_success(self, reached: torch.Tensor) -> None:
        """Update current, edge-triggered, and sticky success buffers."""
        self.is_success.copy_(reached)
        self.new_success.copy_(self.is_success & ~self.ever_success)
        self.ever_success |= self.is_success


class StableOrderInvariantStackGoal(_StackSuccessContext):
    """Track stable full-stack success without immediately resetting the scene.

    This non-terminating context exposes current, newly reached, and
    episode-ever success tensors to the reward, reset curriculum, and
    delayed-success termination terms. Returning all false prevents near-goal
    reset rows from producing three-step episodes.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the stability dwell counter for reset environments."""
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._stable_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimum_episode_steps: int = 3,
        hold_steps: int = 5,
        maximum_cube_velocity: float = 0.15,
        minimum_finger_release_position: float = 0.023,
        gripper_cfg: SceneEntityCfg | None = None,
        open_gripper_joint_positions: tuple[float, ...] | None = None,
        closed_gripper_joint_positions: tuple[float, ...] | None = None,
        gripper_finger_joint_counts: tuple[int, int] = (4, 4),
        maximum_gripper_closure: float = 0.2,
        grasp_pair_joint_names: tuple[tuple[str, ...], ...] | None = None,
        grasp_pair_open_joint_positions: tuple[tuple[float, ...], ...] | None = None,
        grasp_pair_closed_joint_positions: tuple[tuple[float, ...], ...] | None = None,
        xy_threshold: float = 0.025,
        height_threshold: float = 0.012,
        cube_height: float = 0.04,
    ) -> torch.Tensor:
        """Update full-task success state and return a non-terminating mask."""
        if (
            minimum_episode_steps < 0
            or hold_steps < 1
            or maximum_cube_velocity <= 0.0
            or minimum_finger_release_position < 0.0
            or xy_threshold <= 0.0
            or height_threshold <= 0.0
            or cube_height <= 0.0
        ):
            raise ValueError("Invalid stable stack-goal thresholds.")
        stack_progress = order_invariant_stack_progress(
            env,
            xy_threshold=xy_threshold,
            height_threshold=height_threshold,
            cube_height=cube_height,
        )
        _, velocities = _order_invariant_cube_state(
            env,
            (
                SceneEntityCfg("cube_1"),
                SceneEntityCfg("cube_2"),
                SceneEntityCfg("cube_3"),
            ),
        )
        slow = torch.amax(torch.abs(velocities[..., :3]), dim=(1, 2)) < maximum_cube_velocity
        robot: Articulation = env.scene["robot"]
        pair_kwargs = {}
        if (
            grasp_pair_joint_names is not None
            or grasp_pair_open_joint_positions is not None
            or grasp_pair_closed_joint_positions is not None
        ):
            pair_kwargs = {
                "grasp_pair_joint_names": grasp_pair_joint_names,
                "grasp_pair_open_joint_positions": grasp_pair_open_joint_positions,
                "grasp_pair_closed_joint_positions": grasp_pair_closed_joint_positions,
            }
        released = _gripper_is_released(
            env,
            robot,
            minimum_finger_position=minimum_finger_release_position,
            gripper_cfg=gripper_cfg,
            open_joint_positions=open_gripper_joint_positions,
            closed_joint_positions=closed_gripper_joint_positions,
            finger_joint_counts=gripper_finger_joint_counts,
            maximum_gripper_closure=maximum_gripper_closure,
            **pair_kwargs,
        )
        stable = (stack_progress >= 2.0) & slow & released & (env.episode_length_buf >= minimum_episode_steps)
        self._stable_steps = torch.where(stable, self._stable_steps + 1, torch.zeros_like(self._stable_steps))
        self._update_success(self._stable_steps >= hold_steps)
        env.extras["successes"] = self.ever_success
        return self._no_termination


def _oriented_box_point_signed_distance(
    points: torch.Tensor,
    centers: torch.Tensor,
    orientations: torch.Tensor,
    half_extent: float,
) -> torch.Tensor:
    """Return point-to-box signed distances for batched oriented cubes.

    Args:
        points: Query positions [m], shape ``(N, P, 3)``.
        centers: Cube centers [m], shape ``(N, C, 3)``.
        orientations: Cube orientations in ``(x, y, z, w)`` order, shape
            ``(N, C, 4)``.
        half_extent: Cube half-edge length [m].

    Returns:
        Signed distances [m], shape ``(N, P, C)``. Positive values are
        outside a cube, zero is on its surface, and negative values are
        inside.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("Points must have shape (N, P, 3).")
    if centers.ndim != 3 or centers.shape[-1] != 3 or centers.shape[0] != points.shape[0]:
        raise ValueError("Centers must have shape (N, C, 3) and share the point batch size.")
    if orientations.shape != centers.shape[:-1] + (4,):
        raise ValueError("Orientations must have shape (N, C, 4).")
    if half_extent <= 0.0:
        raise ValueError("half_extent must be positive.")

    point_offsets = points.unsqueeze(2) - centers.unsqueeze(1)
    expanded_orientations = orientations.unsqueeze(1).expand(-1, points.shape[1], -1, -1)
    local_offsets = math_utils.quat_apply_inverse(expanded_orientations, point_offsets)
    face_offsets = torch.abs(local_offsets) - half_extent
    outside_distance = torch.linalg.vector_norm(torch.clamp_min(face_offsets, 0.0), dim=-1)
    inside_distance = torch.clamp_max(torch.amax(face_offsets, dim=-1), 0.0)
    return outside_distance + inside_distance


class StableFullHandOrderInvariantStackGoal(_StackSuccessContext):
    """Track physically released full-hand stack success.

    Unlike a parallel jaw or a fixed two-finger synergy, a fully actuated
    hand has no single joint posture that proves object release. This context
    therefore requires every configured fingertip origin to clear every
    oriented cube by a physical margin. It also checks both linear and angular
    cube speed before accumulating the success dwell.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the stability dwell counter for reset environments."""
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._stable_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimum_episode_steps: int = 3,
        hold_steps: int = 5,
        maximum_cube_linear_velocity: float = 0.10,
        maximum_cube_angular_velocity: float = 1.0,
        minimum_fingertip_cube_clearance: float = 0.010,
        fingertip_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        cube_cfgs: tuple[SceneEntityCfg, ...] = (
            SceneEntityCfg("cube_1"),
            SceneEntityCfg("cube_2"),
            SceneEntityCfg("cube_3"),
        ),
        xy_threshold: float = 0.025,
        height_threshold: float = 0.012,
        cube_height: float = 0.08,
    ) -> torch.Tensor:
        """Update full-task success state and return a non-terminating mask."""
        if (
            minimum_episode_steps < 0
            or hold_steps < 1
            or maximum_cube_linear_velocity <= 0.0
            or maximum_cube_angular_velocity <= 0.0
            or minimum_fingertip_cube_clearance < 0.0
            or min(xy_threshold, height_threshold, cube_height) <= 0.0
        ):
            raise ValueError("Invalid stable full-hand stack-goal thresholds.")
        if fingertip_cfg.body_names is None:
            raise ValueError("fingertip_cfg must select at least one fingertip body.")

        stack_progress = order_invariant_stack_progress(
            env,
            cube_cfgs=cube_cfgs,
            xy_threshold=xy_threshold,
            height_threshold=height_threshold,
            cube_height=cube_height,
        )
        _, velocities = _order_invariant_cube_state(env, cube_cfgs)
        slow_linear = torch.amax(torch.linalg.vector_norm(velocities[..., :3], dim=-1), dim=1)
        slow_angular = torch.amax(torch.linalg.vector_norm(velocities[..., 3:], dim=-1), dim=1)
        cubes_are_slow = (slow_linear < maximum_cube_linear_velocity) & (slow_angular < maximum_cube_angular_velocity)

        robot: Articulation = env.scene[fingertip_cfg.name]
        fingertip_positions = robot.data.body_pos_w.torch[:, fingertip_cfg.body_ids]
        cube_positions = torch.stack(tuple(env.scene[cfg.name].data.root_pos_w.torch for cfg in cube_cfgs), dim=1)
        cube_orientations = torch.stack(
            tuple(env.scene[cfg.name].data.root_quat_w.torch for cfg in cube_cfgs),
            dim=1,
        )
        fingertip_clearances = _oriented_box_point_signed_distance(
            fingertip_positions,
            cube_positions,
            cube_orientations,
            half_extent=0.5 * cube_height,
        )
        hand_is_clear = torch.amin(fingertip_clearances, dim=(1, 2)) >= minimum_fingertip_cube_clearance

        stable = (
            (stack_progress >= 2.0) & cubes_are_slow & hand_is_clear & (env.episode_length_buf >= minimum_episode_steps)
        )
        self._stable_steps = torch.where(stable, self._stable_steps + 1, torch.zeros_like(self._stable_steps))
        self._update_success(self._stable_steps >= hold_steps)
        env.extras["successes"] = self.ever_success
        return self._no_termination


class StackResetLearningProgress(_StackSuccessContext):
    """Track meaningful forward progress from every sampled reset row.

    This is curriculum evidence only: it never terminates an episode. Every
    rollout remains free to continue toward the strict released full-stack
    goal, while early table/pick rows can still become competent before the
    complete multi-pick sequence is learned.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._initial_potential = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._target_potential = torch.ones_like(self._initial_potential)
        potential_param_names = (
            "robot_cfg",
            "tool_body_name",
            "tool_offset",
            "grasp_pair_tool_offsets",
            "gripper_cfg",
            "open_gripper_joint_positions",
            "closed_gripper_joint_positions",
            "gripper_finger_joint_counts",
            "grasp_pair_joint_names",
            "grasp_pair_open_joint_positions",
            "grasp_pair_closed_joint_positions",
            "minimum_gripper_closure",
            "maximum_gripper_closure",
            "xy_threshold",
            "height_threshold",
            "cube_height",
        )
        self._potential_kwargs = {name: cfg.params[name] for name in potential_param_names if name in cfg.params}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Set a non-free target from the newly sampled physical row."""
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        initial = role_conditioned_stack_potential(
            self._env,
            **getattr(self, "_potential_kwargs", {}),
        )
        self._initial_potential[env_ids] = initial[env_ids]
        reset_state = get_stack_reset_runtime_state(self._env)
        row_target = reset_state.target_potentials[env_ids]
        self._target_potential[env_ids] = torch.maximum(
            row_target,
            initial[env_ids] + 0.25,
        ).clamp_max(10.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimum_episode_steps: int = 3,
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
        xy_threshold: float = 0.025,
        height_threshold: float = 0.012,
        cube_height: float = 0.04,
    ) -> torch.Tensor:
        """Update sticky curriculum success and return a non-terminating mask."""
        if minimum_episode_steps < 0 or min(xy_threshold, height_threshold, cube_height) <= 0.0:
            raise ValueError("Invalid reset-learning progress thresholds.")
        current = role_conditioned_stack_potential(
            env,
            robot_cfg=robot_cfg,
            tool_body_name=tool_body_name,
            tool_offset=tool_offset,
            grasp_pair_tool_offsets=grasp_pair_tool_offsets,
            gripper_cfg=gripper_cfg,
            open_gripper_joint_positions=open_gripper_joint_positions,
            closed_gripper_joint_positions=closed_gripper_joint_positions,
            gripper_finger_joint_counts=gripper_finger_joint_counts,
            grasp_pair_joint_names=grasp_pair_joint_names,
            grasp_pair_open_joint_positions=grasp_pair_open_joint_positions,
            grasp_pair_closed_joint_positions=grasp_pair_closed_joint_positions,
            minimum_gripper_closure=minimum_gripper_closure,
            maximum_gripper_closure=maximum_gripper_closure,
            xy_threshold=xy_threshold,
            height_threshold=height_threshold,
            cube_height=cube_height,
        )
        reached = (current >= self._target_potential) & (env.episode_length_buf >= minimum_episode_steps)
        self._update_success(reached)
        return self._no_termination
