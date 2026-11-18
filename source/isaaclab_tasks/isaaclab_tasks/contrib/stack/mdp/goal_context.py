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

from .rewards import (
    _resolve_finger_joint_ids,
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
        self._reported_success = torch.zeros_like(self.ever_success)
        self._finger_joint_ids = _resolve_finger_joint_ids(env, env.scene["robot"])

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
        velocities = torch.stack(
            [env.scene[name].data.root_vel_w.torch[:, :3] for name in ("cube_1", "cube_2", "cube_3")], dim=1
        )
        slow = torch.amax(torch.abs(velocities), dim=(1, 2)) < maximum_cube_velocity
        robot: Articulation = env.scene["robot"]
        released = (robot.data.joint_pos.torch[:, self._finger_joint_ids] > minimum_finger_release_position).all(dim=1)
        stable = (stack_progress >= 2.0) & slow & released & (env.episode_length_buf >= minimum_episode_steps)
        self._stable_steps = torch.where(stable, self._stable_steps + 1, torch.zeros_like(self._stable_steps))
        self._update_success(self._stable_steps >= hold_steps)
        # The manager resets episode state before returning the step extras.
        self._reported_success.copy_(self.ever_success)
        env.extras["successes"] = self._reported_success
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
        self._target_potential = torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
        potential_param_names = (
            "robot_cfg",
            "tool_body_name",
            "tool_offset",
            "xy_threshold",
            "height_threshold",
            "cube_height",
        )
        self._potential_kwargs = {name: cfg.params[name] for name in potential_param_names if name in cfg.params}
        self._finger_joint_ids = _resolve_finger_joint_ids(env, env.scene[cfg.params["robot_cfg"].name])

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Set a non-free target from the newly sampled physical row."""
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        initial = role_conditioned_stack_potential(
            self._env,
            finger_joint_ids=self._finger_joint_ids,
            env_ids=env_ids,
            **self._potential_kwargs,
        )
        reset_state = get_stack_reset_runtime_state(self._env)
        row_target = reset_state.target_potentials[env_ids]
        self._target_potential[env_ids] = torch.maximum(
            row_target,
            initial + 0.25,
        ).clamp_max(10.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimum_episode_steps: int = 3,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        tool_body_name: str = "panda_hand",
        tool_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
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
            xy_threshold=xy_threshold,
            height_threshold=height_threshold,
            cube_height=cube_height,
            finger_joint_ids=self._finger_joint_ids,
        )
        reached = (current >= self._target_potential) & (env.episode_length_buf >= minimum_episode_steps)
        self._update_success(reached)
        return self._no_termination
