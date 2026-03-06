# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common curriculum classes for the drone navigation environment.

The curriculum classes can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ObstacleDensityCurriculum(ManagerTermBase):
    """Curriculum that adjusts obstacle density based on performance.

    The difficulty state is stored internally in the class instance, avoiding
    the need to store state on the environment object.

    The curriculum tracks per-environment difficulty levels used to control
    the number of obstacles spawned in each environment. Difficulty progresses
    based on agent performance (successful goal reaching vs. collisions).

    Attributes:
        cfg: The configuration of the curriculum term.
        _min_difficulty: Minimum difficulty level for obstacle density.
        _max_difficulty: Maximum difficulty level for obstacle density.
        _difficulty_levels: Tensor of shape (num_envs,) tracking difficulty per environment.
        _asset_cfg: Scene entity configuration for the robot.
        _command_name: Name of the command to track.
    """

    cfg: CurriculumTermCfg
    """The configuration of the curriculum term."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        """Initialize the curriculum term.

        Args:
            cfg: Configuration for the curriculum term.
            env: The manager-based RL environment instance.
        """
        super().__init__(cfg, env)

        # Extract parameters from config
        self._min_difficulty = cfg.params["min_difficulty"]
        self._max_difficulty = cfg.params["max_difficulty"]
        self._asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._command_name = cfg.params.get("command_name", "target_pose")

        # Initialize difficulty levels for all environments
        self._difficulty_levels = torch.ones(env.num_envs, device=env.device) * self._min_difficulty

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "target_pose",
        min_difficulty: int | None = None,
        max_difficulty: int | None = None,
    ) -> float:
        """Update obstacle density curriculum based on performance.

        Args:
            env: The manager-based RL environment instance.
            env_ids: Environment indices to update.
            asset_cfg: Scene entity configuration for the robot. Defaults to SceneEntityCfg("robot").
            command_name: Name of the command to track. Defaults to "target_pose".
            max_difficulty: Maximum difficulty level. Defaults to 10.
            min_difficulty: Minimum difficulty level. Defaults to 2.

        Returns:
            Mean difficulty level across all environments (for logging).
        """
        # Extract robot and command
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        target_position_w = command[:, :3].clone()
        current_position = asset.data.root_pos_w - env.scene.env_origins
        position_error = torch.norm(target_position_w[env_ids] - current_position[env_ids], dim=1)

        # Decide difficulty changes
        crashed = env.termination_manager.terminated[env_ids]
        move_up = position_error < 1.5  # Success
        move_down = crashed & ~move_up

        # Update difficulty levels
        self._difficulty_levels[env_ids] += move_up.long() - move_down.long()
        self._difficulty_levels[env_ids] = torch.clamp(
            self._difficulty_levels[env_ids], min=self._min_difficulty, max=self._max_difficulty - 1
        )

        return self._difficulty_levels.float().mean().item()

    @property
    def difficulty_levels(self) -> torch.Tensor:
        """Get the current difficulty levels for all environments.

        Returns:
            Tensor of shape (num_envs,) with difficulty levels.
        """
        return self._difficulty_levels

    @property
    def min_difficulty(self) -> int:
        """Get the minimum difficulty level."""
        return self._min_difficulty

    @property
    def max_difficulty(self) -> int:
        """Get the maximum difficulty level."""
        return self._max_difficulty


def get_obstacle_curriculum_term(env: ManagerBasedRLEnv) -> ObstacleDensityCurriculum | None:
    """Get the ObstacleDensityCurriculum instance from the curriculum manager.

    This helper function searches the curriculum manager for an active
    ObstacleDensityCurriculum term and returns it if found. This allows
    other MDP components (rewards, events) to access the curriculum state.

    Args:
        env: The manager-based RL environment instance.

    Returns:
        The ObstacleDensityCurriculum instance if found, None otherwise.
    """
    curriculum_manager = env.curriculum_manager
    for term_cfg in curriculum_manager._term_cfgs:
        if isinstance(term_cfg.func, ObstacleDensityCurriculum):
            return term_cfg.func
    return None
