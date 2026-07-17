# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .rewards import DirectReorientReward, evaluate_reorient_success

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import ReorientCommand


def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int, command_name: str) -> torch.Tensor:
    """Check if the task has been completed consecutively for a certain number of times.

    .. deprecated:: 9.0.0
        Only consumed by the deprecated
        :class:`~isaaclab_tasks.core.reorient.reorient_manager_env_cfg.ReorientObjectEnvCfg`.
        Use :class:`DirectReorientTimeout` instead.

    Args:
        env: The environment object.
        num_success: Threshold for the number of consecutive successes required.
        command_name: The command term to be used for extracting the goal.
    """
    if not globals().get("_warned_max_consecutive_success"):
        globals()["_warned_max_consecutive_success"] = True
        warnings.warn(
            "max_consecutive_success() is deprecated; use DirectReorientTimeout instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    command_term: ReorientCommand = env.command_manager.get_term(command_name)

    return command_term.metrics["consecutive_success"] >= num_success


def object_away_from_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has gone far from the goal.

    The object is considered to be out-of-reach if the distance between the goal and the object is greater
    than the threshold.

    .. deprecated:: 9.0.0
        Only consumed by the deprecated
        :class:`~isaaclab_tasks.core.reorient.reorient_manager_env_cfg.ReorientObjectEnvCfg`.
        Use :class:`object_reorientation_out_of_reach` instead.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot and the object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    if not globals().get("_warned_object_away_from_goal"):
        globals()["_warned_object_away_from_goal"] = True
        warnings.warn(
            "object_away_from_goal() is deprecated; use object_reorientation_out_of_reach instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    # extract useful elements
    command_term: ReorientCommand = env.command_manager.get_term(command_name)
    asset = env.scene[object_cfg.name]

    # object pos
    asset_pos_e = asset.data.root_pos_w.torch - env.scene.env_origins
    goal_pos_e = command_term.command[:, :3]

    return torch.linalg.norm(asset_pos_e - goal_pos_e, ord=2, dim=1) > threshold


class object_reorientation_out_of_reach(ManagerTermBase):
    """Terminate when object-to-goal distance is at least the threshold [m].

    The scalar task parameters arrive as term params, set at the configuration
    declaration site to match the Direct environment's values.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # resolved on first call: the command term does not exist yet during manager construction
        self._command_term: ReorientCommand | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        threshold: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """Return per-environment termination flags."""
        asset = env.scene[object_cfg.name]
        if self._command_term is None:
            self._command_term = env.command_manager.get_term(command_name)
        distance = torch.linalg.norm(asset.data.root_pos_w.torch - self._command_term.pos_command_w, ord=2, dim=-1)
        return distance >= threshold


class DirectReorientTimeout(ManagerTermBase):
    """Apply the Direct OpenAI progress-reset and timeout semantics.

    Resets the episode timer whenever the goal is reached so episodes extend
    across goal streaks (real Direct dynamics, not boundary cosmetics), and
    terminates after the consecutive-success cap or the usual timeout. The
    scalar task parameters arrive as term params, set at the configuration
    declaration site to match the Direct environment's values.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # resolved on first call: the command term does not exist yet during manager construction
        self._command_term: ReorientCommand | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        reward_name: str,
        success_tolerance: float,
        max_successes: int,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """Return per-environment timeout flags.

        Args:
            env: Environment containing the object, goal, and reward term.
            command_name: Goal command term name.
            reward_name: Reorientation reward term name.
            success_tolerance: Goal orientation tolerance [rad].
            max_successes: Consecutive-success cap before forcing a reset.
            object_cfg: Object scene entity.
        """
        object_asset = env.scene[object_cfg.name]
        if self._command_term is None:
            self._command_term = env.command_manager.get_term(command_name)
        goal_reached, _ = evaluate_reorient_success(
            object_asset.data.root_quat_w.torch, self._command_term.quat_command_w, success_tolerance
        )
        # in place: rebinding env.episode_length_buf would orphan references held elsewhere
        env.episode_length_buf.masked_fill_(goal_reached, 0)
        reward_term: DirectReorientReward = env.reward_manager.get_term_cfg(reward_name).func
        max_success_reached = reward_term.successes >= max_successes
        return (env.episode_length_buf >= env.max_episode_length - 1) | max_success_reached


class object_away_from_robot(ManagerTermBase):
    """Terminate when the object is farther than the threshold [m] from the robot."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """Return per-environment termination flags."""
        robot = env.scene[asset_cfg.name]
        obj = env.scene[object_cfg.name]
        distance = torch.linalg.norm(robot.data.root_pos_w.torch - obj.data.root_pos_w.torch, ord=2, dim=-1)
        return distance > threshold
