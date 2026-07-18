# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .rewards import ReorientReward, evaluate_reorient_success

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import ReorientCommand


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


class ReorientTimeout(ManagerTermBase):
    """Apply progress-reset and timeout semantics with a consecutive-success cap.

    Matches the OpenAI-variant timeout semantics of the Direct-workflow implementation
    (:class:`~isaaclab_tasks.core.reorient.reorient_direct_env.ReorientDirectEnv`):
    resets the episode timer whenever the goal is reached so episodes extend
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
        reward_term: ReorientReward = env.reward_manager.get_term_cfg(reward_name).func
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
