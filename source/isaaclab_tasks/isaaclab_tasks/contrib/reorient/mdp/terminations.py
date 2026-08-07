# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms specific to the OpenAI Shadow Hand variant."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.core.reorient.mdp.commands import ReorientCommand


class reorient_timeout(ManagerTermBase):
    """Time out an episode that has run its full length without reaching a goal.

    The timer restarts on every goal reach, so episodes extend across success streaks.
    This matches the OpenAI Direct variant, which is the only configuration that enables
    the behavior. Pair it with :func:`max_consecutive_success` to also stop on the streak
    cap, and declare both with ``time_out=True``.

    Args:
        cfg: Configuration object specifying term parameters.
        env: The manager-based RL environment.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._steps_since_success = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # resolved on first call: the command term does not exist yet during manager construction
        self._command_term: ReorientCommand | None = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._steps_since_success[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        success_tolerance: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """Return per-environment timeout flags.

        Args:
            env: The environment object.
            command_name: The command term to be used for extracting the goal.
            success_tolerance: Maximum successful orientation error [rad].
            object_cfg: The configuration for the scene entity. Default is "object".
        """
        asset = env.scene[object_cfg.name]
        if self._command_term is None:
            self._command_term = env.command_manager.get_term(command_name)
        # terminations run before the command manager, so its metrics are one step stale
        dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w.torch, self._command_term.quat_command_w)
        goal_reached = dtheta <= success_tolerance
        self._steps_since_success += 1
        # masked_fill_ rather than boolean indexing: the latter forces a host synchronization
        self._steps_since_success.masked_fill_(goal_reached, 0)

        return self._steps_since_success >= env.max_episode_length - 1
