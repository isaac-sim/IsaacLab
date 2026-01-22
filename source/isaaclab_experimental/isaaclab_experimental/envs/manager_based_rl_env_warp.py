# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental manager-based RL env entry point.

This is intentionally a minimal shim to bootstrap an experimental entry point
without copying task code. The initial behavior matches the existing
`isaaclab.envs.ManagerBasedRLEnv` exactly.

Future work will incrementally replace internals with Warp-first, graph-friendly
pipelines while keeping the manager-based task authoring model.
"""

from __future__ import annotations

from isaaclab_experimental.managers import RewardManager

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandManager, CurriculumManager, TerminationManager


class ManagerBasedRLEnvWarp(ManagerBasedRLEnv):
    """Experimental drop-in replacement for `ManagerBasedRLEnv`.

    Notes:
    - No behavior changes are introduced yet. This class exists to provide a new
      Gym entry point (`isaaclab_experimental.envs:ManagerBasedRLEnvWarp`) that
      we can evolve independently.
    """

    cfg: ManagerBasedRLEnvCfg

    def load_managers(self):
        """Load managers but use experimental `RewardManager`.

        This keeps behavior identical to `isaaclab.envs.ManagerBasedRLEnv` while allowing the reward
        pipeline to diverge in `isaaclab_experimental.managers.reward_manager`.
        """
        # -- command manager (order matters: observations may depend on commands/actions)
        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations/actions/events/recorders.
        ManagerBasedEnv.load_managers(self)

        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager (experimental fork)
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
