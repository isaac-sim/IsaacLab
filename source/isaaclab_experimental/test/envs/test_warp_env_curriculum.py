# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curricula of Warp environments."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

from collections import Counter

import torch
from isaaclab_experimental.envs.frontend import WarpFrontend
from isaaclab_experimental.utils import WarpGraphCache

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_TASK = "Isaac-Reach-Franka"
_NUM_ENVS = 16


def test_unchanged_term_settings_keep_the_recorded_stages(monkeypatch):
    """Curricula set their terms on every reset; a stage records again only when a setting it read changes.

    Past its threshold, Reach's ``modify_reward_weight`` raises two penalties on every reset (episodes of 15
    steps), and the recorded reward stage reads weights on the device. Setting every term of every manager to
    its own configuration records nothing either.
    """
    recordings = Counter()
    call_steps = WarpGraphCache.call_steps

    def counting_call_steps(self, stage, *args, **kwargs):
        recorded = set(self.captured_stages)
        result = call_steps(self, stage, *args, **kwargs)
        recordings[stage] += len(set(self.captured_stages) - recorded)
        return result

    monkeypatch.setattr(WarpGraphCache, "call_steps", counting_call_steps)
    env_cfg, _ = resolve_task_config(_TASK, "", overrides=("physics=newton_mjwarp",))
    env_cfg.scene.num_envs = _NUM_ENVS
    env_cfg.seed = 3
    env_cfg.episode_length_s = 0.5
    sim_utils.create_new_stage()
    env = WarpFrontend.build_env(env_cfg, _TASK).unwrapped
    try:
        env.reset()
        env.common_step_counter = 5000
        actions = torch.zeros((_NUM_ENVS, env.action_space.shape[-1]), device=env.device)

        def step(num_steps: int) -> int:
            reset_steps = 0
            for _ in range(num_steps):
                _, _, terminated, truncated, _ = env.step(actions)
                reset_steps += int(bool((terminated | truncated).any()))
            return reset_steps

        assert step(60) >= 3
        assert env.reward_manager.get_term_cfg("action_rate").weight == -0.005
        assert recordings["RewardManager_compute"] == 1

        recorded = dict(recordings)
        term_names = {
            env.reward_manager: env.reward_manager.active_terms,
            env.termination_manager: env.termination_manager.active_terms,
            env.command_manager: env.command_manager.active_terms,
            env.event_manager: [name for names in env.event_manager.active_terms.values() for name in names],
            env.observation_manager: [
                f"{group}/{name}" for group, names in env.observation_manager.active_terms.items() for name in names
            ],
        }
        for manager, names in term_names.items():
            for name in names:
                manager.set_term_cfg(name, manager.get_term_cfg(name))
        assert step(20) >= 1
        assert dict(recordings) == recorded
    finally:
        env.close()
