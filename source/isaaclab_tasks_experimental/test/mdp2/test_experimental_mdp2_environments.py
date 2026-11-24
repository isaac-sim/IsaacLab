# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests that build and step every registered experimental MDP2 task on Newton MJWarp."""

import gymnasium as gym
import isaaclab_tasks_experimental  # noqa: F401
import pytest
import torch

from isaaclab_tasks.utils import resolve_task_config

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")]

MDP2_TASKS = sorted(
    spec.id
    for spec in gym.registry.values()
    if str(spec.kwargs.get("env_cfg_entry_point", "")).startswith("isaaclab_tasks_experimental.mdp2.")
)

NUM_GRAPHS = {
    "IsaacContrib-Velocity-Flat-UnitreeA1-MDP2": 1,
    "Isaac-Velocity-Flat-AnymalD-MDP2": 2,
    "IsaacContrib-Velocity-Flat-UnitreeGo1-MDP2": 2,
}
"""Graphs that a captured step replays: one for PD actuators, and two for actuator networks from Torch checkpoints,
which Newton cannot capture, so the physics and the scene reset run between the graphs."""


@pytest.mark.parametrize("task_name", MDP2_TASKS)
def test_mdp2_environment_resets_and_steps(task_name):
    env_cfg, _ = resolve_task_config(task_name, None, overrides=["physics=newton_mjwarp"])
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = 4
    env = gym.make(task_name, cfg=env_cfg)
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(3):
            actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            obs, reward, terminated, truncated, _ = env.step(actions)
        assert all(torch.isfinite(value).all() for value in [*obs.values(), reward])
        # The second step captured the step's graphs.
        if task_name in NUM_GRAPHS:
            assert env.unwrapped.num_graphs == NUM_GRAPHS[task_name]
    finally:
        env.close()
