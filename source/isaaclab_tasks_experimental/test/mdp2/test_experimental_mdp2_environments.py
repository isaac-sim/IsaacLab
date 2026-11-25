# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests that build and step every registered experimental MDP2 task on Newton MJWarp."""

import gymnasium as gym
import isaaclab_tasks_experimental  # noqa: F401
import pytest
import torch
from isaaclab_experimental import mdp2

from isaaclab.sim import SimulationContext
from isaaclab.utils import replace

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


@pytest.mark.parametrize("disabled", [None, "action_rate", "joint_vel"])
def test_reach_curriculum_sets_reward_weights_at_first_reset_after_threshold(disabled):
    """The Reach curriculum keeps its reward weights until the first reset after its step threshold, then sets them
    in every world, from a captured step. It skips a reward that the task disables and leaves every other weight
    unchanged."""
    task_name = "Isaac-Reach-Franka-MDP2"
    env_cfg, _ = resolve_task_config(task_name, None, overrides=["physics=newton_mjwarp"])
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = 4
    env_cfg.curriculum_after_steps = 5
    # Episodes of four steps: the first resets come before the threshold, the next ones after it.
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    time_limit = mdp2.TimeLimit(episode_length_s=4 * step_dt)
    rewards = {**env_cfg.task.rewards, **({disabled: None} if disabled else {})}
    terminations = {**env_cfg.task.terminations, "time_out": time_limit}
    env_cfg.task = replace(env_cfg.task, rewards=rewards, terminations=terminations)
    env = gym.make(task_name, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        indices = {op.use: op.index for op in unwrapped.mdp.ops[mdp2.Phase.REWARDS]}
        names = [name for name in ("action_rate", "joint_vel") if name != disabled]
        assert f"rewards.{disabled}" not in indices
        weights = unwrapped.state.reward_weights.torch
        initial, final = weights.clone(), weights.clone()
        for name in names:
            final[:, indices[f"rewards.{name}"]] = getattr(env_cfg, f"curriculum_{name}")

        env.reset(seed=0)
        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
        early_reset, applied_at = False, None
        for step in range(1, 13):
            _, _, terminated, truncated, _ = env.step(actions)
            resetting = bool((terminated | truncated).any())
            early_reset |= resetting and step <= env_cfg.curriculum_after_steps
            if applied_at is None and resetting and step > env_cfg.curriculum_after_steps:
                applied_at = step
            expected = final if applied_at is not None else initial
            torch.testing.assert_close(weights, expected, msg=f"weights after step {step}")
        assert early_reset and applied_at is not None
        # Steps after the first replay the captured graphs, so the curriculum ran inside one.
        assert applied_at > 1 and unwrapped.num_graphs > 0
    finally:
        env.close()


def test_locomotion_world_configs_must_share_gear_ratios():
    """World configs that give a gear-weighted reward different gear ratios fail the build, which then leaves no
    simulation behind."""
    task_name = "Isaac-Humanoid-MDP2"
    env_cfg, _ = resolve_task_config(task_name, None, overrides=["physics=newton_mjwarp"])
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = 4
    energy = env_cfg.task.rewards["energy"]
    even_gears = replace(energy, params={**energy.params, "gear_ratio": {".*": 1.0}})
    env_cfg.task = [env_cfg.task, replace(env_cfg.task, rewards={**env_cfg.task.rewards, "energy": even_gears})]
    with pytest.raises(ValueError, match="gear ratios"):
        gym.make(task_name, cfg=env_cfg)
    assert SimulationContext.instance() is None
