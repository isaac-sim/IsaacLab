# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior of tasks that run controllers inside the Newton step program."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import gymnasium as gym
import pytest
import torch
import warp as wp
from isaaclab_newton.envs.mdp.actions.newton_task_space_actions import NewtonOperationalSpaceControllerAction

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.newton_step_program.captured_cartpole import CapturedCartpole
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_OSC_TASK = "IsaacContrib-StepProgram-Reach-Franka-NewtonOSC"


def _rollout_osc(num_steps: int) -> tuple[torch.Tensor, bool]:
    """Step the Newton OSC reach task with fixed pose targets and return the arm trajectory and loop ownership."""
    sim_utils.create_new_stage()
    env_cfg = resolve_presets(load_cfg_from_registry(_OSC_TASK, "env_cfg_entry_point"), selected=("newton_mjwarp",))
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = 4
    env_cfg.seed = 7
    env = gym.make(_OSC_TASK, cfg=env_cfg)
    try:
        env.unwrapped.sim._app_control_on_stop_handle = None
        env.reset()
        robot = env.unwrapped.scene["robot"]
        generator = torch.Generator(device="cuda:0").manual_seed(0)
        # Absolute root-frame pose targets around a reachable pose with the hand pointing down.
        center = torch.tensor([0.45, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0], device="cuda:0")
        trajectory = []
        with torch.inference_mode():
            for _ in range(num_steps):
                offset = 0.1 * (torch.rand(env.unwrapped.num_envs, 3, device="cuda:0", generator=generator) - 0.5)
                actions = center.repeat(env.unwrapped.num_envs, 1)
                actions[:, :3] += offset
                env.step(actions)
                trajectory.append(robot.data.joint_pos.torch.clone())
        return torch.stack(trajectory), env.unwrapped._physics_handles_decimation
    finally:
        env.close()


def test_newton_osc_in_step_program_matches_host_controller(monkeypatch: pytest.MonkeyPatch):
    """The captured, physics-step controller tracks targets exactly like the same controller run on the host.

    On the host the term must run before every physics step, so the environment drives the decimation loop; inside
    the step program the loop stays folded into one physics call.
    """
    in_program, folded = _rollout_osc(num_steps=20)
    with monkeypatch.context() as patch:
        patch.setattr(NewtonOperationalSpaceControllerAction, "_supports_step_program", lambda self: False)
        on_host, host_folded = _rollout_osc(num_steps=20)

    assert folded and not host_folded
    # The arm moves toward the targets, so the comparison is not between two resting trajectories.
    assert (on_host[-1] - on_host[0]).abs().max() > 0.05
    torch.testing.assert_close(in_program, on_host, atol=1e-4, rtol=0.0)


def _rollout_cartpole(actions: list[torch.Tensor], captured: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Step the Warp cart-pole MDP eagerly or as one captured environment-step graph."""
    sim_utils.create_new_stage()
    env_cfg = resolve_presets(
        load_cfg_from_registry("Isaac-Cartpole", "env_cfg_entry_point"), selected=("newton_mjwarp",)
    )
    env_cfg.sim.device = "cuda:0"
    env_cfg.scene.num_envs = 64
    # Short episodes guarantee partial resets inside the rollout.
    env_cfg.episode_length_s = 0.25
    env_cfg.seed = 3
    env = gym.make("Isaac-Cartpole", cfg=env_cfg)
    try:
        env.unwrapped.sim._app_control_on_stop_handle = None
        env.reset()
        mdp = CapturedCartpole(env.unwrapped)
        if captured:
            mdp.capture()
        obs, reward, done = [], [], []
        for action in actions:
            wp.copy(mdp.actions, wp.from_torch(action))
            mdp.replay() if captured else mdp.step()
            obs.append(wp.to_torch(mdp.obs).clone())
            reward.append(wp.to_torch(mdp.reward).clone())
            done.append(wp.to_torch(mdp.truncated).clone() | wp.to_torch(mdp.terminated))
        return torch.stack(obs), torch.stack(reward), torch.stack(done)
    finally:
        env.close()


def test_whole_environment_step_captures_with_newton_physics():
    """An environment step with MDP stages, partial resets, and the Newton step program replays as one graph.

    The captured step records the physics program into the caller's graph and must reproduce eager stepping,
    including worlds reset inside the graph.
    """
    generator = torch.Generator(device="cuda:0").manual_seed(0)
    actions = [2 * torch.rand(64, 1, device="cuda:0", generator=generator) - 1 for _ in range(60)]
    eager = _rollout_cartpole(actions, captured=False)
    captured = _rollout_cartpole(actions, captured=True)

    # Episodes last 15 steps, so every world resets several times.
    assert int(eager[2].sum()) >= 3 * 64
    for eager_values, captured_values in zip(eager, captured):
        torch.testing.assert_close(captured_values, eager_values, atol=1e-5, rtol=0.0)
