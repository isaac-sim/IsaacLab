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
from isaaclab_newton.envs.mdp.actions.newton_task_space_actions import NewtonOperationalSpaceControllerAction

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
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
