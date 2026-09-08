# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import sys

# Import pinocchio before AppLauncher so Isaac Lab's dependency wins over Isaac Sim's bundled copy.
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Local imports should be imported last
from env_test_utils import _check_valid_tensor, _run_environments, setup_environment  # isort: skip


_SKIPPED_TASKS = {
    "IsaacContrib-AutoMate-Assembly-Direct": "Requires CUDA support outside the standard environment test runner.",
    "IsaacContrib-AutoMate-Disassembly-Direct": "Requires CUDA support outside the standard environment test runner.",
}
_SKIPPED_TASK_SUBSTRINGS = {
    "RmpFlow": "Uses SingleArticulation, which requires an update.",
    "Skillgen": "Requires cuRobo-specific coverage.",
    "Suction": "Requires CPU simulation.",
}
_COVERED_TASKS = [
    "IsaacContrib-Lift-Cube-Franka",  # Already covered by test_environment_determinism.py
]


def _skip_reason(task_name: str) -> str | None:
    """Return the documented reason for skipping a contributed environment."""
    if task_name in _SKIPPED_TASKS:
        return _SKIPPED_TASKS[task_name]
    return next((reason for substring, reason in _SKIPPED_TASK_SUBSTRINGS.items() if substring in task_name), None)


def _contrib_environment_params() -> list:
    """Return each contributed environment with its documented test marks."""
    params = []
    for task_param in setup_environment(
        multi_agent=False,
        tier="contrib",
        exclude_task_names=_COVERED_TASKS,
    ):
        task_name = getattr(task_param, "values", (task_param,))[0]
        marks = getattr(task_param, "marks", ())
        skip_reason = _skip_reason(task_name)
        if skip_reason is not None:
            marks = (*marks, pytest.mark.skip(reason=skip_reason))
        params.append(pytest.param(task_name, id=task_name, marks=marks))
    return params


def test_dr_legs_walk_seed_zero_remains_finite():
    """Keep DR Legs state finite under the random actions that exposed P-ADMM divergence."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg("IsaacContrib-DrLegs-Walk", device="cuda", num_envs=2)
        env_cfg.seed = 0
        env = gym.make("IsaacContrib-DrLegs-Walk", cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

        action_space = gym.vector.utils.batch_space(env.unwrapped.single_action_space, 2)
        action_space.seed(0)
        obs, _ = env.reset(seed=0)
        assert _check_valid_tensor(obs)

        with torch.inference_mode():
            for _ in range(5):
                actions = torch.as_tensor(action_space.sample(), device=env.unwrapped.device, dtype=torch.float32)
                transition = env.step(actions)
                assert all(_check_valid_tensor(data) for data in transition[:-1])
                joint_velocity = env.unwrapped.scene["robot"].data.joint_vel
                joint_velocity = joint_velocity.torch if hasattr(joint_velocity, "torch") else joint_velocity
                max_joint_velocity = torch.max(torch.abs(joint_velocity)).item()
                assert max_joint_velocity < 100.0, f"Joint velocity diverged to {max_joint_velocity} rad/s"
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()


@pytest.mark.parametrize("task_name", _contrib_environment_params())
def test_contrib_environments(task_name):
    num_envs = 3 if task_name == "IsaacContrib-Multitask-Manipulation" else 2
    _run_environments(task_name, device="cuda", num_envs=num_envs)
