# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the runtime stepping helpers."""

import numpy as np
import torch

from isaaclab.test.benchmark.stepping import (
    EnvironmentStepTimer,
    SimulationStepTimer,
    run_runtime_loop,
    sample_random_actions,
)


class _Space:
    def __init__(self, n):
        self.shape = (n,)


class _PhysicsManager:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1


class _Sim:
    def __init__(self):
        self.physics_manager = _PhysicsManager()

    def step(self):
        self.physics_manager.step()


class _Env:
    class _U:
        num_envs = 4
        device = "cpu"
        single_action_space = _Space(3)

    def __init__(self):
        self.unwrapped = _Env._U()
        self.unwrapped.sim = _Sim()
        self.reset_called = False
        self.steps = 0

    def reset(self):
        self.reset_called = True

    def step(self, actions):
        self.unwrapped.sim.step()
        self.unwrapped.sim.step()
        self.steps += 1
        return (None, None, None, {})


def test_sample_single_agent_shape_and_range():
    a = sample_random_actions(_Env())
    assert isinstance(a, torch.Tensor)
    assert tuple(a.shape) == (4, 3)
    assert float(a.min()) >= -1.0 - 1e-6 and float(a.max()) <= 1.0 + 1e-6


def test_run_runtime_loop_steps_and_times():
    env = _Env()
    times = run_runtime_loop(env, num_frames=5)
    assert env.reset_called and env.steps == 5
    assert len(times) == 5 and all(t >= 0.0 for t in times)


def test_run_runtime_loop_can_skip_reset():
    env = _Env()
    run_runtime_loop(env, num_frames=2, reset=False)
    assert not env.reset_called and env.steps == 2


def test_simulation_step_timer_counts_calls_and_restores_method():
    env = _Env()
    simulation_context = env.unwrapped.sim
    original_step = simulation_context.step

    with SimulationStepTimer(env, synchronize=False) as timer:
        run_runtime_loop(env, num_frames=3, reset=False)

    assert timer.num_calls == 6
    assert timer.total_time_s >= 0.0
    assert "step" not in vars(simulation_context)
    assert simulation_context.step.__func__ is original_step.__func__


def test_simulation_step_timer_restores_method_after_exception():
    env = _Env()
    simulation_context = env.unwrapped.sim
    original_step = simulation_context.step

    try:
        with SimulationStepTimer(env, synchronize=False):
            raise RuntimeError("expected")
    except RuntimeError:
        pass

    assert "step" not in vars(simulation_context)
    assert simulation_context.step.__func__ is original_step.__func__


def test_environment_step_timer_measures_only_step_calls():
    env = _Env()

    with EnvironmentStepTimer(env, synchronize=False) as timer:
        run_runtime_loop(env, num_frames=3, reset=False)

    assert timer.num_calls == 3
    assert timer.simulation_step_calls == 6
    assert timer.total_time_s >= timer.simulation_time_s
    assert "step" not in vars(env)
    assert "step" not in vars(env.unwrapped.sim)


class _MASpace:
    def __init__(self, n):
        self._n = n

    def sample(self):
        return np.zeros(self._n, dtype=np.float32)


class _MAEnv:
    class _U:
        num_envs = 4
        device = "cpu"
        action_spaces = {"a0": _MASpace(3), "a1": _MASpace(2)}

    def __init__(self):
        self.unwrapped = _MAEnv._U()


def test_sample_multi_agent_returns_dict_per_agent():
    actions = sample_random_actions(_MAEnv())
    assert isinstance(actions, dict)
    assert set(actions) == {"a0", "a1"}
    assert tuple(actions["a0"].shape) == (4, 3)
    assert tuple(actions["a1"].shape) == (4, 2)
