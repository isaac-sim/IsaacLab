# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the runtime stepping helpers."""

import numpy as np
import pytest
import torch

from isaaclab.test.benchmark import stepping
from isaaclab.test.benchmark.stepping import measure_runtime_loop, run_runtime_loop, sample_random_actions


class _Space:
    def __init__(self, n):
        self.shape = (n,)


class _Env:
    class _U:
        num_envs = 4
        device = "cpu"
        single_action_space = _Space(3)

    def __init__(self):
        self.unwrapped = _Env._U()
        self.reset_called = False
        self.steps = 0
        self.action_ids = []

    def reset(self):
        self.reset_called = True

    def step(self, actions):
        self.steps += 1
        self.action_ids.append(id(actions))
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


def test_run_runtime_loop_excludes_warmup_and_reuses_actions():
    env = _Env()
    times = run_runtime_loop(env, num_frames=3, warmup_frames=2, reuse_action_buffer=True)
    assert env.steps == 5
    assert len(times) == 3
    assert len(set(env.action_ids)) == 1


def test_measure_runtime_loop_reports_first_step_separately():
    env = _Env()
    timing = measure_runtime_loop(env, num_frames=3, warmup_frames=2)
    assert timing.first_step_s >= 0.0
    assert len(timing.step_times_s) == 3
    assert env.steps == 5


def test_measure_runtime_loop_excludes_startup_action_sampling_and_synchronizes(monkeypatch):
    """The startup sample should time one completed step without action-generation work."""
    events = []
    clock = iter((100, 200))
    env = _Env()

    monkeypatch.setattr(stepping, "sample_random_actions", lambda _env: events.append("sample") or torch.zeros((4, 3)))
    monkeypatch.setattr(stepping, "_synchronize_env_device", lambda _env: events.append("sync"))
    monkeypatch.setattr(stepping.time, "perf_counter_ns", lambda: events.append("clock") or next(clock))
    original_step = env.step
    env.step = lambda actions: events.append("step") or original_step(actions)

    timing = measure_runtime_loop(env, num_frames=0, warmup_frames=1, synchronize_steps=True)

    assert timing.first_step_s == pytest.approx(1e-7)
    assert events[:6] == ["sample", "sync", "clock", "step", "sync", "clock"]


def test_run_runtime_loop_rejects_negative_frame_counts():
    env = _Env()
    for kwargs in ({"num_frames": -1}, {"num_frames": 1, "warmup_frames": -1}):
        with pytest.raises(ValueError):
            run_runtime_loop(env, **kwargs)


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
