# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the runtime stepping helpers."""

import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from isaaclab.benchmark.stepping import (
    PHYSICS_PROFILE_SCOPE,
    RENDER_PROFILE_SCOPE,
    EnvironmentStepTimingRecorder,
    profile_physics_steps,
    profile_renderers,
    run_runtime_loop,
    run_runtime_warmup,
    sample_random_actions,
)


@pytest.mark.parametrize(
    ("active", "inherited", "fail"),
    [
        pytest.param(False, False, False, id="disabled"),
        pytest.param(True, False, False, id="enabled"),
        pytest.param(True, True, True, id="inherited-failure"),
    ],
)
def test_profile_physics_steps_times_complete_step_once(monkeypatch, capsys, active, inherited, fail):
    """Profile complete steps once and restore the class across failures and repeated runs."""
    import warp as wp

    from isaaclab.physics import PhysicsManager

    calls = []

    class BaseManager(PhysicsManager):
        @classmethod
        def step(cls):
            assert cls is manager
            calls.append("base")

    class CoupledManager(BaseManager):
        @classmethod
        def step(cls):
            calls.append("before")
            super().step()
            calls.append("after")
            if fail:
                raise ValueError("step failed")

    class InheritedManager(CoupledManager):
        pass

    manager = InheritedManager if inherited else CoupledManager
    synchronize = Mock(wraps=wp.synchronize)
    monkeypatch.setattr(wp, "synchronize", synchronize)

    timings = []
    original_step = manager.step
    with pytest.raises(ValueError, match="step failed") if fail else nullcontext():
        with profile_physics_steps(manager, active=active, timings=timings) as collected:
            assert collected is timings
            manager.step()

    assert calls == ["before", "base", "after"]
    assert synchronize.call_count == (2 if active else 0)
    assert len(timings) == int(active)
    assert all(scope == PHYSICS_PROFILE_SCOPE and elapsed >= 0.0 for scope, elapsed in timings)
    assert PHYSICS_PROFILE_SCOPE not in capsys.readouterr().out
    assert manager.step == original_step
    assert ("step" in vars(manager)) is not inherited

    fail = False
    manager.step()
    for enabled in (True, False):
        with profile_physics_steps(manager, active=enabled) as later_timings:
            manager.step()
        assert len(later_timings) == int(enabled)
        assert len(timings) == int(active)
        assert manager.step == original_step
    assert synchronize.call_count == 2 * (int(active) + 1)


def test_profile_renderers_wraps_each_renderer(monkeypatch, capsys):
    """Renderer methods and instance overrides are restored after failures and repeated runs."""
    import warp as wp

    second_render = Mock(side_effect=ValueError("render failed"))

    class Renderer:
        def render_batch(self, render_data):
            second_render(render_data)

    first, second = SimpleNamespace(render_batch=Mock()), Renderer()
    originals = [first.render_batch, second.render_batch]
    context = SimpleNamespace(_renderer_entries=[(None, first), (None, second)])
    synchronize = Mock()
    monkeypatch.setattr(wp, "synchronize", synchronize)

    timings = []
    with pytest.raises(ValueError, match="render failed"):
        with profile_renderers(context, timings=timings) as collected:
            assert collected is timings
            first.render_batch(["first"])
            second.render_batch(["second"])

    originals[0].assert_called_once_with(["first"])
    second_render.assert_called_once_with(["second"])
    assert synchronize.call_count == 4
    assert len(timings) == 2
    assert all(scope == RENDER_PROFILE_SCOPE and elapsed >= 0.0 for scope, elapsed in timings)
    assert RENDER_PROFILE_SCOPE not in capsys.readouterr().out
    assert [first.render_batch, second.render_batch] == originals
    assert "render_batch" not in vars(second)

    second_render.side_effect = None
    first.render_batch(["unprofiled"])
    for enabled in (True, False):
        with profile_renderers(context, active=enabled) as later_timings:
            first.render_batch(["first"])
            second.render_batch(["second"])
        assert len(later_timings) == 2 * int(enabled)
        assert len(timings) == 2
        assert [first.render_batch, second.render_batch] == originals
        assert "render_batch" not in vars(second)
    assert synchronize.call_count == 8


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
    times = run_runtime_loop(env, num_steps=5)
    assert env.reset_called and env.steps == 5
    assert len(times) == 5 and all(t >= 0.0 for t in times)


def test_run_runtime_loop_can_skip_reset():
    env = _Env()
    run_runtime_loop(env, num_steps=2, reset=False)
    assert not env.reset_called and env.steps == 2


@pytest.mark.parametrize("num_steps", [0, 1])
def test_run_runtime_warmup_runs_exact_requested_steps(num_steps: int):
    env = _Env()

    times = run_runtime_warmup(env, num_steps=num_steps)

    assert env.reset_called
    assert env.steps == num_steps
    assert len(times) == num_steps


def test_environment_step_timer_measures_env_step_without_simulation_timing():
    env = _Env()

    with EnvironmentStepTimingRecorder(env, warmup_steps=2) as timer:
        run_runtime_loop(env, num_steps=5, reset=False)

    # All five steps run, but the first two are excluded from the recorded timings.
    assert env.steps == 5
    assert len(timer.step_times_s) == 3
    assert timer.simulation_step_times_s is None
    assert timer.simulation_step_calls is None
    assert "step" not in vars(env)
    assert "step" not in vars(env.unwrapped.sim)


def test_environment_step_timer_measures_only_step_calls():
    env = _Env()

    with EnvironmentStepTimingRecorder(env, measure_synchronized_step_breakdown=True) as timer:
        env.unwrapped.sim.step()
        run_runtime_loop(env, num_steps=3, reset=False)
        env.unwrapped.sim.step()

    assert len(timer.step_times_s) == 3
    assert timer.simulation_step_calls == 6
    assert len(timer.simulation_step_times_s) == 3
    assert all(total >= simulation for total, simulation in zip(timer.step_times_s, timer.simulation_step_times_s))
    assert "step" not in vars(env)
    assert "step" not in vars(env.unwrapped.sim)


def test_environment_step_timer_warmup_keeps_simulation_accounting_consistent():
    env = _Env()
    recorder = EnvironmentStepTimingRecorder(env, measure_synchronized_step_breakdown=True, warmup_steps=1)

    with recorder:
        run_runtime_loop(env, num_steps=3, reset=False)

    # The first env step is warmup; each _Env.step calls sim.step twice, so only the two
    # recorded steps' four simulation-step calls are counted (not the warmup step's two).
    assert len(recorder.step_times_s) == 2
    assert len(recorder.simulation_step_times_s) == 2
    assert recorder.simulation_step_calls == 4

    # Reusing the same recorder resets the warmup counter and the recorded series.
    with recorder:
        run_runtime_loop(env, num_steps=4, reset=False)

    assert len(recorder.step_times_s) == 3
    assert recorder.simulation_step_calls == 6


def test_environment_step_timer_excludes_pending_work_before_step(monkeypatch):
    import warp as wp

    env = _Env()
    clock_s = 0.0
    pending_work_s = 0.05

    def drain_pending_work():
        nonlocal clock_s, pending_work_s
        clock_s += pending_work_s
        pending_work_s = 0.0

    monkeypatch.setattr(time, "perf_counter", lambda: clock_s)
    monkeypatch.setattr(wp, "synchronize", drain_pending_work)

    with EnvironmentStepTimingRecorder(env, measure_synchronized_step_breakdown=True) as timer:
        env.step(None)

    assert timer.step_times_s == [0.0]


def test_environment_step_timer_synchronizes_torch_action_device(monkeypatch):
    import warp as wp

    events = []
    env = _Env()

    class _CudaAction:
        device = "cuda:1"

    env.step = lambda actions: events.append("environment")
    monkeypatch.setattr(wp, "synchronize", lambda: events.append("warp"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: events.append(f"torch:{device}"))

    with EnvironmentStepTimingRecorder(env, measure_synchronized_step_breakdown=True):
        env.step(_CudaAction())

    assert events == ["torch:cuda:1", "warp", "environment", "torch:cuda:1", "warp"]


def test_environment_step_timer_attributes_queued_work_to_its_boundary(monkeypatch):
    import warp as wp

    env = _Env()
    clock_s = 0.0
    pending_work_s = 0.0

    def queue_work(duration_s: float):
        nonlocal pending_work_s
        pending_work_s += duration_s

    def drain_pending_work():
        nonlocal clock_s, pending_work_s
        clock_s += pending_work_s
        pending_work_s = 0.0

    def simulation_step():
        queue_work(0.02)

    def environment_step(actions):
        queue_work(0.01)
        env.unwrapped.sim.step()
        queue_work(0.03)

    env.step = environment_step
    env.unwrapped.sim.step = simulation_step
    monkeypatch.setattr(time, "perf_counter", lambda: clock_s)
    monkeypatch.setattr(wp, "synchronize", drain_pending_work)

    with EnvironmentStepTimingRecorder(env, measure_synchronized_step_breakdown=True) as timer:
        env.step(None)

    assert timer.step_times_s == pytest.approx([0.06])
    assert timer.simulation_step_times_s == pytest.approx([0.02])


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
