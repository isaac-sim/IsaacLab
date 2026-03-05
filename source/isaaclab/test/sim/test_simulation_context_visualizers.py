# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SimulationContext visualizer orchestration."""

from __future__ import annotations

from isaaclab.sim.simulation_context import SimulationContext


class _FakePhysicsManager:
    def __init__(self):
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class _FakeProvider:
    def __init__(self):
        self.update_calls = []

    def update(self, env_ids=None):
        self.update_calls.append(env_ids)


class _FakeVisualizer:
    def __init__(
        self,
        *,
        env_ids=None,
        running=True,
        closed=False,
        rendering_paused=False,
        training_paused_steps=0,
        raises_on_step=False,
        requires_forward=False,
    ):
        self._env_ids = env_ids
        self._running = running
        self._closed = closed
        self._rendering_paused = rendering_paused
        self._training_paused_steps = training_paused_steps
        self._raises_on_step = raises_on_step
        self._requires_forward = requires_forward
        self.step_calls = []
        self.close_calls = 0

    @property
    def is_closed(self):
        return self._closed

    def is_running(self):
        return self._running

    def is_rendering_paused(self):
        return self._rendering_paused

    def is_training_paused(self):
        if self._training_paused_steps > 0:
            self._training_paused_steps -= 1
            return True
        return False

    def step(self, dt):
        self.step_calls.append(dt)
        if self._raises_on_step:
            raise RuntimeError("step failed")

    def close(self):
        self.close_calls += 1
        self._closed = True

    def get_visualized_env_ids(self):
        return self._env_ids

    def requires_forward_before_step(self):
        return self._requires_forward


def _make_context(visualizers, provider=None):
    ctx = object.__new__(SimulationContext)
    ctx._visualizers = list(visualizers)
    ctx._scene_data_provider = provider
    ctx.physics_manager = _FakePhysicsManager()
    ctx._visualizer_step_counter = 0
    return ctx


def test_update_scene_data_provider_unions_env_ids_and_forwards():
    provider = _FakeProvider()
    viz_a = _FakeVisualizer(env_ids=[0, 2], requires_forward=True)
    viz_b = _FakeVisualizer(env_ids=[2, 3])
    viz_c = _FakeVisualizer(env_ids=None)
    ctx = _make_context([viz_a, viz_b, viz_c], provider=provider)

    ctx.update_scene_data_provider()

    assert ctx.physics_manager.forward_calls == 1
    assert provider.update_calls == [[0, 2, 3]]
    assert ctx._visualizer_step_counter == 1


def test_update_scene_data_provider_force_forward_with_no_visualizers():
    provider = _FakeProvider()
    ctx = _make_context([], provider=provider)
    ctx.update_scene_data_provider(force_require_forward=True)
    assert ctx.physics_manager.forward_calls == 1
    assert provider.update_calls == [None]


def test_update_visualizers_removes_closed_nonrunning_and_failed(caplog):
    provider = _FakeProvider()
    closed_viz = _FakeVisualizer(closed=True)
    stopped_viz = _FakeVisualizer(running=False)
    failing_viz = _FakeVisualizer(raises_on_step=True)
    paused_viz = _FakeVisualizer(rendering_paused=True)
    healthy_viz = _FakeVisualizer(env_ids=[1])
    ctx = _make_context([closed_viz, stopped_viz, failing_viz, paused_viz, healthy_viz], provider=provider)

    with caplog.at_level("ERROR"):
        ctx.update_visualizers(0.1)

    assert ctx._visualizers == [paused_viz, healthy_viz]
    assert closed_viz.close_calls == 1
    assert stopped_viz.close_calls == 1
    assert failing_viz.close_calls == 1
    assert paused_viz.close_calls == 0
    assert healthy_viz.step_calls == [0.1]
    assert any("Error stepping visualizer" in r.message for r in caplog.records)


def test_update_visualizers_handles_training_pause_loop():
    provider = _FakeProvider()
    viz = _FakeVisualizer(training_paused_steps=1)
    ctx = _make_context([viz], provider=provider)

    ctx.update_visualizers(0.2)

    assert viz.step_calls == [0.0, 0.2]
