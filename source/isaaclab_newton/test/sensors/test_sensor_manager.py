# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Newton sensor manager."""

import pytest

from isaaclab_newton.sensors import NewtonSensorManager


class _FakeModel:
    """Minimal Newton model interface needed by the eager sensor manager path."""

    shape_count = 1
    bvh_shapes = object()

    def __init__(self):
        self.refit_states = []

    def bvh_refit_shapes(self, state) -> None:
        self.refit_states.append(state)


def test_update_runs_requested_tasks_and_refits_once_per_state_change():
    """Requested tasks run independently while all consumers share one state refit."""
    model = _FakeModel()
    state_a = object()
    state_b = object()
    updates = []
    manager = NewtonSensorManager(
        model=model,
        state=state_a,
        device="cpu",
        use_cuda_graph=False,
        capture_fn=lambda _: None,
    )
    manager.register("first", lambda: updates.append("first"))
    manager.register("second", lambda: updates.append("second"))

    manager.update("second")
    manager.update("first")
    assert updates == ["second", "first"]
    assert model.refit_states == [state_a]

    manager.set_state(state_b)
    manager.update("first", "second")
    assert updates == ["second", "first", "first", "second"]
    assert model.refit_states == [state_a, state_b]
    assert manager.state is state_b


def test_registration_errors_and_cleanup_are_explicit():
    """Duplicate and unknown task names fail, while cleanup is idempotent."""
    manager = NewtonSensorManager(
        model=_FakeModel(),
        state=object(),
        device="cpu",
        use_cuda_graph=False,
        capture_fn=lambda _: None,
    )
    manager.register("sensor", lambda: None)

    with pytest.raises(ValueError, match="already registered"):
        manager.register("sensor", lambda: None)
    with pytest.raises(KeyError, match="not registered"):
        manager.update("missing")

    manager.unregister("sensor")
    manager.unregister("sensor")
    assert manager.task_names == ()
