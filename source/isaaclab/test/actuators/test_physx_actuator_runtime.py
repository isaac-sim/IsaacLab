# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused unit tests for the shared host-side PhysX actuator runtime."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import warp as wp

from isaaclab.actuators.newton.physx_runtime import PhysxActuatorRuntime


def _runtime() -> PhysxActuatorRuntime:
    return PhysxActuatorRuntime(SimpleNamespace(device="cuda:0"), logger=Mock())


def test_graph_capture_builds_two_graphs_and_restores_adapter_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture both alternating state graphs without leaking the capture-time state swap."""
    graphs = [object(), object()]

    class _Capture:
        def __init__(self, *args, **kwargs):
            self.graph = graphs.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    state_a, state_b = object(), object()
    runtime = _runtime()
    runtime.adapter = SimpleNamespace(_states_a=state_a, _states_b=state_b)
    monkeypatch.setattr(wp, "ScopedCapture", _Capture)
    monkeypatch.setattr(runtime, "_run_native_actuator_kernels", Mock())

    runtime._capture_native_actuator_graphs(SimpleNamespace(), 0.01)

    assert len(runtime.native_actuator_graphs) == 2
    assert runtime.adapter._states_a is state_a
    assert runtime.adapter._states_b is state_b
    assert runtime._native_actuator_graph_index == 0


def test_graph_capture_failure_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial capture failure must restore adapter state before eager execution."""

    class _FailingCapture:
        capture_count = 0

        def __init__(self, *args, **kwargs):
            self.capture_index = self.capture_count
            type(self).capture_count += 1
            self.graph = object()

        def __enter__(self):
            if self.capture_index == 1:
                raise RuntimeError("second capture unavailable")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    state_a, state_b = object(), object()
    runtime = _runtime()
    runtime.adapter = SimpleNamespace(_states_a=state_a, _states_b=state_b)

    def _swap_adapter_state(*args, **kwargs) -> None:
        runtime.adapter._states_a, runtime.adapter._states_b = runtime.adapter._states_b, runtime.adapter._states_a

    monkeypatch.setattr(wp, "ScopedCapture", _FailingCapture)
    monkeypatch.setattr(runtime, "_run_native_actuator_kernels", _swap_adapter_state)

    runtime._capture_native_actuator_graphs(SimpleNamespace(), 0.01)

    assert runtime.native_actuator_graphs == ()
    assert runtime.adapter._states_a is state_a
    assert runtime.adapter._states_b is state_b
    runtime._logger.warning.assert_called_once()


def test_compute_falls_back_to_eager_after_graph_capture_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed host capture during compute must execute the current command eagerly."""

    class _FailingCapture:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("capture unavailable")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    runtime = _runtime()
    runtime.adapter = SimpleNamespace(
        is_stateful=False,
        is_all_graphable=True,
        _states_a=object(),
        _states_b=object(),
    )
    eager_compute = Mock()
    monkeypatch.setattr(wp, "get_device", lambda device: SimpleNamespace(is_cuda=True, is_capturing=False))
    monkeypatch.setattr(wp, "ScopedCapture", _FailingCapture)
    monkeypatch.setattr(runtime, "_run_native_actuator_kernels", eager_compute)
    collection = SimpleNamespace()

    runtime.compute(collection, 0.01)

    eager_compute.assert_called_once_with(collection, 0.01)


def test_compute_launches_alternating_graphs_and_swaps_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successive graphable computes must ping-pong graphs and adapter state exactly once."""
    graph_a, graph_b = object(), object()
    swap_state = Mock()
    runtime = _runtime()
    runtime.adapter = SimpleNamespace(is_stateful=True, is_all_graphable=True, _swap_state_buffers=swap_state)
    runtime.native_actuator_graphs = (graph_a, graph_b)
    eager_compute = Mock()
    launch_graph = Mock()
    monkeypatch.setattr(wp, "get_device", lambda device: SimpleNamespace(is_cuda=True, is_capturing=False))
    monkeypatch.setattr(wp, "capture_launch", launch_graph)
    monkeypatch.setattr(runtime, "_run_native_actuator_kernels", eager_compute)

    runtime.compute(SimpleNamespace(), 0.01)
    runtime.compute(SimpleNamespace(), 0.01)

    assert launch_graph.call_args_list == [call(graph_a), call(graph_b)]
    assert swap_state.call_count == 2
    assert runtime._native_actuator_graph_index == 0
    eager_compute.assert_not_called()


def test_stateful_actuator_rejects_outer_cuda_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stateful adapters cannot safely mutate their buffers inside an outer CUDA capture."""
    runtime = _runtime()
    runtime.adapter = SimpleNamespace(is_stateful=True)
    monkeypatch.setattr(wp, "get_device", lambda device: SimpleNamespace(is_cuda=True, is_capturing=True))

    with pytest.raises(RuntimeError, match="stateful Newton actuators cannot run inside an outer CUDA graph capture"):
        runtime.compute(SimpleNamespace(), 0.01)
