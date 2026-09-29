# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the CUDA-graph handling of the host-side PhysX actuator runtime."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import warp as wp

from isaaclab.actuators.newton.physx_runtime import PhysxActuatorRuntime


def _runtime() -> PhysxActuatorRuntime:
    """Create a runtime for a CUDA articulation with a mocked logger."""
    return PhysxActuatorRuntime(SimpleNamespace(device="cuda:0"), logger=Mock())


def test_graph_capture_failure_restores_adapter_state_and_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed second capture discards the graphs and restores the state swapped by the first capture."""

    class _FailingCapture:
        capture_count = 0

        def __init__(self, *args, **kwargs) -> None:
            self.capture_index = type(self).capture_count
            type(self).capture_count += 1
            self.graph = object()

        def __enter__(self):
            if self.capture_index == 1:
                raise RuntimeError("second capture unavailable")
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
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


def test_compute_runs_eagerly_after_graph_capture_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A graphable adapter whose capture fails still computes the current command eagerly."""

    class _FailingCapture:
        capture_count = 0

        def __init__(self, *args, **kwargs) -> None:
            type(self).capture_count += 1

        def __enter__(self):
            raise RuntimeError("capture unavailable")

        def __exit__(self, exc_type, exc_value, traceback) -> bool:
            return False

    runtime = _runtime()
    runtime.adapter = SimpleNamespace(is_stateful=False, is_all_graphable=True, _states_a=object(), _states_b=object())
    eager_compute = Mock()
    monkeypatch.setattr(wp, "get_device", lambda device: SimpleNamespace(is_cuda=True, is_capturing=False))
    monkeypatch.setattr(wp, "ScopedCapture", _FailingCapture)
    monkeypatch.setattr(runtime, "_run_native_actuator_kernels", eager_compute)
    collection = SimpleNamespace()

    runtime.compute(collection, 0.01)
    runtime.compute(collection, 0.01)

    # The failed capture is not retried, and every step falls back to the eager kernels.
    assert _FailingCapture.capture_count == 1
    assert runtime.native_actuator_graphs == ()
    assert eager_compute.call_count == 2
    eager_compute.assert_called_with(collection, 0.01)


def test_stateful_actuator_rejects_outer_cuda_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stateful adapters cannot safely mutate their buffers inside an outer CUDA capture."""
    runtime = _runtime()
    runtime.adapter = SimpleNamespace(is_stateful=True)
    monkeypatch.setattr(wp, "get_device", lambda device: SimpleNamespace(is_cuda=True, is_capturing=True))

    with pytest.raises(RuntimeError, match="stateful Newton actuators cannot run inside an outer CUDA graph capture"):
        runtime.compute(SimpleNamespace(), 0.01)
