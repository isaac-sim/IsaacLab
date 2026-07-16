# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for recorded replay on capture-unsafe Newton hot paths."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.actuators import NewtonActuatorAdapter
from isaaclab_newton.assets import ArticulationData, RigidObjectData
from isaaclab_newton.physics import NewtonManager

from isaaclab.physics import PhysicsManager
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.warp import ProxyArray, WarpLaunchCache


class _NoOpActuator:
    """Minimal Newton actuator for adapter launch tests."""

    def __init__(self, indices: wp.array):
        self.indices = indices
        self.control_computed_output_attr = None

    def state(self):
        return None

    def step(self, sim_state, sim_control, state_in, state_out, dt: float) -> None:
        del sim_state, sim_control, state_in, state_out, dt

    def is_graphable(self) -> bool:
        return False


class _RebindReset(RuntimeError):
    """Sentinel raised when the rebind hook resets its launch cache."""


class _ResetProbe:
    """Cache probe that stops a simulation rebind after observing reset."""

    def __init__(self, events: list[tuple[str, str | None]]):
        self.events = events

    def reset(self) -> None:
        self.events.append(("reset", None))
        raise _RebindReset


@pytest.fixture(scope="module")
def cuda_device() -> str:
    """Return a CUDA device or skip the module when CUDA is unavailable."""
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA is required for Warp launch replay tests.")
    return "cuda:0"


def _spatial_vectors(num_envs: int, value: float, device: str) -> wp.array:
    """Create constant spatial vectors for a derived-state test."""
    values = np.full((num_envs, 6), value, dtype=np.float32)
    return wp.array(values, dtype=wp.spatial_vectorf, device=device)


def _make_root_velocity_data(data_cls: type, source: wp.array, warp_launch: WarpLaunchCache, device: str):
    """Build the fields used by a data class' lazy root-link velocity property."""
    num_envs = source.shape[0]
    root_pose = np.zeros((num_envs, 7), dtype=np.float32)
    root_pose[:, 6] = 1.0

    data = object.__new__(data_cls)
    data.device = device
    data._warp_launch = warp_launch
    data._num_instances = num_envs
    data._sim_timestamp = 1.0
    data._root_com_vel_w_ta = ProxyArray(source)
    data._root_link_pose_w_ta = ProxyArray(wp.array(root_pose, dtype=wp.transformf, device=device))
    data._body_com_pos_b_ta = ProxyArray(wp.zeros((num_envs, 1), dtype=wp.vec3f, device=device))
    data._sim_bind_body_com_pos_b = data._body_com_pos_b_ta.warp
    data._root_link_vel_w = TimestampedBuffer(
        shape=(num_envs,),
        dtype=wp.spatial_vectorf,
        device=device,
    )
    data._root_link_vel_w_ta = ProxyArray(data._root_link_vel_w.data)
    return data


@pytest.mark.parametrize("data_cls", [ArticulationData, RigidObjectData])
def test_lazy_derived_replay_refreshes_stale_timestamp_and_rebound_source(data_cls: type, cuda_device: str):
    """Replay should refresh stale data, while invalidation should admit a rebound source pointer."""
    cache = WarpLaunchCache(mode="replay", debug=True, device=cuda_device)
    source = _spatial_vectors(8, 1.0, cuda_device)
    data = _make_root_velocity_data(data_cls, source, cache, cuda_device)

    first = data.root_link_vel_w
    wp.synchronize_device(cuda_device)
    np.testing.assert_allclose(first.warp.numpy(), source.numpy())

    source.fill_(2.0)
    data._sim_timestamp = 2.0
    second = data.root_link_vel_w
    wp.synchronize_device(cuda_device)
    np.testing.assert_allclose(second.warp.numpy(), source.numpy())

    rebound_source = _spatial_vectors(8, 3.0, cuda_device)
    data._root_com_vel_w_ta = ProxyArray(rebound_source)
    cache.invalidate()
    data._sim_timestamp = 3.0
    third = data.root_link_vel_w
    wp.synchronize_device(cuda_device)
    np.testing.assert_allclose(third.warp.numpy(), rebound_source.numpy())


@pytest.mark.parametrize("data_cls", [ArticulationData, RigidObjectData])
def test_simulation_rebind_delegates_cache_drain(data_cls: type, monkeypatch: pytest.MonkeyPatch):
    """Simulation rebinding should delegate conditional synchronization to the cache."""
    events: list[tuple[str, str | None]] = []
    data = object.__new__(data_cls)
    data.device = "cuda:0"
    data._warp_launch = _ResetProbe(events)
    monkeypatch.setattr(wp, "synchronize_device", lambda device: pytest.fail(f"unexpected direct sync on {device}"))

    with pytest.raises(_RebindReset):
        data._create_simulation_bindings()

    assert events == [("reset", None)]


def test_adapter_teardown_delegates_cache_drain(monkeypatch: pytest.MonkeyPatch):
    """Adapter teardown should delegate conditional synchronization to the cache."""
    events: list[tuple[str, str | None]] = []
    adapter = object.__new__(NewtonActuatorAdapter)
    adapter._device = "cuda:0"
    adapter._warp_launch = _ResetProbe(events)
    monkeypatch.setattr(wp, "synchronize_device", lambda device: pytest.fail(f"unexpected direct sync on {device}"))

    with pytest.raises(_RebindReset):
        adapter._invalidate_launch_cache()

    assert events == [("reset", None)]


def test_newton_close_releases_graph_before_asset_stop_callbacks(monkeypatch: pytest.MonkeyPatch):
    """Newton shutdown should release graphs and drain work before asset caches are invalidated."""
    events: list[tuple[str, object]] = []
    graph = object()
    saved_graph = NewtonManager._graph
    saved_capture_pending = NewtonManager._graph_capture_pending
    saved_device = PhysicsManager._device
    try:
        NewtonManager._graph = graph
        NewtonManager._graph_capture_pending = True
        PhysicsManager._device = "cuda:0"
        monkeypatch.setattr(
            wp,
            "synchronize_device",
            lambda device: events.append(("synchronize", device)),
        )
        monkeypatch.setattr(
            PhysicsManager,
            "close",
            classmethod(lambda cls: events.append(("stop", NewtonManager._graph))),
        )
        monkeypatch.setattr(
            NewtonManager,
            "clear",
            classmethod(lambda cls: events.append(("clear", NewtonManager._graph))),
        )

        NewtonManager.close()

        assert NewtonManager._graph is None
        assert NewtonManager._graph_capture_pending is False
        assert events == [("synchronize", "cuda:0"), ("stop", None), ("clear", None)]
    finally:
        NewtonManager._graph = saved_graph
        NewtonManager._graph_capture_pending = saved_capture_pending
        PhysicsManager._device = saved_device


@pytest.mark.parametrize("use_cache", [False, True])
def test_adapter_step_zeros_current_effort_with_eager_or_replay(use_cache: bool, cuda_device: str):
    """Adapter zeroing should preserve eager behavior and replay correctly when enabled."""
    indices = wp.array([1, 3], dtype=wp.uint32, device=cuda_device)
    actuator = _NoOpActuator(indices)
    warp_launch = WarpLaunchCache(mode="replay" if use_cache else "eager", debug=True, device=cuda_device)
    adapter = NewtonActuatorAdapter(
        [actuator],
        num_envs=1,
        num_joints=4,
        dof_offset=0,
        device=cuda_device,
        warp_launch=warp_launch,
    )
    control = SimpleNamespace(joint_f=wp.full(4, 7.0, dtype=wp.float32, device=cuda_device))

    adapter.step(None, control, 0.01)
    wp.synchronize_device(cuda_device)
    np.testing.assert_allclose(control.joint_f.numpy(), [7.0, 0.0, 7.0, 0.0])

    control.joint_f.fill_(9.0)
    adapter.step(None, control, 0.01)
    wp.synchronize_device(cuda_device)
    np.testing.assert_allclose(control.joint_f.numpy(), [9.0, 0.0, 9.0, 0.0])

    assert bool(warp_launch._entries) is use_cache
    adapter._invalidate_launch_cache()
    assert not warp_launch._entries


@pytest.mark.parametrize("use_cache", [False, True])
def test_newton_manager_flag_selects_adapter_launch_mode(use_cache: bool, cuda_device: str):
    """The Newton manager should share an eager or replay launcher with its adapter."""
    had_active_flag = hasattr(NewtonManager, "_use_newton_actuators_active")
    saved_state = {
        "adapter": NewtonManager._adapter,
        "warp_launch": NewtonManager._warp_launch,
        "control": NewtonManager._control,
        "device": PhysicsManager._device,
        "cfg": PhysicsManager._cfg,
        "model": NewtonManager._model,
        "num_envs": NewtonManager._num_envs,
        "use_newton_actuators_active": getattr(NewtonManager, "_use_newton_actuators_active", False),
    }
    try:
        actuator = _NoOpActuator(wp.array([1, 3], dtype=wp.uint32, device=cuda_device))
        PhysicsManager._cfg = SimpleNamespace(use_warp_launch_cache=use_cache)
        PhysicsManager._device = cuda_device
        NewtonManager._model = SimpleNamespace(actuators=[actuator], joint_dof_count=4)
        NewtonManager._num_envs = 1
        NewtonManager._control = SimpleNamespace()
        NewtonManager._adapter = None
        NewtonManager._use_newton_actuators_active = False

        NewtonManager.activate_newton_actuator_path()

        assert NewtonManager._adapter is not None
        assert NewtonManager._adapter._warp_launch is NewtonManager._warp_launch
        assert NewtonManager._warp_launch._mode == ("auto" if use_cache else "eager")
    finally:
        if NewtonManager._adapter is not None:
            NewtonManager._adapter._invalidate_launch_cache()
        NewtonManager._adapter = saved_state["adapter"]
        NewtonManager._warp_launch = saved_state["warp_launch"]
        NewtonManager._control = saved_state["control"]
        PhysicsManager._device = saved_state["device"]
        PhysicsManager._cfg = saved_state["cfg"]
        NewtonManager._model = saved_state["model"]
        NewtonManager._num_envs = saved_state["num_envs"]
        if had_active_flag:
            NewtonManager._use_newton_actuators_active = saved_state["use_newton_actuators_active"]
        else:
            del NewtonManager._use_newton_actuators_active
