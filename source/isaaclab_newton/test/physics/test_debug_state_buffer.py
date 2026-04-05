# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none, reportArgumentType=none

"""Tests for debug state buffer (NaN incident replay).

Uses real warp arrays on the default device so torch interop paths are tested.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab_newton.physics.debug_state_buffer import DebugStateBuffer

wp.init()

NUM_BODIES = 4
NUM_JOINTS = 3


class FakeState:
    """Minimal stand-in for newton.State backed by real warp arrays."""

    def __init__(self, num_bodies: int = NUM_BODIES, num_joints: int = NUM_JOINTS, device: str = "cpu") -> None:
        self.body_q = wp.zeros(num_bodies, dtype=wp.transformf, device=device)
        self.body_qd = wp.zeros(num_bodies, dtype=wp.spatial_vectorf, device=device)
        self.joint_q = wp.zeros(num_joints, dtype=wp.float32, device=device)
        self.joint_qd = wp.zeros(num_joints, dtype=wp.float32, device=device)

    def assign(self, other: FakeState) -> None:
        wp.copy(self.body_q, other.body_q)
        wp.copy(self.body_qd, other.body_qd)
        wp.copy(self.joint_q, other.joint_q)
        wp.copy(self.joint_qd, other.joint_qd)

    def numpy(self):  # noqa: D102
        raise NotImplementedError


class FakeModel:
    """Minimal stand-in for newton.Model."""

    def __init__(
        self,
        num_bodies: int = NUM_BODIES,
        num_joints: int = NUM_JOINTS,
        world_count: int = 0,
        device: str = "cpu",
    ) -> None:
        self._num_bodies = num_bodies
        self._num_joints = num_joints
        self._device = device
        self.world_count = world_count
        if world_count > 1:
            bodies_per_world = num_bodies // world_count
            joints_per_world = num_joints // world_count
            self.body_world_start = wp.array(
                [i * bodies_per_world for i in range(world_count)] + [num_bodies],
                dtype=wp.int32,
                device="cpu",
            )
            self.joint_coord_world_start = wp.array(
                [i * joints_per_world for i in range(world_count)] + [num_joints],
                dtype=wp.int32,
                device="cpu",
            )
            self.joint_dof_world_start = wp.array(
                [i * joints_per_world for i in range(world_count)] + [num_joints],
                dtype=wp.int32,
                device="cpu",
            )

    def state(self) -> FakeState:
        return FakeState(self._num_bodies, self._num_joints, self._device)


def _inject_nan_body_q(state: FakeState, body_idx: int, component: int = 0) -> None:
    """Set one element of body_q to NaN on the device."""
    t = wp.to_torch(state.body_q)
    t[body_idx, component] = float("nan")


def _inject_nan_joint_qd(state: FakeState, joint_idx: int) -> None:
    t = wp.to_torch(state.joint_qd)
    t[joint_idx] = float("nan")


# ------------------------------------------------------------------
# Basic buffer tests
# ------------------------------------------------------------------


def test_buffer_size_clamped():
    model = FakeModel()
    buf = DebugStateBuffer(model, 5)
    assert buf.size == 5

    buf_big = DebugStateBuffer(model, 999999)
    assert buf_big.size == 2000


@pytest.mark.parametrize("size", [1, 3, 10])
def test_size_property(size):
    model = FakeModel()
    buf = DebugStateBuffer(model, size)
    assert buf.size == size


def test_step_rolls_index():
    model = FakeModel()
    buf = DebugStateBuffer(model, 3)
    state = FakeState()
    for i in range(7):
        buf.step(state, sim_time=float(i))
    assert buf._write_idx == 7 % 3


def test_no_export_on_clean_state():
    model = FakeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 3, export_path=tmpdir)
        state = FakeState()
        for i in range(5):
            buf.step(state, sim_time=float(i) * 0.01)
        assert len(list(Path(tmpdir).glob("nan_replay_*.npz"))) == 0


def test_export_on_nan_body_q():
    model = FakeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 4, export_path=tmpdir)
        clean = FakeState()
        buf.step(clean, sim_time=0.0)
        buf.step(clean, sim_time=0.01)

        bad = FakeState()
        _inject_nan_body_q(bad, 0)
        buf.step(bad, sim_time=0.02)

        files = list(Path(tmpdir).glob("nan_replay_*.npz"))
        assert len(files) == 1
        data = np.load(files[0])
        assert data["buffer_size"] == 4
        assert float(data["sim_time"]) == pytest.approx(0.02)
        assert np.isnan(data["body_q"]).any()


def test_export_on_nan_joint_qd():
    model = FakeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 2, export_path=tmpdir)
        bad = FakeState()
        _inject_nan_joint_qd(bad, 1)
        buf.step(bad, sim_time=1.0)
        files = list(Path(tmpdir).glob("nan_replay_*.npz"))
        assert len(files) == 1
        data = np.load(files[0])
        assert np.isnan(data["joint_qd"]).any()


def test_per_env_nan_detection():
    """With 2 worlds, only the env containing NaN should be reported."""
    model = FakeModel(num_bodies=4, num_joints=2, world_count=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 2, export_path=tmpdir, export_envs_only=True)
        assert buf._world_count == 2

        bad = FakeState(num_bodies=4, num_joints=2)
        _inject_nan_body_q(bad, body_idx=2, component=0)  # env 1 (bodies 2-3)

        buf.step(bad, sim_time=0.5)

        files = list(Path(tmpdir).glob("nan_replay_*.npz"))
        assert len(files) == 1
        data = np.load(files[0])
        env_ids = data["exported_env_ids"]
        assert len(env_ids) == 1
        assert env_ids[0] == 1


def test_clear_resets():
    model = FakeModel()
    buf = DebugStateBuffer(model, 5)
    assert buf.size == 5
    buf.clear()
    assert buf.size == 0
    assert buf._ring == []


def test_step_after_clear_is_noop():
    """step() after clear() should not crash."""
    model = FakeModel()
    buf = DebugStateBuffer(model, 2)
    buf.clear()
    state = FakeState()
    buf.step(state, sim_time=0.0)


def test_export_envs_only_false_exports_all():
    """With export_envs_only=False, full state is exported even with world layout."""
    model = FakeModel(num_bodies=4, num_joints=2, world_count=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 2, export_path=tmpdir, export_envs_only=False)
        bad = FakeState(num_bodies=4, num_joints=2)
        _inject_nan_body_q(bad, body_idx=3)
        buf.step(bad, sim_time=0.1)

        files = list(Path(tmpdir).glob("nan_replay_*.npz"))
        assert len(files) == 1
        data = np.load(files[0])
        assert "exported_env_ids" not in data
        assert data["body_q"].shape[1] == 4  # all bodies, not sliced


# ------------------------------------------------------------------
# Deduplication: already-NaN env_ids are suppressed
# ------------------------------------------------------------------


def test_same_env_nan_consecutive_steps_single_export():
    """When the same env NaNs on consecutive steps, only one npz is produced."""
    model = FakeModel(num_bodies=4, num_joints=2, world_count=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 3, export_path=tmpdir, max_exports=10)
        bad = FakeState(num_bodies=4, num_joints=2)
        _inject_nan_body_q(bad, body_idx=2)  # env 1

        for t in range(5):
            buf.step(bad, sim_time=float(t) * 0.01)

        files = list(Path(tmpdir).glob("nan_replay_*.npz"))
        assert len(files) == 1, f"Expected 1 export, got {len(files)}"
        assert 1 in buf._exported_envs


def test_second_env_nan_after_first_gets_separate_export():
    """A new env NaN'ing after the first env was already exported gets its own export."""
    model = FakeModel(num_bodies=4, num_joints=2, world_count=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 2, export_path=tmpdir, max_exports=10)

        bad_env1 = FakeState(num_bodies=4, num_joints=2)
        _inject_nan_body_q(bad_env1, body_idx=2)  # env 1
        buf.step(bad_env1, sim_time=0.01)
        assert len(list(Path(tmpdir).glob("nan_replay_*.npz"))) == 1

        # Ensure timestamp differs
        time.sleep(0.01)

        bad_both = FakeState(num_bodies=4, num_joints=2)
        _inject_nan_body_q(bad_both, body_idx=0)  # env 0
        _inject_nan_body_q(bad_both, body_idx=2)  # env 1 (already exported)
        buf.step(bad_both, sim_time=0.02)

        files = list(Path(tmpdir).glob("nan_replay_*.npz"))
        assert len(files) == 2, f"Expected 2 exports, got {len(files)}"
        assert buf._exported_envs == {0, 1}


# ------------------------------------------------------------------
# max_exports and nan_halt
# ------------------------------------------------------------------


def test_nan_halt_after_max_exports_default():
    """With default max_exports=1, nan_halt is True after first export."""
    model = FakeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 2, export_path=tmpdir)
        assert not buf.nan_halt

        bad = FakeState()
        _inject_nan_body_q(bad, 0)
        buf.step(bad, sim_time=0.0)

        assert buf.nan_halt
        assert buf._export_count == 1


def test_nan_halt_stops_recording():
    """After nan_halt, step() is a no-op (no more exports)."""
    model = FakeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 2, export_path=tmpdir, max_exports=1)
        bad = FakeState()
        _inject_nan_body_q(bad, 0)
        buf.step(bad, sim_time=0.0)
        assert buf.nan_halt

        write_idx_before = buf._write_idx
        buf.step(bad, sim_time=0.01)
        assert buf._write_idx == write_idx_before  # no advancement


def test_max_exports_allows_multiple():
    """With max_exports=3, exactly 3 exports are allowed."""
    model = FakeModel(num_bodies=6, num_joints=3, world_count=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 2, export_path=tmpdir, max_exports=3)

        for env_idx in range(3):
            time.sleep(0.01)
            bad = FakeState(num_bodies=6, num_joints=3)
            _inject_nan_body_q(bad, body_idx=env_idx * 2)  # each env's first body
            buf.step(bad, sim_time=float(env_idx) * 0.01)

        files = list(Path(tmpdir).glob("nan_replay_*.npz"))
        assert len(files) == 3
        assert buf.nan_halt
        assert buf._export_count == 3


# ------------------------------------------------------------------
# Scene exporter
# ------------------------------------------------------------------


def test_scene_exporter_called_on_nan():
    """scene_exporter callable is invoked with (usd_path, env_ids) on NaN."""
    calls: list[tuple[str, list[int]]] = []

    def mock_exporter(usd_path: str, env_ids: list[int]) -> None:
        calls.append((usd_path, env_ids))

    model = FakeModel(num_bodies=4, num_joints=2, world_count=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(
            model, 2, export_path=tmpdir, scene_exporter=mock_exporter, max_exports=5,
        )
        bad = FakeState(num_bodies=4, num_joints=2)
        _inject_nan_body_q(bad, body_idx=2)  # env 1
        buf.step(bad, sim_time=0.5)

        assert len(calls) == 1
        usd_path, env_ids = calls[0]
        assert usd_path.endswith(".usd")
        assert env_ids == [1]


def test_scene_exporter_not_called_without_nan():
    """scene_exporter is not called when no NaN."""
    calls: list = []

    def mock_exporter(usd_path: str, env_ids: list[int]) -> None:
        calls.append(1)

    model = FakeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 2, export_path=tmpdir, scene_exporter=mock_exporter)
        clean = FakeState()
        buf.step(clean, sim_time=0.0)
        assert len(calls) == 0


def test_scene_exporter_single_env_empty_ids():
    """For single-env (no world layout), scene_exporter receives empty env_ids."""
    calls: list[tuple[str, list[int]]] = []

    def mock_exporter(usd_path: str, env_ids: list[int]) -> None:
        calls.append((usd_path, env_ids))

    model = FakeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 2, export_path=tmpdir, scene_exporter=mock_exporter)
        bad = FakeState()
        _inject_nan_body_q(bad, 0)
        buf.step(bad, sim_time=0.0)

        assert len(calls) == 1
        assert calls[0][1] == []


def test_scene_exporter_failure_does_not_crash():
    """If scene_exporter raises, export still completes (npz is written)."""

    def broken_exporter(usd_path: str, env_ids: list[int]) -> None:
        raise RuntimeError("USD not available")

    model = FakeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        buf = DebugStateBuffer(model, 2, export_path=tmpdir, scene_exporter=broken_exporter)
        bad = FakeState()
        _inject_nan_body_q(bad, 0)
        buf.step(bad, sim_time=0.0)

        files = list(Path(tmpdir).glob("nan_replay_*.npz"))
        assert len(files) == 1
        assert buf.nan_halt


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
