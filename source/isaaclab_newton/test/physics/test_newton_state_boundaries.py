# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for asset-write and direct-read Newton state boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data import RigidObjectCollectionData
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.physics._authored_state_transaction import AuthoredStateTransaction
from isaaclab_newton.sensors.joint_wrench import joint_wrench_sensor as joint_wrench_module
from isaaclab_newton.sensors.joint_wrench.joint_wrench_sensor import JointWrenchSensor


@pytest.mark.parametrize("data_type", [ArticulationData, RigidObjectData, RigidObjectCollectionData])
@pytest.mark.parametrize("reset_method", ["_reset_pose", "_reset_velocity"])
def test_skip_cache_invalidation_still_publishes_state_write(monkeypatch, data_type, reset_method):
    """skip_forward may preserve asset caches but cannot hide authored state from the solver."""
    articulation_ids = object()
    env_ids = object()
    calls: list[dict] = []
    data = object.__new__(data_type)
    data._root_view = SimpleNamespace(articulation_ids=articulation_ids)
    data._fk_timestamp = 17.0
    monkeypatch.setattr(
        NewtonManager,
        "invalidate_fk",
        classmethod(lambda cls, **kwargs: calls.append(kwargs)),
    )

    getattr(data, reset_method)(env_ids=env_ids, invalidate_cache=False)

    assert data._fk_timestamp == 17.0
    expected = {"env_mask": None, "env_ids": env_ids, "articulation_ids": articulation_ids}
    if data_type is RigidObjectCollectionData:
        expected["articulation_selection"] = None
    assert calls == [expected]


@pytest.mark.parametrize("reset_method", ["_reset_pose", "_reset_velocity"])
def test_collection_publishes_selected_articulation_columns(monkeypatch, reset_method):
    """Partial collection writes preserve their body-column selection."""
    articulation_ids = object()
    body_ids = object()
    calls: list[dict] = []
    data = object.__new__(RigidObjectCollectionData)
    data._root_view = SimpleNamespace(articulation_ids=articulation_ids)
    monkeypatch.setattr(
        NewtonManager,
        "invalidate_fk",
        classmethod(lambda cls, **kwargs: calls.append(kwargs)),
    )

    getattr(data, reset_method)(body_ids=body_ids, invalidate_cache=False)

    assert calls == [
        {
            "env_mask": None,
            "env_ids": None,
            "articulation_ids": articulation_ids,
            "articulation_selection": body_ids,
        }
    ]


def test_empty_collection_selection_does_not_publish_work(monkeypatch):
    """A no-op collection write does not trigger backend synchronization."""
    transaction = AuthoredStateTransaction(1, 1, "cpu", lambda worlds, articulations: None)
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)
    NewtonManager.invalidate_fk(
        env_ids=wp.array([0], dtype=wp.int32, device="cpu"),
        articulation_ids=wp.array([[0]], dtype=wp.int32, device="cpu"),
        articulation_selection=wp.empty(0, dtype=wp.int32, device="cpu"),
    )

    assert transaction.world_mask.numpy().tolist() == [False]
    assert transaction._pending.numpy().tolist() == [0]


def test_joint_wrench_read_flushes_pending_state(monkeypatch):
    """Direct body-state sensors establish the same coherent read boundary as asset data."""
    events: list[str] = []
    sensor = object.__new__(JointWrenchSensor)
    sensor._sim_bind_body_parent_f = object()
    sensor._sim_bind_body_q = object()
    sensor._sim_bind_body_com = object()
    sensor._sim_bind_joint_X_c = object()
    sensor._joint_child = object()
    sensor._num_envs = 1
    sensor._num_joints = 1
    sensor._timestamp = object()
    sensor._device = "cpu"
    sensor._data = SimpleNamespace(_force=object(), _torque=object())
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None
    monkeypatch.setattr(NewtonManager, "forward", classmethod(lambda cls: events.append("forward")))
    monkeypatch.setattr(joint_wrench_module.wp, "launch", lambda *args, **kwargs: events.append("launch"))

    sensor._update_buffers_impl(object())

    assert events == ["forward", "launch"]
