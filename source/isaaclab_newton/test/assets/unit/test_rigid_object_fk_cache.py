# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for Newton rigid-body FK cache invalidation."""

from types import SimpleNamespace

import pytest
from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data import RigidObjectCollectionData
from isaaclab_newton.physics import NewtonManager

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("data_type", [RigidObjectData, RigidObjectCollectionData])
def test_stale_fk_timestamp_forwards_once(data_type, monkeypatch) -> None:
    """The first same-timestamp body-pose read refreshes FK and later reads reuse it."""
    data = object.__new__(data_type)
    data._sim_timestamp = 2.0
    data._fk_timestamp = -1.0
    calls = []
    monkeypatch.setattr(NewtonManager, "forward", lambda: calls.append("forward"))

    data._ensure_fk_fresh()
    data._ensure_fk_fresh()

    assert calls == ["forward"]
    assert data._fk_timestamp == 2.0


@pytest.mark.parametrize("data_type", [RigidObjectData, RigidObjectCollectionData])
def test_pose_reset_invalidates_fk_for_the_view_articulation_ids(data_type, monkeypatch) -> None:
    """A root/body pose write invalidates Newton FK using the owning view mapping."""
    data = object.__new__(data_type)
    data._sim_timestamp = 3.0
    data._fk_timestamp = 3.0
    data._root_view = SimpleNamespace(articulation_ids="mapped-articulations")
    for name in (
        "_root_com_pose_w",
        "_root_link_vel_w",
        "_projected_gravity_b",
        "_heading_w",
        "_root_link_lin_vel_b",
        "_root_link_ang_vel_b",
        "_root_com_lin_vel_b",
        "_root_com_ang_vel_b",
        "_root_state_w",
        "_root_link_state_w",
        "_root_com_state_w",
        "_body_com_pose_w",
        "_body_link_vel_w",
        "_body_link_lin_vel_b",
        "_body_link_ang_vel_b",
        "_body_com_lin_vel_b",
        "_body_com_ang_vel_b",
        "_body_state_w",
        "_body_link_state_w",
        "_body_com_state_w",
    ):
        setattr(data, name, None)
    calls = []
    monkeypatch.setattr(NewtonManager, "invalidate_fk", lambda **kwargs: calls.append(kwargs))

    data._reset_pose(env_ids="selected-envs")

    assert data._fk_timestamp == -1.0
    assert calls == [
        {
            "env_mask": None,
            "env_ids": "selected-envs",
            "articulation_ids": "mapped-articulations",
        }
    ]
