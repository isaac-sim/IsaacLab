# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused FK invalidation tests for Newton articulation data."""

from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import NewtonManager as SimulationManager

pytestmark = pytest.mark.unit


def test_stale_articulation_fk_forwards_and_republishes_ordered_state_once(monkeypatch) -> None:
    """A stale articulation FK stamp must forward and refresh public-order shadows exactly once."""
    calls = []
    data = object.__new__(ArticulationData)
    data._sim_timestamp = 2.0
    data._fk_timestamp = -1.0
    data._refresh_user_order_body_state = lambda: calls.append("refresh")
    monkeypatch.setattr(SimulationManager, "forward", classmethod(lambda cls: calls.append("forward")))

    data._ensure_fk_fresh()
    data._ensure_fk_fresh()

    assert calls == ["forward", "refresh"]
    assert data._fk_timestamp == 2.0


def test_joint_pose_reset_invalidates_only_selected_articulation_instances(monkeypatch) -> None:
    """Joint writes must invalidate Newton FK for the selected instances and owning view IDs."""
    invalidations = []
    data = object.__new__(ArticulationData)
    data._fk_timestamp = 4.0
    data._root_view = SimpleNamespace(articulation_ids=wp.array([3, 8], dtype=wp.int32, device="cpu"))
    for name in (
        "_root_com_pose_w",
        "_body_com_pose_w",
        "_root_link_vel_w",
        "_body_link_vel_w",
        "_projected_gravity_b",
        "_heading_w",
        "_root_link_lin_vel_b",
        "_root_link_ang_vel_b",
        "_root_com_lin_vel_b",
        "_root_com_ang_vel_b",
        "_root_state_w",
        "_root_link_state_w",
        "_root_com_state_w",
        "_body_state_w",
        "_body_link_state_w",
        "_body_com_state_w",
    ):
        setattr(data, name, None)
    monkeypatch.setattr(
        SimulationManager,
        "invalidate_fk",
        classmethod(lambda cls, **kwargs: invalidations.append(kwargs)),
    )
    env_ids = wp.array([1], dtype=wp.int32, device="cpu")

    data._reset_pose(env_ids=env_ids)

    assert data._fk_timestamp == -1.0
    assert len(invalidations) == 1
    assert invalidations[0]["env_ids"] is env_ids
    assert invalidations[0]["env_mask"] is None
    assert invalidations[0]["articulation_ids"] is data._root_view.articulation_ids
