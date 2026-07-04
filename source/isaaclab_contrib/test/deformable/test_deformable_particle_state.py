# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-authored contrib deformable particle state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import NewtonManager

from isaaclab_contrib.deformable import deformable_object as deformable_object_module
from isaaclab_contrib.deformable.deformable_object import DeformableObject


class _DeformableObjectUnderTest(DeformableObject):
    """Deformable object with teardown disabled for lightweight method tests."""

    def __del__(self) -> None:
        pass


def _make_deformable(monkeypatch) -> DeformableObject:
    asset = object.__new__(_DeformableObjectUnderTest)
    asset._device = "cpu"
    asset._num_instances = 3
    asset._particles_per_body = 1
    asset._particle_offsets = object()
    asset._ALL_INDICES = object()
    asset._ALL_ENV_MASK = object()
    asset._data = SimpleNamespace(
        _nodal_pos_w=SimpleNamespace(timestamp=1.0),
        _nodal_vel_w=SimpleNamespace(timestamp=1.0),
        _nodal_state_w=SimpleNamespace(timestamp=1.0),
        _root_pos_w=SimpleNamespace(timestamp=1.0),
        _root_vel_w=SimpleNamespace(timestamp=1.0),
    )
    state = SimpleNamespace(particle_q=object(), particle_qd=object())
    monkeypatch.setattr(asset, "_iter_particle_states", lambda: iter((state,)))
    monkeypatch.setattr(asset, "assert_shape_and_dtype", lambda *args, **kwargs: None)
    monkeypatch.setattr(deformable_object_module.wp, "launch", lambda *args, **kwargs: None)
    return asset


@pytest.mark.parametrize(
    ("method_name", "selector_name"),
    [
        ("write_nodal_pos_to_sim_index", "env_ids"),
        ("write_nodal_velocity_to_sim_index", "env_ids"),
        ("write_nodal_state_to_sim_mask", "env_mask"),
        ("write_nodal_pos_to_sim_mask", "env_mask"),
        ("write_nodal_velocity_to_sim_mask", "env_mask"),
    ],
)
def test_deformable_writers_publish_the_resolved_selection(monkeypatch, method_name, selector_name):
    """Every concrete deformable state writer publishes its resolved selector."""
    asset = _make_deformable(monkeypatch)
    caller_selection = object()
    resolved_selection = SimpleNamespace(shape=(3,))
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(asset, "_resolve_env_ids", lambda value: resolved_selection)
    monkeypatch.setattr(asset, "_resolve_mask", lambda value, full_mask: resolved_selection)
    monkeypatch.setattr(
        NewtonManager,
        "invalidate_particles",
        classmethod(lambda cls, **kwargs: calls.append(kwargs)),
    )

    getattr(asset, method_name)(object(), **{selector_name: caller_selection})

    assert calls == [{selector_name: resolved_selection}]


def test_kinematic_control_dirties_caches_and_render_without_publishing_reset_state(monkeypatch):
    """Per-step kinematic controls cannot trigger the authored-state reset policy."""
    asset = _make_deformable(monkeypatch)
    asset._data.nodal_kinematic_target = SimpleNamespace(warp=object())
    asset._default_particle_inv_mass = object()
    asset._default_particle_flags = object()
    model = SimpleNamespace(particle_inv_mass=object(), particle_flags=object())
    events: list[str] = []

    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: model))
    monkeypatch.setattr(NewtonManager, "_mark_particles_dirty", classmethod(lambda cls: events.append("render")))
    monkeypatch.setattr(
        NewtonManager,
        "invalidate_particles",
        classmethod(lambda cls, **kwargs: events.append("authored")),
    )
    monkeypatch.setattr(asset, "_invalidate_nodal_state_cache", lambda: events.append("cache"))

    asset.write_data_to_sim()

    assert events == ["cache", "render"]
