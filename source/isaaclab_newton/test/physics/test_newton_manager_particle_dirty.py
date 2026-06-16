# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for NewtonManager particle dirty-flag semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaaclab_newton.physics.newton_manager import NewtonManager


@pytest.fixture(autouse=True)
def _reset_particle_dirty_state():
    NewtonManager._particles_dirty = False
    NewtonManager._usdrt_stage = None
    NewtonManager._particle_visual_prims = {}
    NewtonManager._deformable_registry = []
    NewtonManager._state_0 = None
    yield
    NewtonManager._particles_dirty = False
    NewtonManager._usdrt_stage = None
    NewtonManager._particle_visual_prims = {}
    NewtonManager._deformable_registry = []
    NewtonManager._state_0 = None


def test_sync_particles_to_usd_preserves_dirty_without_fabric(monkeypatch: pytest.MonkeyPatch):
    """No-op Fabric sync must not clear the dirty flag for kitless render consumers."""
    NewtonManager._particles_dirty = True
    NewtonManager._state_0 = SimpleNamespace(particle_q=object())
    monkeypatch.setattr(NewtonManager, "_sync_fabric_mesh_particles", classmethod(lambda cls: False))
    monkeypatch.setattr(NewtonManager, "_sync_particle_points_prims", classmethod(lambda cls: False))

    NewtonManager.sync_particles_to_usd()

    assert NewtonManager.particles_dirty() is True


def test_sync_particles_to_usd_clears_dirty_after_fabric_sync(monkeypatch: pytest.MonkeyPatch):
    """Fabric mesh sync clears the dirty flag when no throttled points prims remain."""
    NewtonManager._particles_dirty = True
    NewtonManager._state_0 = SimpleNamespace(particle_q=object())
    monkeypatch.setattr(NewtonManager, "_sync_fabric_mesh_particles", classmethod(lambda cls: True))
    monkeypatch.setattr(NewtonManager, "_sync_particle_points_prims", classmethod(lambda cls: False))

    NewtonManager.sync_particles_to_usd()

    assert NewtonManager.particles_dirty() is False


def test_has_surface_deformable_registry_entries():
    """Surface deformable registry entries are detected for post-step dirty marking."""
    NewtonManager._deformable_registry = [
        SimpleNamespace(deformable_type="volume"),
        SimpleNamespace(deformable_type="surface"),
    ]

    assert NewtonManager._has_surface_deformable_registry_entries() is True

    NewtonManager._deformable_registry = [SimpleNamespace(deformable_type="volume")]

    assert NewtonManager._has_surface_deformable_registry_entries() is False
