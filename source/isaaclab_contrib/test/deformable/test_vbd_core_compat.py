# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility tests for contrib deformables using core VBD."""

from types import SimpleNamespace

from isaaclab_newton.physics import (
    NewtonManager,
    NewtonSoftContactCfg,
)
from isaaclab_newton.physics import (
    VBDSolverCfg as CoreVBDSolverCfg,
)
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager as CoreNewtonVBDManager

from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject

import isaaclab_contrib.deformable.deformable_object as deformable_object_module
from isaaclab_contrib.deformable import DeformableObject, VBDSolverCfg
from isaaclab_contrib.deformable.deformable_object import (
    add_registered_deformables_to_builder,
    clear_deformable_builder_hooks,
    install_deformable_builder_hooks,
    setup_registered_deformable_fabric_sync,
)
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg
from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager


def _fake_base_init(self, cfg) -> None:
    self.cfg = cfg
    self._DTYPE_TO_TORCH_TRAILING_DIMS = {}
    self._initialize_handle = None
    self._invalidate_initialize_handle = None
    self._prim_deletion_handle = None
    self._debug_vis_handle = None
    self._physics_ready_handle = None


def test_contrib_vbd_symbols_alias_core_implementations():
    """Retained contrib symbols resolve to their core implementations."""
    assert NewtonModelCfg is NewtonSoftContactCfg
    assert VBDSolverCfg is CoreVBDSolverCfg
    assert NewtonVBDManager is CoreNewtonVBDManager


def test_contrib_soft_contact_alias_uses_core_defaults():
    """The compatibility alias uses the core soft-contact defaults."""
    cfg = NewtonModelCfg()

    assert cfg.soft_contact_ke == 1.0e3
    assert cfg.soft_contact_kd == 10.0
    assert cfg.soft_contact_mu == 0.5


def test_builder_hook_lifecycle_preserves_third_party_state(monkeypatch):
    """Hook setup is idempotent and preserves third-party state."""

    def third_party_per_world(*args):
        pass

    def third_party_post_start(*args):
        pass

    registry_entry = object()
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [registry_entry])
    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", [third_party_per_world])
    monkeypatch.setattr(NewtonManager, "_post_start_simulation_hooks", [third_party_post_start])

    install_deformable_builder_hooks()
    install_deformable_builder_hooks()

    assert NewtonManager._deformable_registry == [registry_entry]
    assert NewtonManager._per_world_builder_hooks.count(add_registered_deformables_to_builder) == 1
    assert NewtonManager._post_start_simulation_hooks.count(setup_registered_deformable_fabric_sync) == 1

    clear_deformable_builder_hooks()

    assert NewtonManager._per_world_builder_hooks == [third_party_per_world]
    assert NewtonManager._post_start_simulation_hooks == [third_party_post_start]


def test_deformable_installs_hooks_before_registration(monkeypatch):
    """Asset construction installs hooks before registry mutation."""
    calls = []
    registry_entry = object()

    monkeypatch.setattr(BaseDeformableObject, "__init__", _fake_base_init)
    monkeypatch.setattr(deformable_object_module, "install_deformable_builder_hooks", lambda: calls.append("install"))
    monkeypatch.setattr(
        DeformableObject,
        "_register_deformable",
        lambda self: calls.append("register") or registry_entry,
    )

    asset = DeformableObject(SimpleNamespace())

    assert calls == ["install", "register"]
    assert asset._registry_entry is registry_entry


def test_deformable_registration_preserves_existing_assets(monkeypatch):
    """Constructing multiple deformables retains every registry entry."""
    entries = [object(), object()]

    def fake_register(self):
        entry = entries[len(NewtonManager._deformable_registry)]
        NewtonManager._deformable_registry.append(entry)
        return entry

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [])
    monkeypatch.setattr(BaseDeformableObject, "__init__", _fake_base_init)
    monkeypatch.setattr(DeformableObject, "_register_deformable", fake_register)

    first = DeformableObject(SimpleNamespace())
    second = DeformableObject(SimpleNamespace())

    assert NewtonManager._deformable_registry == entries
    assert first._registry_entry is entries[0]
    assert second._registry_entry is entries[1]


def test_deformable_cleanup_removes_only_its_registry_entry(monkeypatch):
    """Asset cleanup preserves other registered deformables."""
    first_entry = object()
    second_entry = object()
    asset = object.__new__(DeformableObject)
    asset._registry_entry = first_entry
    asset._physics_ready_handle = None
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [first_entry, second_entry])
    monkeypatch.setattr(BaseDeformableObject, "_clear_callbacks", lambda self: None)

    asset._clear_callbacks()

    assert NewtonManager._deformable_registry == [second_entry]
