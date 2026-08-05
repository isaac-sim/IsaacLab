# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core Newton VBD integration."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from isaaclab_newton.physics import NewtonCfg, NewtonManager


def test_vbd_symbols_are_exported_from_core():
    """Core exports the VBD manager and configuration."""
    physics = importlib.import_module("isaaclab_newton.physics")

    assert physics.NewtonVBDManager.__name__ == "NewtonVBDManager"
    assert physics.VBDSolverCfg.__name__ == "VBDSolverCfg"
    assert physics.VBDSolverCfg().class_type.__name__ == "NewtonVBDManager"
    assert issubclass(physics.NewtonVBDManager, NewtonManager)


def test_soft_contact_cfg_defaults_match_newton():
    """Soft-contact defaults match the pinned Newton model."""
    physics = importlib.import_module("isaaclab_newton.physics")
    cfg = physics.NewtonSoftContactCfg()

    assert cfg.soft_contact_ke == 1.0e3
    assert cfg.soft_contact_kd == 10.0
    assert cfg.soft_contact_mu == 0.5
    assert NewtonCfg().soft_contact_cfg is None


def test_vbd_excludes_registered_deformable_meshes(monkeypatch):
    """VBD excludes registered simulation and visual meshes from USD import."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []
    registry = [
        SimpleNamespace(sim_mesh_prim_path="/World/cloth/sim", vis_mesh_prim_path="/World/cloth/visual"),
        SimpleNamespace(sim_mesh_prim_path="/World/soft/sim", vis_mesh_prim_path="/World/soft/visual"),
    ]

    def instantiate_builder(cls, ignore_paths=()):
        events.append(("import", cls, list(ignore_paths)))
        NewtonManager._builder = SimpleNamespace(color=lambda: events.append(("color",)))

    monkeypatch.setattr(NewtonManager, "instantiate_builder_from_stage", classmethod(instantiate_builder))
    monkeypatch.setattr(physics.NewtonVBDManager, "_deformable_registry", registry)

    physics.NewtonVBDManager.instantiate_builder_from_stage()

    assert events == [
        (
            "import",
            physics.NewtonVBDManager,
            ["/World/cloth/sim", "/World/cloth/visual", "/World/soft/sim", "/World/soft/visual"],
        ),
        ("color",),
    ]


def test_vbd_rebuilds_particle_bvh_before_physics_step(monkeypatch):
    """VBD rebuilds its particle BVH before the base physics step."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []
    state = object()

    class Solver:
        def rebuild_bvh(self, solver_state):
            events.append(("rebuild", solver_state))

    def simulate_physics_only(cls):
        events.append(("step", cls))

    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(simulate_physics_only))
    monkeypatch.setattr(physics.NewtonVBDManager, "_model", SimpleNamespace(particle_count=1))
    monkeypatch.setattr(physics.NewtonVBDManager, "_solver", Solver())
    monkeypatch.setattr(physics.NewtonVBDManager, "_state_0", state)

    physics.NewtonVBDManager._simulate_physics_only()

    assert events == [("rebuild", state), ("step", physics.NewtonVBDManager)]
