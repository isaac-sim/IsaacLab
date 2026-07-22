# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.integration


# -------------------------------------------------------------------------------------
# Fragment metadata -- DeformableBodyFragment marker, OmniPhysicsDeformableBodyCfg
# -------------------------------------------------------------------------------------


def test_deformable_body_fragment_metadata_defaults():
    from isaaclab.sim.schemas import DeformableBodyFragment, OmniPhysicsDeformableBodyCfg, SchemaFragment

    cfg = OmniPhysicsDeformableBodyCfg(mass=2.0)
    assert isinstance(cfg, DeformableBodyFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == "omniphysics"
    assert type(cfg)._usd_applied_schema is None  # anchor applied by the backend manager
    assert type(cfg)._deformable_types == ("volume", "surface")
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"
    assert cfg.mass == 2.0 and cfg.deformable_body_enabled is None and cfg.kinematic_enabled is None


# -------------------------------------------------------------------------------------
# Family writers -- apply_volume/surface_deformable_properties
# -------------------------------------------------------------------------------------


def _make_volume_asset(stage, path="/World/Asset"):
    """Container Xform with a hand-authored single-tet TetMesh child (avoids pytetwild)."""
    from pxr import UsdGeom

    body = UsdGeom.Xform.Define(stage, path).GetPrim()
    sim = UsdGeom.TetMesh.Define(stage, f"{path}/tet").GetPrim()
    sim.GetAttribute("points").Set([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    sim.GetAttribute("tetVertexIndices").Set([(0, 1, 2, 3)])
    return body


class _StubManager:
    calls: list = []

    @classmethod
    def setup_deformable_body(cls, prim, deformable_type, sim_mesh_prim, vis_mesh_prim, stage=None):
        cls.calls.append((prim.GetPath().pathString, deformable_type, sim_mesh_prim.GetPath().pathString))
        prim.AddAppliedSchema("PhysicsDeformableBodyAPI")


def _fresh_sim_with_stub(monkeypatch):
    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.01))
    monkeypatch.setattr(sim, "physics_manager", _StubManager)
    _StubManager.calls = []
    return sim_utils.get_current_stage()


def test_apply_volume_deformable_creates_and_authors(monkeypatch):
    from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg, apply_volume_deformable_properties

    stage = _fresh_sim_with_stub(monkeypatch)
    _make_volume_asset(stage, "/World/A1")
    result = apply_volume_deformable_properties(
        "/World/A1", [OmniPhysicsDeformableBodyCfg(mass=0.5)], create_if_missing=True, stage=stage
    )
    assert result is True
    # the hand-authored TetMesh child is reused as the simulation mesh (no re-tetrahedralization)
    assert _StubManager.calls == [("/World/A1", "volume", "/World/A1/tet")]
    prim = stage.GetPrimAtPath("/World/A1")
    assert abs(prim.GetAttribute("omniphysics:mass").Get() - 0.5) < 1e-6
    # backend-neutral structural work happened in core
    sim_mesh = stage.GetPrimAtPath("/World/A1/tet")
    assert sim_mesh.GetTypeName() == "TetMesh"
    from pxr import UsdPhysics

    assert sim_mesh.HasAPI(UsdPhysics.CollisionAPI)


def test_apply_deformable_modify_path_skips_setup(monkeypatch):
    from pxr import UsdGeom

    from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg, apply_volume_deformable_properties

    stage = _fresh_sim_with_stub(monkeypatch)
    prim = UsdGeom.Xform.Define(stage, "/World/A2").GetPrim()
    prim.AddAppliedSchema("PhysicsDeformableBodyAPI")  # already anchored
    result = apply_volume_deformable_properties(
        "/World/A2", [OmniPhysicsDeformableBodyCfg(deformable_body_enabled=True)], stage=stage
    )
    assert result is True
    assert _StubManager.calls == []  # no re-setup on anchored prims
    assert prim.GetAttribute("omniphysics:deformableBodyEnabled").Get() is True


def test_apply_deformable_zero_targets_warns_and_returns_false(monkeypatch, caplog):
    from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg, apply_surface_deformable_properties

    stage = _fresh_sim_with_stub(monkeypatch)
    result = apply_surface_deformable_properties("/World/Nope", [OmniPhysicsDeformableBodyCfg(mass=1.0)], stage=stage)
    assert result is False


def test_apply_deformable_warns_on_type_mismatched_fragment(monkeypatch, caplog):
    import logging
    from typing import ClassVar

    from pxr import UsdGeom

    from isaaclab.sim.schemas import DeformableBodyFragment, apply_volume_deformable_properties
    from isaaclab.utils.configclass import configclass

    @configclass
    class _SurfaceOnlyCfg(DeformableBodyFragment):
        _usd_namespace: ClassVar[str | None] = "omniphysics"
        _deformable_types: ClassVar[tuple[str, ...]] = ("surface",)
        mass: float | None = None

    stage = _fresh_sim_with_stub(monkeypatch)
    prim = UsdGeom.Xform.Define(stage, "/World/A3").GetPrim()
    prim.AddAppliedSchema("PhysicsDeformableBodyAPI")
    with caplog.at_level(logging.WARNING):
        result = apply_volume_deformable_properties("/World/A3", [_SurfaceOnlyCfg(mass=1.0)], stage=stage)
    assert result is True  # warned, authored anyway
    assert any("volume" in rec.message for rec in caplog.records)
    assert prim.GetAttribute("omniphysics:mass").Get() is not None
