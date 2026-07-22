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
    import logging

    from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg, apply_surface_deformable_properties

    stage = _fresh_sim_with_stub(monkeypatch)
    with caplog.at_level(logging.WARNING):
        result = apply_surface_deformable_properties(
            "/World/Nope", [OmniPhysicsDeformableBodyCfg(mass=1.0)], stage=stage
        )
    assert result is False
    assert any("No deformable-body targets matched" in rec.message for rec in caplog.records)


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


# -------------------------------------------------------------------------------------
# Spawner slots -- volume/surface_deformable_props on mesh and USD-file spawners
# -------------------------------------------------------------------------------------


def test_deformable_slot_exclusivity_raises():
    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg

    cfg = sim_utils.MeshCuboidCfg(
        size=(0.1, 0.1, 0.1),
        volume_deformable_props={"": [OmniPhysicsDeformableBodyCfg()]},
        surface_deformable_props={"": [OmniPhysicsDeformableBodyCfg()]},
    )
    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    with pytest.raises(ValueError, match="one deformable"):
        cfg.func("/World/Bad", cfg)


def test_mesh_surface_deformable_spawn_with_collision_props(monkeypatch, caplog):
    """Surface slot end-to-end on the mesh path (no tetrahedralization needed), with collision
    offsets riding the collision family keyed to the sim mesh.

    With a deformable ``physics_material`` configured, the spawner suppresses the writer's
    missing-material check (the material binds after authoring), so the happy path must not
    emit the "without a physics material binding" warning.
    """
    import logging

    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg, UsdPhysicsCollisionCfg
    from isaaclab.sim.spawners.materials import OmniPhysicsSurfaceDeformableMaterialCfg

    stage = _fresh_sim_with_stub(monkeypatch)
    cfg = sim_utils.MeshCuboidCfg(
        size=(0.1, 0.1, 0.1),
        surface_deformable_props={"": [OmniPhysicsDeformableBodyCfg(mass=0.2)]},
        physics_material=[OmniPhysicsSurfaceDeformableMaterialCfg(surface_thickness=0.01)],
        collision_props={"sim_mesh": [UsdPhysicsCollisionCfg(collision_enabled=True)]},
    )
    with caplog.at_level(logging.WARNING):
        cfg.func("/World/Cloth", cfg)
    prim = stage.GetPrimAtPath("/World/Cloth")
    assert _StubManager.calls and _StubManager.calls[0][1] == "surface"
    assert abs(prim.GetAttribute("omniphysics:mass").Get() - 0.2) < 1e-6
    sim_mesh = stage.GetPrimAtPath("/World/Cloth/sim_mesh")
    assert sim_mesh.IsValid()
    assert sim_mesh.GetAttribute("physics:collisionEnabled").Get() is True
    # a bound deformable material must silence the missing-material warning on the happy path
    assert not any("without a physics material binding" in rec.message for rec in caplog.records)


def test_usd_file_volume_deformable_spawn(monkeypatch, tmp_path):
    """Volume slot on the USD-file path with a pre-tetrahedralized asset (no pytetwild)."""
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas import OmniPhysicsDeformableBodyCfg

    asset = str(tmp_path / "tet_asset.usda")
    layer_stage = Usd.Stage.CreateNew(asset)
    UsdGeom.Xform.Define(layer_stage, "/Asset")
    tet = UsdGeom.TetMesh.Define(layer_stage, "/Asset/tet").GetPrim()
    tet.GetAttribute("points").Set([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    tet.GetAttribute("tetVertexIndices").Set([(0, 1, 2, 3)])
    layer_stage.SetDefaultPrim(layer_stage.GetPrimAtPath("/Asset"))
    layer_stage.Save()

    stage = _fresh_sim_with_stub(monkeypatch)
    cfg = sim_utils.UsdFileCfg(usd_path=asset, volume_deformable_props={"": [OmniPhysicsDeformableBodyCfg()]})
    cfg.func("/World/Soft", cfg)
    assert stage is sim_utils.get_current_stage()
    # the pre-authored TetMesh is reused in place as the simulation mesh (path keeps its name)
    assert _StubManager.calls == [("/World/Soft", "volume", "/World/Soft/tet")]
