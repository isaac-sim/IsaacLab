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

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.integration


def _make_xform(stage, path="/World/Body"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


# -------------------------------------------------------------------------------------
# CollisionFragment marker + UsdPhysicsCollisionCfg
# -------------------------------------------------------------------------------------


def test_collision_fragment_metadata_defaults():
    from isaaclab.sim.schemas import CollisionFragment, SchemaFragment, UsdPhysicsCollisionCfg

    cfg = UsdPhysicsCollisionCfg(collision_enabled=True)
    assert isinstance(cfg, CollisionFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == "physics"
    assert type(cfg)._usd_applied_schema is None  # anchor applies CollisionAPI, not the fragment
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"
    assert cfg.collision_enabled is True


# -------------------------------------------------------------------------------------
# UsdPhysicsCollisionCfg writes its physics namespace via apply_namespaced
# -------------------------------------------------------------------------------------


def test_usd_physics_collision_fragment_writes_physics_namespace():
    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage)
    UsdPhysics.CollisionAPI.Apply(prim)
    apply_namespaced(UsdPhysicsCollisionCfg(collision_enabled=True), "/World/Body", stage)
    assert prim.GetAttribute("physics:collisionEnabled").Get() is True


# -------------------------------------------------------------------------------------
# PhysxCollisionCfg (isaaclab_physx)
# -------------------------------------------------------------------------------------


def test_physx_collision_fragment_writes_physx_namespace():
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/C2")
    UsdPhysics.CollisionAPI.Apply(prim)
    apply_namespaced(
        PhysxCollisionCfg(contact_offset=0.02, rest_offset=0.0, torsional_patch_radius=0.1), "/World/C2", stage
    )
    assert abs(prim.GetAttribute("physxCollision:contactOffset").Get() - 0.02) < 1e-6
    assert abs(prim.GetAttribute("physxCollision:restOffset").Get() - 0.0) < 1e-6
    assert abs(prim.GetAttribute("physxCollision:torsionalPatchRadius").Get() - 0.1) < 1e-6


# -------------------------------------------------------------------------------------
# NewtonCollisionCfg (isaaclab_newton)
# -------------------------------------------------------------------------------------


def test_newton_collision_fragment_writes_newton_namespace():
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/C3")
    UsdPhysics.CollisionAPI.Apply(prim)
    apply_namespaced(NewtonCollisionCfg(contact_margin=0.01, contact_gap=0.005), "/World/C3", stage)
    assert abs(prim.GetAttribute("newton:contactMargin").Get() - 0.01) < 1e-6
    assert abs(prim.GetAttribute("newton:contactGap").Get() - 0.005) < 1e-6


# -------------------------------------------------------------------------------------
# apply_collision_properties dispatch (explicit anchor creation + multi-namespace)
# -------------------------------------------------------------------------------------


def test_apply_collision_properties_composes_namespaces():
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg

    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, apply_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/C4")
    apply_collision_properties(
        "/World/C4",
        [
            UsdPhysicsCollisionCfg(collision_enabled=True),
            PhysxCollisionCfg(contact_offset=0.02),
            NewtonCollisionCfg(contact_margin=0.01),
        ],
        create_if_missing=True,
        stage=stage,
    )
    prim = stage.GetPrimAtPath("/World/C4")
    assert bool(UsdPhysics.CollisionAPI(prim))  # implicit anchor applied
    assert prim.GetAttribute("physics:collisionEnabled").Get() is True
    assert abs(prim.GetAttribute("physxCollision:contactOffset").Get() - 0.02) < 1e-6
    assert abs(prim.GetAttribute("newton:contactMargin").Get() - 0.01) < 1e-6


# -------------------------------------------------------------------------------------
# expression targeting: gprim gate, pattern narrowing, zero-target warning
# -------------------------------------------------------------------------------------


def test_mesh_collision_fragments_author_on_every_matched_collider():
    """Mesh-collision fragments author on all matched colliders; the expression is trusted."""
    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    xform_collider = UsdGeom.Xform.Define(stage, "/World/Grp/agg").GetPrim()
    mesh_collider = UsdGeom.Cube.Define(stage, "/World/Grp/box").GetPrim()
    UsdPhysics.CollisionAPI.Apply(xform_collider)
    UsdPhysics.CollisionAPI.Apply(mesh_collider)

    result = apply_collision_properties(
        "/World/Grp/**", [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")], stage=stage
    )

    assert result is True
    for prim in (xform_collider, mesh_collider):
        assert prim.HasAPI(UsdPhysics.MeshCollisionAPI), prim.GetPath()
        assert prim.GetAttribute("physics:approximation").HasAuthoredValue(), prim.GetPath()


def test_collision_fragments_pattern_narrows_targets():
    """An expression targets only the colliders it matches."""
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg

    from isaaclab.sim.schemas import apply_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    for path in ("/World/Bot/colL", "/World/Bot/colR"):
        prim = UsdGeom.Cube.Define(stage, path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
    result = apply_collision_properties("/World/Bot/colL", [PhysxCollisionCfg(contact_offset=0.02)], stage=stage)
    assert result is True
    left = stage.GetPrimAtPath("/World/Bot/colL")
    right = stage.GetPrimAtPath("/World/Bot/colR")
    assert abs(left.GetAttribute("physxCollision:contactOffset").Get() - 0.02) < 1e-6
    assert not right.GetAttribute("physxCollision:contactOffset").HasAuthoredValue()


def test_collision_fragments_zero_targets_warn_and_return_false(caplog):
    """No collider matched and no creation requested: warn, author nothing, report failure."""
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg

    from isaaclab.sim.schemas import apply_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    with caplog.at_level("WARNING"):
        result = apply_collision_properties(
            "/World/DoesNotExist", [PhysxCollisionCfg(contact_offset=0.02)], stage=stage
        )
    assert result is False
    assert "/World/DoesNotExist" in caplog.text


# -------------------------------------------------------------------------------------
# spawner slot accepts a fragment mapping + routing by type
# -------------------------------------------------------------------------------------


def test_spawn_shape_with_collision_fragment_list():
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg

    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    cfg = sim_utils.CuboidCfg(
        size=(1, 1, 1),
        collision_props={"": [UsdPhysicsCollisionCfg(collision_enabled=True), PhysxCollisionCfg(contact_offset=0.03)]},
    )
    cfg.func("/World/Cube", cfg)
    prim = sim_utils.get_current_stage().GetPrimAtPath("/World/Cube/geometry/mesh")
    assert bool(UsdPhysics.CollisionAPI(prim))
    assert abs(prim.GetAttribute("physxCollision:contactOffset").Get() - 0.03) < 1e-6


# -------------------------------------------------------------------------------------
# public imports
# -------------------------------------------------------------------------------------


def test_public_imports():
    from isaaclab_newton.sim.schemas import NewtonCollisionCfg  # noqa: F401
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg  # noqa: F401

    from isaaclab.sim.schemas import (  # noqa: F401
        CollisionFragment,
        SchemaFragment,
        UsdPhysicsCollisionCfg,
        apply_collision_properties,
        apply_namespaced,
    )


def test_collision_fragments_create_on_every_matched_prim():
    """Creation applies ``CollisionAPI`` to every matched prim lacking it."""
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg

    from isaaclab.sim.schemas import apply_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    for path in ("/World/Grp", "/World/Grp/a", "/World/Grp/b"):
        UsdGeom.Xform.Define(stage, path)

    result = apply_collision_properties(
        "/World/Grp/**", [PhysxCollisionCfg(contact_offset=0.02)], create_if_missing=True, stage=stage
    )

    assert result is True
    for path in ("/World/Grp", "/World/Grp/a", "/World/Grp/b"):
        prim = stage.GetPrimAtPath(path)
        assert prim.HasAPI(UsdPhysics.CollisionAPI), path
        assert prim.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.02), path
