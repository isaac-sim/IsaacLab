# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def _make_xform(stage, path="/World/Mesh"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


def _has_authored_api_schema(prim, schema_name: str) -> bool:
    """Return whether a schema name is applied or authored in ``apiSchemas`` metadata.

    A schema that is authored via ``AddAppliedSchema`` but not registered in the current build
    appears in the ``apiSchemas`` listOp yet not in the composed ``GetAppliedSchemas()``.
    """
    if schema_name in prim.GetAppliedSchemas():
        return True
    api_schemas = prim.GetMetadata("apiSchemas")
    if api_schemas is None:
        return False
    return any(
        schema_name in getattr(api_schemas, item_list)
        for item_list in ("explicitItems", "prependedItems", "appendedItems", "addedItems")
    )


# -------------------------------------------------------------------------------------
# Fragment metadata + marker hierarchy
# -------------------------------------------------------------------------------------


def test_mesh_collision_fragment_metadata_defaults():
    from isaaclab.sim.schemas import MeshCollisionFragment, SchemaFragment, UsdPhysicsMeshCollisionCfg

    cfg = UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")
    assert isinstance(cfg, MeshCollisionFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == "physics"
    assert type(cfg)._usd_applied_schema is None  # anchor applies MeshCollisionAPI, not the fragment
    assert cfg.func == "isaaclab.sim.schemas:apply_mesh_collision"
    assert cfg.mesh_approximation_name == "convexHull"


# -------------------------------------------------------------------------------------
# Core USD fragment: physics:approximation token via apply_mesh_collision_properties
# -------------------------------------------------------------------------------------


def test_usd_mesh_collision_fragment_writes_approximation_token():
    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/M0")
    apply_mesh_collision_properties(
        "/World/M0", [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="boundingCube")], stage
    )
    prim = stage.GetPrimAtPath("/World/M0")
    assert bool(UsdPhysics.MeshCollisionAPI(prim))
    assert prim.GetAttribute("physics:approximation").Get() == "boundingCube"


# -------------------------------------------------------------------------------------
# PhysX cooking fragments (isaaclab_physx): each writes its own physx*Collision namespace
# -------------------------------------------------------------------------------------


def test_physx_convex_hull_fragment_writes_namespace():
    from isaaclab_physx.sim.schemas import PhysxConvexHullCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/M1")
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    apply_namespaced(PhysxConvexHullCfg(hull_vertex_limit=32, min_thickness=0.002), "/World/M1", stage)
    assert prim.GetAttribute("physxConvexHullCollision:hullVertexLimit").Get() == 32
    assert abs(prim.GetAttribute("physxConvexHullCollision:minThickness").Get() - 0.002) < 1e-6
    # ``mesh_approximation_name`` must NOT be authored as a namespaced attr by the generic applier.
    assert not prim.HasAttribute("physxConvexHullCollision:meshApproximationName")


def test_physx_convex_decomposition_fragment_writes_namespace():
    from isaaclab_physx.sim.schemas import PhysxConvexDecompositionCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/M2")
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    apply_namespaced(PhysxConvexDecompositionCfg(max_convex_hulls=8, shrink_wrap=True), "/World/M2", stage)
    assert prim.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls").Get() == 8
    assert prim.GetAttribute("physxConvexDecompositionCollision:shrinkWrap").Get() is True


def test_physx_triangle_mesh_fragment_writes_namespace():
    from isaaclab_physx.sim.schemas import PhysxTriangleMeshCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/M3")
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    apply_namespaced(PhysxTriangleMeshCfg(weld_tolerance=0.01), "/World/M3", stage)
    assert abs(prim.GetAttribute("physxTriangleMeshCollision:weldTolerance").Get() - 0.01) < 1e-6


def test_physx_triangle_mesh_simplification_fragment_writes_namespace():
    from isaaclab_physx.sim.schemas import PhysxTriangleMeshSimplificationCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/M4")
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    apply_namespaced(PhysxTriangleMeshSimplificationCfg(simplification_metric=0.7), "/World/M4", stage)
    ns = "physxTriangleMeshSimplificationCollision"
    assert abs(prim.GetAttribute(f"{ns}:simplificationMetric").Get() - 0.7) < 1e-6


def test_physx_sdf_mesh_fragment_writes_namespace():
    from isaaclab_physx.sim.schemas import PhysxSDFMeshCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/M5")
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    apply_namespaced(PhysxSDFMeshCfg(sdf_resolution=128, sdf_margin=0.02), "/World/M5", stage)
    assert prim.GetAttribute("physxSDFMeshCollision:sdfResolution").Get() == 128
    assert abs(prim.GetAttribute("physxSDFMeshCollision:sdfMargin").Get() - 0.02) < 1e-6


# -------------------------------------------------------------------------------------
# Newton cooking fragments (isaaclab_newton): newton namespace + applied schema
# -------------------------------------------------------------------------------------


def test_newton_mesh_collision_fragment_writes_namespace():
    from isaaclab_newton.sim.schemas import NewtonMeshCollisionCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/M6")
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    apply_namespaced(NewtonMeshCollisionCfg(max_hull_vertices=24), "/World/M6", stage)
    assert prim.GetAttribute("newton:maxHullVertices").Get() == 24
    assert "NewtonMeshCollisionAPI" in prim.GetAppliedSchemas()


def test_newton_sdf_collision_fragment_writes_namespace():
    from isaaclab_newton.sim.schemas import NewtonSDFCollisionCfg

    from isaaclab.sim.schemas import apply_namespaced

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/M7")
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    apply_namespaced(NewtonSDFCollisionCfg(sdf_max_resolution=64, hydroelastic_enabled=True), "/World/M7", stage)
    assert prim.GetAttribute("newton:sdfMaxResolution").Get() == 64
    assert prim.GetAttribute("newton:hydroelasticEnabled").Get() is True
    # ``NewtonSDFCollisionAPI`` is authored into the ``apiSchemas`` listOp (like the legacy cfg) but
    # is not a registered schema in this Newton build, so it is absent from the composed
    # ``GetAppliedSchemas()``. Assert the authored token, matching the legacy Newton test.
    assert _has_authored_api_schema(prim, "NewtonSDFCollisionAPI")


# -------------------------------------------------------------------------------------
# Composition through apply_mesh_collision_properties: token coupling + multi-namespace
# -------------------------------------------------------------------------------------


def test_apply_mesh_collision_properties_composes_namespaces():
    from isaaclab_newton.sim.schemas import NewtonMeshCollisionCfg
    from isaaclab_physx.sim.schemas import PhysxConvexHullCfg

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/M8")
    apply_mesh_collision_properties(
        "/World/M8",
        [
            UsdPhysicsMeshCollisionCfg(),
            PhysxConvexHullCfg(hull_vertex_limit=48),
            NewtonMeshCollisionCfg(max_hull_vertices=48),
        ],
        stage,
    )
    prim = stage.GetPrimAtPath("/World/M8")
    assert bool(UsdPhysics.MeshCollisionAPI(prim))  # implicit anchor applied
    # token coupling: the convex-hull cooking fragment sets ``physics:approximation``
    assert prim.GetAttribute("physics:approximation").Get() == "convexHull"
    assert prim.GetAttribute("physxConvexHullCollision:hullVertexLimit").Get() == 48
    assert prim.GetAttribute("newton:maxHullVertices").Get() == 48


def test_apply_mesh_collision_properties_rejects_invalid_token():
    import pytest

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/M9")
    with pytest.raises(ValueError):
        apply_mesh_collision_properties(
            "/World/M9", [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="notAToken")], stage
        )


def test_apply_mesh_collision_properties_raises_on_invalid_prim():
    import pytest

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    with pytest.raises(ValueError):
        apply_mesh_collision_properties("/World/DoesNotExist", [UsdPhysicsMeshCollisionCfg()], stage)


def test_apply_mesh_collision_properties_aggregates_fragment_results():
    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/Magg")

    # a fragment whose applier reports failure must make the aggregate return False
    failing = UsdPhysicsMeshCollisionCfg()
    failing.func = lambda cfg, prim_path, stage=None: False
    assert apply_mesh_collision_properties("/World/Magg", [failing], stage) is False

    # all-succeeding fragments return True
    ok = UsdPhysicsMeshCollisionCfg()
    ok.func = lambda cfg, prim_path, stage=None: True
    assert apply_mesh_collision_properties("/World/Magg", [ok], stage) is True


def test_apply_mesh_collision_properties_accepts_generator():
    # the writer dispatches fragments from any iterable; a one-shot generator is consumed once and
    # each fragment authors both its namespace and its implied approximation token
    from isaaclab_physx.sim.schemas import PhysxConvexHullCfg

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/Mgen")
    frags = (f for f in [UsdPhysicsMeshCollisionCfg(), PhysxConvexHullCfg(hull_vertex_limit=48)])
    apply_mesh_collision_properties("/World/Mgen", frags, stage)
    prim = stage.GetPrimAtPath("/World/Mgen")
    # both passes ran: approximation token resolved AND the per-fragment namespaced attr written
    assert prim.GetAttribute("physics:approximation").Get() == "convexHull"
    assert prim.GetAttribute("physxConvexHullCollision:hullVertexLimit").Get() == 48


# -------------------------------------------------------------------------------------
# apply_mesh_collision: the per-fragment func carrying the approximation-token coupling
# -------------------------------------------------------------------------------------


def test_apply_mesh_collision_writes_namespace_and_implied_token():
    # the per-fragment func (the default ``func`` of every MeshCollisionFragment) writes the
    # fragment's namespaced cooking attrs AND the ``physics:approximation`` token it implies
    from isaaclab_physx.sim.schemas import PhysxConvexHullCfg

    from isaaclab.sim.schemas import apply_mesh_collision

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/Mfunc")
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    apply_mesh_collision(PhysxConvexHullCfg(hull_vertex_limit=16), "/World/Mfunc", stage)
    assert prim.GetAttribute("physxConvexHullCollision:hullVertexLimit").Get() == 16
    assert prim.GetAttribute("physics:approximation").Get() == "convexHull"


def test_apply_mesh_collision_rejects_invalid_token():
    import pytest

    from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    _make_xform(stage, "/World/Mfunc2")
    with pytest.raises(ValueError):
        apply_mesh_collision(UsdPhysicsMeshCollisionCfg(mesh_approximation_name="notAToken"), "/World/Mfunc2", stage)


# -------------------------------------------------------------------------------------
# Public imports
# -------------------------------------------------------------------------------------


def test_public_imports():
    from isaaclab_newton.sim.schemas import NewtonMeshCollisionCfg, NewtonSDFCollisionCfg  # noqa: F401
    from isaaclab_physx.sim.schemas import (  # noqa: F401
        PhysxConvexDecompositionCfg,
        PhysxConvexHullCfg,
        PhysxSDFMeshCfg,
        PhysxTriangleMeshCfg,
        PhysxTriangleMeshSimplificationCfg,
    )

    from isaaclab.sim.schemas import (  # noqa: F401
        MeshCollisionFragment,
        SchemaFragment,
        UsdPhysicsMeshCollisionCfg,
        apply_mesh_collision,
        apply_mesh_collision_properties,
        apply_namespaced,
    )
