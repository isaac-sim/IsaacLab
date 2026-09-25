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
from isaaclab_newton.sim.schemas import NewtonMeshCollisionCfg, NewtonSDFCollisionCfg
from isaaclab_physx.sim.schemas import (
    PhysxConvexDecompositionCfg,
    PhysxConvexHullCfg,
    PhysxSDFMeshCfg,
    PhysxTriangleMeshCfg,
    PhysxTriangleMeshSimplificationCfg,
)

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.schemas import UsdPhysicsMeshCollisionCfg, apply_mesh_collision

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def cleanup_simulation_context():
    """Release the simulation context after each test."""
    yield
    SimulationContext.clear_instance()


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
# apply_mesh_collision: each fragment writes its own namespace and the token it implies
# -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fragment", "attrs", "token", "authored_schema"),
    [
        (
            PhysxConvexHullCfg(hull_vertex_limit=32, min_thickness=0.002),
            {"physxConvexHullCollision:hullVertexLimit": 32, "physxConvexHullCollision:minThickness": 0.002},
            "convexHull",
            None,
        ),
        (
            PhysxConvexDecompositionCfg(max_convex_hulls=8, shrink_wrap=True),
            {
                "physxConvexDecompositionCollision:maxConvexHulls": 8,
                "physxConvexDecompositionCollision:shrinkWrap": True,
            },
            "convexDecomposition",
            None,
        ),
        (
            PhysxTriangleMeshCfg(weld_tolerance=0.01),
            {"physxTriangleMeshCollision:weldTolerance": 0.01},
            None,
            None,
        ),
        (
            PhysxTriangleMeshSimplificationCfg(simplification_metric=0.7),
            {"physxTriangleMeshSimplificationCollision:simplificationMetric": 0.7},
            "meshSimplification",
            None,
        ),
        (
            PhysxSDFMeshCfg(sdf_resolution=128, sdf_margin=0.02),
            {"physxSDFMeshCollision:sdfResolution": 128, "physxSDFMeshCollision:sdfMargin": 0.02},
            "sdf",
            None,
        ),
        (
            UsdPhysicsMeshCollisionCfg(mesh_approximation_name="boundingSphere"),
            {},
            "boundingSphere",
            None,
        ),
        # Newton cooking fragments author no token; their API schemas are authored into the
        # ``apiSchemas`` listOp but are not registered in this Newton build
        (
            NewtonMeshCollisionCfg(max_hull_vertices=24),
            {"newton:maxHullVertices": 24},
            None,
            "NewtonMeshCollisionAPI",
        ),
        (
            NewtonSDFCollisionCfg(sdf_max_resolution=64, hydroelastic_enabled=True),
            {"newton:sdfMaxResolution": 64, "newton:hydroelasticEnabled": True},
            None,
            "NewtonSDFCollisionAPI",
        ),
    ],
    ids=lambda value: type(value).__name__ if hasattr(value, "func") else None,
)
def test_apply_mesh_collision_writes_namespace_and_implied_token(fragment, attrs, token, authored_schema):
    """The per-fragment func (the default ``func`` of every MeshCollisionFragment) writes the
    fragment's namespaced cooking attrs AND the ``physics:approximation`` token it implies."""
    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/Mfunc")
    UsdPhysics.MeshCollisionAPI.Apply(prim)
    apply_mesh_collision(fragment, "/World/Mfunc", stage)

    for name, value in attrs.items():
        assert prim.GetAttribute(name).Get() == pytest.approx(value), name
    if token is None:
        assert not prim.GetAttribute("physics:approximation").HasAuthoredValue()
    else:
        assert prim.GetAttribute("physics:approximation").Get() == token
    # ``mesh_approximation_name`` selects the token and is never authored as a namespaced attr
    assert not any(attr.GetName().endswith(":meshApproximationName") for attr in prim.GetAttributes())
    if authored_schema is not None:
        assert _has_authored_api_schema(prim, authored_schema)


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
