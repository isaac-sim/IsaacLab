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

from pxr import Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def cleanup_simulation_context():
    """Release the simulation context after each test."""
    yield
    SimulationContext.clear_instance()


def _make_xform(stage, path="/World/Body"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


# -------------------------------------------------------------------------------------
# MujocoCollisionCfg (isaaclab_newton)
# -------------------------------------------------------------------------------------


def test_mujoco_collision_fragment_writes_mjc_namespace():
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, apply_mujoco_collision

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/C_mjc")
    UsdPhysics.CollisionAPI.Apply(prim)
    apply_mujoco_collision(
        MujocoCollisionCfg(
            condim=4,
            group=2,
            priority=3,
            solimp=(0.9, 0.99, 0.001, 0.5, 2.0),
            solmix=0.75,
            solref=(0.02, 1.0),
        ),
        "/World/C_mjc",
        stage,
    )
    assert prim.GetAttribute("mjc:condim").Get() == 4
    assert prim.GetAttribute("mjc:group").Get() == 2
    assert prim.GetAttribute("mjc:priority").Get() == 3
    assert prim.GetAttribute("mjc:solimp").GetTypeName() == Sdf.ValueTypeNames.DoubleArray
    assert tuple(prim.GetAttribute("mjc:solimp").Get()) == pytest.approx((0.9, 0.99, 0.001, 0.5, 2.0))
    assert prim.GetAttribute("mjc:solmix").Get() == pytest.approx(0.75)
    assert prim.GetAttribute("mjc:solref").GetTypeName() == Sdf.ValueTypeNames.DoubleArray
    assert tuple(prim.GetAttribute("mjc:solref").Get()) == pytest.approx((0.02, 1.0))


def test_mujoco_collision_fragment_writes_only_set_fields():
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, apply_mujoco_collision

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/C_mjc_none")
    UsdPhysics.CollisionAPI.Apply(prim)
    apply_mujoco_collision(MujocoCollisionCfg(condim=6), "/World/C_mjc_none", stage)
    assert prim.GetAttribute("mjc:condim").Get() == 6
    for attr_name in ("group", "priority", "solimp", "solmix", "solref"):
        assert not prim.GetAttribute(f"mjc:{attr_name}").IsValid()


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        ({"condim": 2}, "'condim' must be one of"),
        ({"group": 6}, "'group' must be between"),
        ({"priority": -1}, "'priority' must be non-negative"),
        ({"solmix": -0.1}, "'solmix' must be non-negative"),
        ({"solimp": (0.9, 0.95, 0.001, 0.5)}, "'solimp' must contain exactly 5"),
        ({"solref": (0.02,)}, "'solref' must contain exactly 2"),
    ],
)
def test_mujoco_collision_fragment_rejects_invalid_values(cfg, message):
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, apply_mujoco_collision

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_xform(stage, "/World/C_mjc_invalid")
    UsdPhysics.CollisionAPI.Apply(prim)
    with pytest.raises(ValueError, match=message):
        apply_mujoco_collision(MujocoCollisionCfg(**cfg), "/World/C_mjc_invalid", stage)


# -------------------------------------------------------------------------------------
# apply_collision_properties dispatch (explicit anchor creation + multi-namespace)
# -------------------------------------------------------------------------------------


def test_apply_collision_properties_composes_namespaces():
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg
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
            PhysxCollisionCfg(contact_offset=0.02, rest_offset=0.0, torsional_patch_radius=0.1),
            NewtonCollisionCfg(contact_margin=0.01, contact_gap=0.005),
            MujocoCollisionCfg(condim=4),
        ],
        create_if_missing=True,
        stage=stage,
    )
    prim = stage.GetPrimAtPath("/World/C4")
    assert bool(UsdPhysics.CollisionAPI(prim))  # implicit anchor applied
    assert prim.GetAttribute("physics:collisionEnabled").Get() is True
    assert abs(prim.GetAttribute("physxCollision:contactOffset").Get() - 0.02) < 1e-6
    # an explicit 0.0 is a set value, not an unset field
    assert prim.GetAttribute("physxCollision:restOffset").HasAuthoredValue()
    assert abs(prim.GetAttribute("physxCollision:restOffset").Get() - 0.0) < 1e-6
    assert abs(prim.GetAttribute("physxCollision:torsionalPatchRadius").Get() - 0.1) < 1e-6
    assert abs(prim.GetAttribute("newton:contactMargin").Get() - 0.01) < 1e-6
    assert abs(prim.GetAttribute("newton:contactGap").Get() - 0.005) < 1e-6
    assert prim.GetAttribute("mjc:condim").Get() == 4


# -------------------------------------------------------------------------------------
# expression targeting: mesh-collision fragments on matched colliders
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
        "/World/Grp(/.*)?", [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")], stage=stage
    )

    assert result is True
    for prim in (xform_collider, mesh_collider):
        assert prim.HasAPI(UsdPhysics.MeshCollisionAPI), prim.GetPath()
        assert prim.GetAttribute("physics:approximation").HasAuthoredValue(), prim.GetPath()


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
