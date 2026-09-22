# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils

pytestmark = [pytest.mark.unit, pytest.mark.isaacsim_ci]


@pytest.fixture
def stage():
    return sim_utils.create_new_stage()


@pytest.mark.parametrize(
    ("cfg", "prim_type", "attributes"),
    [
        (sim_utils.ConeCfg(radius=1.0, height=2.0, axis="Y"), "Cone", {"radius": 1.0, "height": 2.0, "axis": "Y"}),
        (
            sim_utils.CapsuleCfg(radius=1.0, height=2.0, axis="Y"),
            "Capsule",
            {"radius": 1.0, "height": 2.0, "axis": "Y"},
        ),
        (
            sim_utils.CylinderCfg(radius=1.0, height=2.0, axis="Y"),
            "Cylinder",
            {"radius": 1.0, "height": 2.0, "axis": "Y"},
        ),
        (sim_utils.CuboidCfg(size=(1.0, 2.0, 3.0)), "Cube", {"size": 1.0, "xformOp:scale": (1.0, 2.0, 3.0)}),
        (sim_utils.SphereCfg(radius=1.0), "Sphere", {"radius": 1.0}),
    ],
    ids=["cone", "capsule", "cylinder", "cuboid", "sphere"],
)
def test_spawn_shape(stage, cfg, prim_type, attributes):
    prim = cfg.func("/World/Shape", cfg)
    assert prim.GetPath() == "/World/Shape"
    assert prim.GetTypeName() == "Xform"
    # the geometry lives on a nested prim so the container can carry the physics schemas
    mesh = stage.GetPrimAtPath("/World/Shape/geometry/mesh")
    assert mesh.GetTypeName() == prim_type
    for name, value in attributes.items():
        assert mesh.GetAttribute(name).Get() == value
    with pytest.raises(ValueError, match="already exists"):
        cfg.func("/World/Shape", cfg)


def test_spawn_shape_with_physics_props(stage):
    usd_rigid_props = sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)
    physx_rigid_props = PhysxRigidBodyCfg(solver_position_iteration_count=8, sleep_threshold=0.1)
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        rigid_props=[usd_rigid_props, physx_rigid_props],
        mass_props=sim_utils.MassCfg(mass=5.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
    )
    prim = cfg.func("/World/Cone", cfg)

    # rigid body and mass schemas on the container, collision on the geometry
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
    assert prim.GetAttribute("physxRigidBody:solverPositionIterationCount").Get() == 8
    assert prim.GetAttribute("physxRigidBody:sleepThreshold").Get() == pytest.approx(0.1)
    assert prim.GetAttribute("physics:mass").Get() == 5.0
    assert stage.GetPrimAtPath("/World/Cone/geometry/mesh").GetAttribute("physics:collisionEnabled").Get() is True
    assert stage.GetPrimAtPath("/World/Cone/geometry/material").IsValid()

    # density instead of mass
    cfg = sim_utils.ConeCfg(radius=1.0, height=2.0, mass_props=sim_utils.MassCfg(density=10.0))
    assert cfg.func("/World/DenseCone", cfg).GetAttribute("physics:density").Get() == 10.0


def test_spawn_shape_clones(stage):
    for i in range(3):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material_path="/Looks/visualMaterial",
        physics_material_path="/Looks/physicsMaterial",
    )

    prim = cfg.func("/World/env_.*/Cone", cfg)

    # the source prim is returned and every matching parent receives a copy
    assert prim.GetPath() == "/World/env_0/Cone"
    assert len(sim_utils.find_matching_prim_paths("/World/env_[^/]+/Cone")) == 3
    # the global materials are shared prims at exactly the configured paths
    assert sim_utils.find_matching_prim_paths("/Looks/visualMaterial") == ["/Looks/visualMaterial"]
    assert sim_utils.find_matching_prim_paths("/Looks/physicsMaterial") == ["/Looks/physicsMaterial"]

    with pytest.raises(RuntimeError, match="Unable to find source prim path"):
        cfg.func("/World/missing/env_.*/Cone", cfg)
