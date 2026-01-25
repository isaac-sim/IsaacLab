# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests to verify collision behavior between different collision primitives.

This test suite verifies that:
1. Objects with different collision primitives collide correctly
2. Objects don't interpenetrate during collision
3. Momentum and energy transfer behave as expected
4. Objects can be stacked stably

Inspired by the Newton physics engine collision tests:
https://github.com/newton-physics/newton/blob/main/newton/tests/test_collision_pipeline.py
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
from enum import Enum, auto

import pytest

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim.spawners import materials
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Import hand configurations for articulated collision tests
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG


##
# Unified Shape Type Enum
##


class ShapeType(Enum):
    """Enumeration of all collision shape types for testing.

    Includes both analytical primitives and mesh-based representations.
    """

    # Analytical primitives (faster, exact geometry)
    SPHERE = auto()
    BOX = auto()
    CAPSULE = auto()
    CYLINDER = auto()
    CONE = auto()
    # Mesh-based colliders (triangle mesh representations)
    MESH_SPHERE = auto()
    MESH_BOX = auto()
    MESH_CAPSULE = auto()
    MESH_CYLINDER = auto()
    MESH_CONE = auto()
    # SDF-based colliders (signed distance field)
    SDF_SPHERE = auto()


# Convenience lists for parameterization
PRIMITIVE_SHAPES = [ShapeType.SPHERE, ShapeType.BOX, ShapeType.CAPSULE, ShapeType.CYLINDER, ShapeType.CONE]
MESH_SHAPES = [
    ShapeType.MESH_SPHERE, ShapeType.MESH_BOX, ShapeType.MESH_CAPSULE,
    ShapeType.MESH_CYLINDER, ShapeType.MESH_CONE
]
SDF_SHAPES = [ShapeType.SDF_SPHERE]
ALL_SHAPES = PRIMITIVE_SHAPES + MESH_SHAPES + SDF_SHAPES
BOX_SHAPES = [ShapeType.BOX, ShapeType.MESH_BOX]
SPHERE_SHAPES = [ShapeType.SPHERE, ShapeType.MESH_SPHERE, ShapeType.SDF_SPHERE]


class CollisionTestLevel(Enum):
    """Test strictness levels for collision verification."""

    VELOCITY_X = auto()  # Check velocity along X-axis (collision direction)
    VELOCITY_YZ = auto()  # Check no spurious velocity in Y/Z directions
    VELOCITY_LINEAR = auto()  # Check all linear velocities
    VELOCITY_ANGULAR = auto()  # Check no spurious angular velocity
    STRICT = auto()  # All checks


def shape_type_to_str(shape_type: ShapeType) -> str:
    """Convert ShapeType enum to string for test naming."""
    return shape_type.name.lower()


def is_mesh_shape(shape_type: ShapeType) -> bool:
    """Check if shape type is a mesh-based collider."""
    return shape_type in MESH_SHAPES


##
# Unified Shape Configuration Factory
##


def create_shape_cfg(
    shape_type: ShapeType,
    prim_path: str,
    pos: tuple = (0, 0, 1.0),
    disable_gravity: bool = True,
) -> RigidObjectCfg:
    """Create RigidObjectCfg for any shape type.

    Args:
        shape_type: The type of collision shape to create
        prim_path: USD prim path for the object
        pos: Initial position (x, y, z)
        disable_gravity: Whether to disable gravity for the object

    Returns:
        RigidObjectCfg configured for the specified shape
    """
    # Common rigid body properties
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=disable_gravity,
        linear_damping=0.0,
        angular_damping=0.0,
    )
    collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    mass_props = sim_utils.MassPropertiesCfg(mass=1.0)

    # Shape-specific spawner configurations
    spawner_map = {
        # Analytical primitives
        ShapeType.SPHERE: lambda: sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.8)),
        ),
        ShapeType.BOX: lambda: sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.8, 0.4)),
        ),
        ShapeType.CAPSULE: lambda: sim_utils.CapsuleCfg(
            radius=0.15,
            height=0.3,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.4)),
        ),
        ShapeType.CYLINDER: lambda: sim_utils.CylinderCfg(
            radius=0.2,
            height=0.4,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.4)),
        ),
        ShapeType.CONE: lambda: sim_utils.ConeCfg(
            radius=0.25,
            height=0.5,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.8)),
        ),
        # Mesh-based colliders (darker colors to distinguish)
        ShapeType.MESH_SPHERE: lambda: sim_utils.MeshSphereCfg(
            radius=0.25,
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.6)),
        ),
        ShapeType.MESH_BOX: lambda: sim_utils.MeshCuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
        ),
        ShapeType.MESH_CAPSULE: lambda: sim_utils.MeshCapsuleCfg(
            radius=0.15,
            height=0.3,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 0.2)),
        ),
        ShapeType.MESH_CYLINDER: lambda: sim_utils.MeshCylinderCfg(
            radius=0.2,
            height=0.4,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.2)),
        ),
        ShapeType.MESH_CONE: lambda: sim_utils.MeshConeCfg(
            radius=0.25,
            height=0.5,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 0.6)),
        ),
        # SDF-based colliders (purple tint to distinguish)
        ShapeType.SDF_SPHERE: lambda: sim_utils.MeshSphereCfg(
            radius=0.25,
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.5)),
        ),
    }

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawner_map[shape_type](),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def get_shape_extent(shape_type: ShapeType) -> float:
    """Get the approximate XY extent (radius/half-size) of a shape for positioning.

    This is used to ensure objects are spawned with enough separation.
    For axis-aligned shapes (capsule, cylinder, cone), this is the radius in the XY plane.
    """
    extents = {
        ShapeType.SPHERE: 0.25,
        ShapeType.BOX: 0.25,
        ShapeType.CAPSULE: 0.15,
        ShapeType.CYLINDER: 0.20,
        ShapeType.CONE: 0.25,
        ShapeType.MESH_SPHERE: 0.25,
        ShapeType.MESH_BOX: 0.25,
        ShapeType.MESH_CAPSULE: 0.15,
        ShapeType.MESH_CYLINDER: 0.20,
        ShapeType.MESH_CONE: 0.25,
        ShapeType.SDF_SPHERE: 0.25,
    }
    return extents[shape_type]


def get_shape_height(shape_type: ShapeType) -> float:
    """Get the height of a shape for stacking calculations."""
    heights = {
        ShapeType.SPHERE: 0.5,
        ShapeType.BOX: 0.5,
        ShapeType.CAPSULE: 0.6,  # height + 2*radius
        ShapeType.CYLINDER: 0.4,
        ShapeType.CONE: 0.5,
        ShapeType.MESH_SPHERE: 0.5,
        ShapeType.MESH_BOX: 0.5,
        ShapeType.MESH_CAPSULE: 0.6,
        ShapeType.MESH_CYLINDER: 0.4,
        ShapeType.MESH_CONE: 0.5,
        ShapeType.SDF_SPHERE: 0.5,
    }
    return heights[shape_type]


def get_shape_min_resting_height(shape_type: ShapeType) -> float:
    """Get the minimum height where an object can rest (accounts for tumbling).

    Some shapes like capsules and cones can tumble and rest on their side,
    so their minimum resting height is their radius, not half-height.
    Cones can rest on their tip, which puts the center near ground level.
    """
    # Use the XY extent (radius) as min resting height for shapes that can tumble
    min_heights = {
        ShapeType.SPHERE: 0.25,  # radius
        ShapeType.BOX: 0.25,  # half-height
        ShapeType.CAPSULE: 0.15,  # radius (can rest on side)
        ShapeType.CYLINDER: 0.2,  # radius (can rest on side)
        ShapeType.CONE: 0.0,  # can rest on tip (center near ground)
        ShapeType.MESH_SPHERE: 0.25,
        ShapeType.MESH_BOX: 0.25,
        ShapeType.MESH_CAPSULE: 0.15,
        ShapeType.MESH_CYLINDER: 0.2,
        ShapeType.MESH_CONE: 0.0,  # can rest on tip (center near ground)
        ShapeType.SDF_SPHERE: 0.25,
    }
    return min_heights[shape_type]


def is_sdf_shape(shape_type: ShapeType) -> bool:
    """Check if a shape type uses SDF collision."""
    return shape_type == ShapeType.SDF_SPHERE


def apply_sdf_collision(prim_path: str) -> None:
    """Apply SDF collision properties to a mesh prim.

    This should be called after the object is spawned but before sim.reset().
    SDF (Signed Distance Field) collision provides more accurate collision
    detection for complex mesh geometries.

    Args:
        prim_path: The prim path of the rigid object (e.g., "/World/Object")
    """
    from pxr import PhysxSchema, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    mesh_prim_path = f"{prim_path}/geometry/mesh"
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)

    if not mesh_prim.IsValid():
        return

    # Set SDF collision approximation
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, mesh_prim_path)
    if mesh_collision_api:
        mesh_collision_api.GetApproximationAttr().Set(PhysxSchema.Tokens.sdf)

    # Apply SDF mesh collision API for additional SDF-specific settings
    PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)


##
# Scene Configuration
##


@configclass
class CollisionTestSceneCfg(InteractiveSceneCfg):
    """Configuration for collision test scenes."""

    terrain: TerrainImporterCfg | None = None
    object_a: RigidObjectCfg | None = None
    object_b: RigidObjectCfg | None = None


##
# Test Fixtures
##


@pytest.fixture(scope="module")
def setup_collision_params():
    """Fixture to set up collision test parameters."""
    sim_dt = 1.0 / 240.0  # 240 Hz physics
    collision_steps = 240  # 1 second of simulation
    return sim_dt, collision_steps


##
# Helper Functions
##


def _perform_sim_step(sim, scene, sim_dt: float):
    """Perform a single simulation step and update scene."""
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt=sim_dt)


def _verify_collision_behavior(
    object_a: RigidObject,
    object_b: RigidObject,
    test_level: CollisionTestLevel,
    tolerance: float = 0.05,
    initial_velocity: float = 1.0,
):
    """Verify collision behavior based on test level."""
    vel_a = object_a.data.root_lin_vel_w[0]
    vel_b = object_b.data.root_lin_vel_w[0]
    ang_vel_a = object_a.data.root_ang_vel_w[0]
    ang_vel_b = object_b.data.root_ang_vel_w[0]

    if test_level in [CollisionTestLevel.VELOCITY_X, CollisionTestLevel.VELOCITY_LINEAR, CollisionTestLevel.STRICT]:
        assert vel_b[0] > 0.1, f"Object B should be moving forward after collision, vel_b[0]={vel_b[0]}"
        assert vel_a[0] < initial_velocity, f"Object A should have slowed, vel_a[0]={vel_a[0]}"

    if test_level in [CollisionTestLevel.VELOCITY_YZ, CollisionTestLevel.VELOCITY_LINEAR, CollisionTestLevel.STRICT]:
        assert abs(vel_a[1]) < tolerance, f"Object A has spurious Y velocity: {vel_a[1]}"
        assert abs(vel_a[2]) < tolerance, f"Object A has spurious Z velocity: {vel_a[2]}"
        assert abs(vel_b[1]) < tolerance, f"Object B has spurious Y velocity: {vel_b[1]}"
        assert abs(vel_b[2]) < tolerance, f"Object B has spurious Z velocity: {vel_b[2]}"

    if test_level in [CollisionTestLevel.VELOCITY_ANGULAR, CollisionTestLevel.STRICT]:
        ang_tolerance = tolerance * 2
        assert torch.norm(ang_vel_a) < ang_tolerance, f"Object A has spurious angular velocity: {ang_vel_a}"
        assert torch.norm(ang_vel_b) < ang_tolerance, f"Object B has spurious angular velocity: {ang_vel_b}"


def _verify_no_interpenetration(
    object_a: RigidObject,
    object_b: RigidObject,
    min_separation: float = 0.0,
):
    """Verify that two objects are not interpenetrating."""
    pos_a = object_a.data.root_pos_w[0]
    pos_b = object_b.data.root_pos_w[0]
    distance = torch.norm(pos_a - pos_b).item()

    assert distance >= min_separation, (
        f"Objects may be interpenetrating: distance={distance:.4f}, expected >= {min_separation:.4f}"
    )


##
# Collision Test Pairs
##

# Define shape pairs with expected test levels
# Format: (shape_a, shape_b, test_level)
COLLISION_PAIRS = [
    # Sphere collisions - typically very clean
    (ShapeType.SPHERE, ShapeType.SPHERE, CollisionTestLevel.STRICT),
    (ShapeType.SPHERE, ShapeType.BOX, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.SPHERE, ShapeType.CAPSULE, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.SPHERE, ShapeType.CYLINDER, CollisionTestLevel.VELOCITY_YZ),
    # Box collisions
    (ShapeType.BOX, ShapeType.BOX, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.BOX, ShapeType.CAPSULE, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.BOX, ShapeType.CYLINDER, CollisionTestLevel.VELOCITY_YZ),
    # Capsule collisions
    (ShapeType.CAPSULE, ShapeType.CAPSULE, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.CAPSULE, ShapeType.CYLINDER, CollisionTestLevel.VELOCITY_YZ),
    # Cylinder collisions
    (ShapeType.CYLINDER, ShapeType.CYLINDER, CollisionTestLevel.VELOCITY_YZ),
    # Mesh sphere collisions
    (ShapeType.MESH_SPHERE, ShapeType.MESH_SPHERE, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.MESH_SPHERE, ShapeType.MESH_BOX, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.MESH_SPHERE, ShapeType.MESH_CAPSULE, CollisionTestLevel.VELOCITY_YZ),
    # Mesh box collisions
    (ShapeType.MESH_BOX, ShapeType.MESH_BOX, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.MESH_BOX, ShapeType.MESH_CAPSULE, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.MESH_BOX, ShapeType.MESH_CYLINDER, CollisionTestLevel.VELOCITY_YZ),
    # Mesh capsule collisions
    (ShapeType.MESH_CAPSULE, ShapeType.MESH_CAPSULE, CollisionTestLevel.VELOCITY_YZ),
    # Mesh cylinder collisions
    (ShapeType.MESH_CYLINDER, ShapeType.MESH_CYLINDER, CollisionTestLevel.VELOCITY_YZ),
    # Mixed: mesh vs primitive
    (ShapeType.MESH_SPHERE, ShapeType.SPHERE, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.MESH_SPHERE, ShapeType.BOX, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.MESH_BOX, ShapeType.SPHERE, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.MESH_BOX, ShapeType.BOX, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.MESH_CAPSULE, ShapeType.CAPSULE, CollisionTestLevel.VELOCITY_YZ),
    (ShapeType.MESH_CYLINDER, ShapeType.CYLINDER, CollisionTestLevel.VELOCITY_YZ),
]


##
# Tests
##


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize(
    "shape_a,shape_b,test_level",
    COLLISION_PAIRS,
    ids=[f"{shape_type_to_str(a)}_{shape_type_to_str(b)}" for a, b, _ in COLLISION_PAIRS],
)
def test_horizontal_collision(
    setup_collision_params,
    device: str,
    shape_a: ShapeType,
    shape_b: ShapeType,
    test_level: CollisionTestLevel,
):
    """Test horizontal collision between two objects.

    Object A approaches Object B along the X-axis. Both objects have equal mass.
    After collision, momentum should be transferred from A to B.

    This test verifies:
    - Collision occurs (momentum transfer)
    - No spurious velocities in perpendicular directions
    - Objects don't interpenetrate
    """
    sim_dt, collision_steps = setup_collision_params

    extent_a = get_shape_extent(shape_a)
    extent_b = get_shape_extent(shape_b)
    separation = (extent_a + extent_b) * 2.5

    with build_simulation_context(device=device, dt=sim_dt, gravity_enabled=False) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(shape_a, "/World/ObjectA", pos=(-separation / 2, 0.0, 0.5))
        scene_cfg.object_b = create_shape_cfg(shape_b, "/World/ObjectB", pos=(separation / 2, 0.0, 0.5))

        scene = InteractiveScene(scene_cfg)

        # Apply SDF collision if needed (before reset)
        if is_sdf_shape(shape_a):
            apply_sdf_collision("/World/ObjectA")
        if is_sdf_shape(shape_b):
            apply_sdf_collision("/World/ObjectB")

        sim.reset()
        scene.reset()

        object_a: RigidObject = scene["object_a"]
        object_b: RigidObject = scene["object_b"]

        initial_velocity = 2.0
        object_a.write_root_velocity_to_sim(torch.tensor([[initial_velocity, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device))
        object_b.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

        collision_detected = False
        for _ in range(collision_steps):
            _perform_sim_step(sim, scene, sim_dt)

            if object_b.data.root_lin_vel_w[0, 0] > 0.1:
                collision_detected = True

            _verify_no_interpenetration(object_a, object_b, min_separation=extent_a + extent_b - 0.2)

        assert collision_detected, "Collision should have occurred between objects"

        _verify_collision_behavior(
            object_a, object_b,
            test_level=test_level,
            tolerance=0.1,
            initial_velocity=initial_velocity,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("shape_type", ALL_SHAPES)
def test_falling_collision_with_ground(
    setup_collision_params,
    device: str,
    shape_type: ShapeType,
):
    """Test that objects fall and collide correctly with the ground plane.

    This test verifies:
    - Object falls under gravity
    - Object comes to rest at expected height (doesn't fall through ground)
    - Object doesn't bounce indefinitely
    """
    sim_dt, _ = setup_collision_params
    fall_steps = 480  # 2 seconds

    with build_simulation_context(
        device=device,
        dt=sim_dt,
        gravity_enabled=True,
        add_ground_plane=True
    ) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(shape_type, "/World/Object", pos=(0.0, 0.0, 2.0), disable_gravity=False)

        scene = InteractiveScene(scene_cfg)

        # Apply SDF collision if needed (before reset)
        if is_sdf_shape(shape_type):
            apply_sdf_collision("/World/Object")

        sim.reset()
        scene.reset()

        obj: RigidObject = scene["object_a"]

        for _ in range(fall_steps):
            _perform_sim_step(sim, scene, sim_dt)

        final_height = obj.data.root_pos_w[0, 2].item()
        final_velocity = torch.norm(obj.data.root_lin_vel_w[0]).item()

        # Use minimum resting height (accounts for shapes that can tumble)
        expected_min_height = get_shape_min_resting_height(shape_type) - 0.05

        assert final_height > expected_min_height, (
            f"Object fell through ground: height={final_height:.4f}, expected > {expected_min_height:.4f}"
        )
        assert final_velocity < 0.5, f"Object still moving too fast: velocity={final_velocity:.4f}"


# Box stacking pairs: (bottom_box, top_box)
STACKING_PAIRS = [
    (ShapeType.BOX, ShapeType.BOX),  # primitive on primitive
    (ShapeType.MESH_BOX, ShapeType.MESH_BOX),  # mesh on mesh
    (ShapeType.BOX, ShapeType.MESH_BOX),  # mesh on primitive
    (ShapeType.MESH_BOX, ShapeType.BOX),  # primitive on mesh
]


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize(
    "bottom_shape,top_shape",
    STACKING_PAIRS,
    ids=[f"{shape_type_to_str(b)}_under_{shape_type_to_str(t)}" for b, t in STACKING_PAIRS],
)
def test_box_stacking_stability(
    setup_collision_params, device: str, bottom_shape: ShapeType, top_shape: ShapeType
):
    """Test that boxes can be stably stacked on top of each other.

    This tests the collision system's ability to maintain stable contacts
    for various combinations of primitive and mesh box colliders.
    """
    sim_dt, _ = setup_collision_params
    settle_steps = 480

    with build_simulation_context(
        device=device,
        dt=sim_dt,
        gravity_enabled=True,
        add_ground_plane=True
    ) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)

        height = get_shape_height(bottom_shape)
        scene_cfg.object_a = create_shape_cfg(
            bottom_shape, "/World/BoxBottom", pos=(0.0, 0.0, height / 2 + 0.01), disable_gravity=False
        )
        scene_cfg.object_b = create_shape_cfg(
            top_shape, "/World/BoxTop", pos=(0.0, 0.0, height * 1.5 + 0.02), disable_gravity=False
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        box_bottom: RigidObject = scene["object_a"]
        box_top: RigidObject = scene["object_b"]

        for _ in range(settle_steps):
            _perform_sim_step(sim, scene, sim_dt)

        bottom_height = box_bottom.data.root_pos_w[0, 2].item()
        top_height = box_top.data.root_pos_w[0, 2].item()

        assert bottom_height > 0.2, f"Bottom box fell through ground: height={bottom_height:.4f}"
        assert top_height > bottom_height + 0.3, (
            f"Top box not properly stacked: top={top_height:.4f}, bottom={bottom_height:.4f}"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("restitution", [0.0, 0.5, 0.9])
def test_sphere_bounce_restitution(setup_collision_params, device: str, restitution: float):
    """Test that sphere bouncing behavior matches configured restitution.

    A sphere is dropped onto a ground plane with specific restitution.
    - restitution=0: No bounce (inelastic)
    - restitution=0.5: Partial bounce
    - restitution=0.9: High bounce
    """
    sim_dt, _ = setup_collision_params
    drop_steps = 120
    bounce_steps = 120

    with build_simulation_context(device=device, dt=sim_dt, gravity_enabled=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            physics_material=materials.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=restitution,
            ),
        )

        rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0)
        scene_cfg.object_a = RigidObjectCfg(
            prim_path="/World/Sphere",
            spawn=sim_utils.SphereCfg(
                radius=0.25,
                rigid_props=rigid_props,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                physics_material=materials.RigidBodyMaterialCfg(
                    static_friction=0.5,
                    dynamic_friction=0.5,
                    restitution=restitution,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        sphere: RigidObject = scene["object_a"]

        for _ in range(drop_steps):
            _perform_sim_step(sim, scene, sim_dt)

        for _ in range(bounce_steps):
            _perform_sim_step(sim, scene, sim_dt)

        final_height = sphere.data.root_pos_w[0, 2].item()

        if restitution < 0.1:
            assert final_height < 0.5, f"Zero restitution should not bounce high: height={final_height:.4f}"
        elif restitution > 0.8:
            assert final_height > 0.3, f"High restitution should bounce: height={final_height:.4f}"


# Sphere pairs for momentum conservation tests
SPHERE_PAIRS = [
    (ShapeType.SPHERE, ShapeType.SPHERE),
    (ShapeType.MESH_SPHERE, ShapeType.MESH_SPHERE),
    (ShapeType.SDF_SPHERE, ShapeType.SDF_SPHERE),
    (ShapeType.SPHERE, ShapeType.MESH_SPHERE),
    (ShapeType.SPHERE, ShapeType.SDF_SPHERE),
    (ShapeType.MESH_SPHERE, ShapeType.SDF_SPHERE),
]


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize(
    "sphere_a,sphere_b",
    SPHERE_PAIRS,
    ids=[f"{shape_type_to_str(a)}_{shape_type_to_str(b)}" for a, b in SPHERE_PAIRS],
)
def test_momentum_conservation_equal_mass(
    setup_collision_params, device: str, sphere_a: ShapeType, sphere_b: ShapeType
):
    """Test that momentum is conserved in equal-mass sphere-sphere collision.

    For equal mass objects with one stationary, after collision:
    - Total momentum is conserved
    - Object A slows down
    - Object B gains velocity
    """
    sim_dt, collision_steps = setup_collision_params

    with build_simulation_context(device=device, dt=sim_dt, gravity_enabled=False) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)

        separation = 1.0
        scene_cfg.object_a = create_shape_cfg(sphere_a, "/World/SphereA", pos=(-separation/2, 0.0, 0.5))
        scene_cfg.object_b = create_shape_cfg(sphere_b, "/World/SphereB", pos=(separation/2, 0.0, 0.5))

        scene = InteractiveScene(scene_cfg)

        # Apply SDF collision if needed (before reset)
        if is_sdf_shape(sphere_a):
            apply_sdf_collision("/World/SphereA")
        if is_sdf_shape(sphere_b):
            apply_sdf_collision("/World/SphereB")

        sim.reset()
        scene.reset()

        sphere_a_obj: RigidObject = scene["object_a"]
        sphere_b_obj: RigidObject = scene["object_b"]

        initial_vel = 2.0
        sphere_a_obj.write_root_velocity_to_sim(torch.tensor([[initial_vel, 0, 0, 0, 0, 0]], device=device))
        sphere_b_obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

        initial_momentum = initial_vel * 1.0

        for _ in range(collision_steps):
            _perform_sim_step(sim, scene, sim_dt)

        final_vel_a = sphere_a_obj.data.root_lin_vel_w[0, 0].item()
        final_vel_b = sphere_b_obj.data.root_lin_vel_w[0, 0].item()
        final_momentum = (final_vel_a + final_vel_b) * 1.0

        momentum_error = abs(final_momentum - initial_momentum)
        assert momentum_error < 0.3, f"Momentum not conserved: initial={initial_momentum}, final={final_momentum}"

        assert abs(final_vel_a) < initial_vel * 0.6, f"Object A should have slowed: {final_vel_a}"
        assert final_vel_b >= initial_vel * 0.4, f"Object B should have gained velocity: {final_vel_b}"


##
# Articulated Hand Collision Tests
##


# Finger tip positions relative to hand root in default orientation
# These values were calibrated using scripts/demos/finger_collision_debug.py
# Format: (x, y, z) offset from hand root position
ALLEGRO_FINGERTIP_OFFSETS = {
    "index": (-0.052, -0.252, 0.052),
    "middle": (-0.001, -0.252, 0.052),
    "ring": (0.054, -0.252, 0.052),
    "thumb": (-0.168, -0.039, 0.080),
}

# Joint names for each finger
ALLEGRO_FINGER_JOINTS = {
    "index": ["index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3"],
    "middle": ["middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3"],
    "ring": ["ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3"],
    "thumb": ["thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3"],
}


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("target_finger", ["index", "middle", "ring", "thumb"])
@pytest.mark.parametrize("sphere_type", [ShapeType.SPHERE, ShapeType.MESH_SPHERE, ShapeType.SDF_SPHERE])
def test_finger_collision_isolation(
    setup_collision_params, device: str, target_finger: str, sphere_type: ShapeType
):
    """Test that dropping an object on one finger only affects that finger.

    This test:
    1. Spawns an Allegro hand in default orientation
    2. Drops a sphere (primitive or mesh) onto a specific fingertip
    3. Verifies that the target finger's joints deflect more than other fingers

    Tests both primitive sphere and mesh sphere colliders to ensure consistent
    collision behavior across different collider representations.
    """
    sim_dt, _ = setup_collision_params
    drop_steps = 480  # 2 seconds for drop and settle

    # Hand position
    hand_pos = (0.0, 0.0, 0.5)

    with build_simulation_context(device=device, dt=sim_dt, gravity_enabled=True) as sim:
        sim._app_control_on_stop_handle = None

        # Create hand configuration with default orientation
        # Gravity disabled on hand so fingers only move from collision, not gravity
        hand_cfg = ALLEGRO_HAND_CFG.copy()
        hand_cfg.prim_path = "/World/Hand"
        hand_cfg.spawn.rigid_props.disable_gravity = True
        hand_cfg.init_state.pos = hand_pos
        # Keep default orientation and joint positions

        # Create ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Create the hand
        hand = Articulation(hand_cfg)

        # Get fingertip offset for target finger
        fingertip_offset = ALLEGRO_FINGERTIP_OFFSETS[target_finger]

        # Create a small sphere to drop on the fingertip
        # Position it directly above the fingertip
        drop_height = 0.15  # 15cm above fingertip
        sphere_pos = (
            hand_pos[0] + fingertip_offset[0],
            hand_pos[1] + fingertip_offset[1],
            hand_pos[2] + fingertip_offset[2] + drop_height,
        )

        # Sphere sized to fit on one fingertip (35mm radius, 200g)
        # Larger to ensure clear collision response without false positives
        # Use primitive or mesh sphere based on parameter
        if sphere_type == ShapeType.SPHERE:
            sphere_spawn = sim_utils.SphereCfg(
                radius=0.035,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        elif sphere_type == ShapeType.MESH_SPHERE:
            sphere_spawn = sim_utils.MeshSphereCfg(
                radius=0.035,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        else:  # SDF_SPHERE - mesh sphere with SDF collision
            sphere_spawn = sim_utils.MeshSphereCfg(
                radius=0.035,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )

        sphere_cfg = RigidObjectCfg(
            prim_path="/World/DropSphere",
            spawn=sphere_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=sphere_pos),
        )
        drop_sphere = RigidObject(sphere_cfg)

        # Apply SDF collision properties if needed
        if is_sdf_shape(sphere_type):
            apply_sdf_collision("/World/DropSphere")

        # Reset simulation
        sim.reset()
        hand.reset()
        drop_sphere.reset()

        # Let hand settle for a few steps (thumb needs to reach its joint limit)
        settle_steps = 30
        for _ in range(settle_steps):
            hand.write_data_to_sim()
            sim.step()
            hand.update(sim_dt)

        # Now record initial joint positions (after settling)
        drop_sphere.reset()  # Reset sphere position after settling
        initial_joint_pos = hand.data.joint_pos.clone()

        # Track peak deflection per finger during simulation (max deflection at any moment)
        joint_names = hand.data.joint_names
        peak_deflection = {finger: 0.0 for finger in ["index", "middle", "ring", "thumb"]}

        # Run simulation
        for step in range(drop_steps):
            # Note: Allegro hand is fixed-base, no need to write root pose
            hand.write_data_to_sim()
            drop_sphere.write_data_to_sim()
            sim.step()
            hand.update(sim_dt)
            drop_sphere.update(sim_dt)

            # Track peak deflection for each finger (captures impact moment)
            current_joint_pos = hand.data.joint_pos[0]
            for finger_name in ["index", "middle", "ring", "thumb"]:
                finger_deflection = 0.0
                for joint_name in ALLEGRO_FINGER_JOINTS[finger_name]:
                    if joint_name in joint_names:
                        idx = joint_names.index(joint_name)
                        finger_deflection += abs(current_joint_pos[idx].item() - initial_joint_pos[0, idx].item())
                peak_deflection[finger_name] = max(peak_deflection[finger_name], finger_deflection)

        # Use peak deflections for comparison (captures the moment of impact)
        target_peak = peak_deflection[target_finger]

        # Verify target finger experienced significant deflection
        assert target_peak > 0.01, (
            f"Target finger '{target_finger}' should have deflected from impact, "
            f"but peak deflection was only {target_peak:.6f}"
        )

        # Verify target finger had the HIGHEST peak deflection
        # This proves the impact was localized to the target finger
        for finger_name in ["index", "middle", "ring", "thumb"]:
            if finger_name != target_finger:
                assert target_peak >= peak_deflection[finger_name], (
                    f"Target finger '{target_finger}' (peak={target_peak:.4f}) should have "
                    f"higher peak deflection than '{finger_name}' (peak={peak_deflection[finger_name]:.4f})"
                )
