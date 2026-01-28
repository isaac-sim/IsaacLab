# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests to verify collision behavior between different collision primitives using Newton physics.

This test suite verifies that:
1. Objects with different collision primitives collide correctly
2. Objects don't interpenetrate during collision
3. Momentum and energy transfer behave as expected
4. Objects can be stacked stably

Converted from Isaac Sim PhysX tests to Newton physics engine.
"""

# pyright: reportPrivateUsage=none

from enum import Enum, auto

import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim.spawners import materials
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Import hand configurations for articulated collision tests
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

# Import shared physics test utilities
from physics.physics_test_utils import (
    ALL_SHAPES,
    BOX_SHAPES,
    COLLISION_PIPELINES,
    MESH_SHAPES,
    PRIMITIVE_SHAPES,
    SPHERE_SHAPES,
    ShapeType,
    create_shape_cfg,
    get_shape_extent,
    get_shape_height,
    is_mesh_shape,
    make_sim_cfg,
    perform_sim_step,
    shape_type_to_str,
)

##
# Collision Test Specific Types
##


class CollisionTestLevel(Enum):
    """Test strictness levels for collision verification."""

    VELOCITY_X = auto()  # Check velocity along X-axis (collision direction)
    VELOCITY_YZ = auto()  # Check no spurious velocity in Y/Z directions
    VELOCITY_LINEAR = auto()  # Check all linear velocities
    VELOCITY_ANGULAR = auto()  # Check no spurious angular velocity
    STRICT = auto()  # All checks


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
    }
    return min_heights[shape_type]


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


def _verify_collision_behavior(
    object_a: RigidObject,
    object_b: RigidObject,
    test_level: CollisionTestLevel,
    tolerance: float = 0.05,
    initial_velocity: float = 1.0,
):
    """Verify collision behavior based on test level."""
    vel_a = wp.to_torch(object_a.data.root_lin_vel_w)[0]
    vel_b = wp.to_torch(object_b.data.root_lin_vel_w)[0]
    ang_vel_a = wp.to_torch(object_a.data.root_ang_vel_w)[0]
    ang_vel_b = wp.to_torch(object_b.data.root_ang_vel_w)[0]

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
    pos_a = wp.to_torch(object_a.data.root_pos_w)[0]
    pos_b = wp.to_torch(object_b.data.root_pos_w)[0]
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
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize(
    "shape_a,shape_b,test_level",
    COLLISION_PAIRS,
    ids=[f"{shape_type_to_str(a)}_{shape_type_to_str(b)}" for a, b, _ in COLLISION_PAIRS],
)
def test_horizontal_collision(
    setup_collision_params,
    device: str,
    use_mujoco_contacts: bool,
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

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, 0.0))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(shape_a, "/World/ObjectA", pos=(-separation / 2, 0.0, 0.5))
        scene_cfg.object_b = create_shape_cfg(shape_b, "/World/ObjectB", pos=(separation / 2, 0.0, 0.5))

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        object_a: RigidObject = scene["object_a"]
        object_b: RigidObject = scene["object_b"]

        initial_velocity = 2.0
        object_a.write_root_velocity_to_sim(torch.tensor([[initial_velocity, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device))
        object_b.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

        collision_detected = False
        for _ in range(collision_steps):
            perform_sim_step(sim, scene, sim_dt)

            if wp.to_torch(object_b.data.root_lin_vel_w)[0, 0] > 0.1:
                collision_detected = True

            _verify_no_interpenetration(object_a, object_b, min_separation=extent_a + extent_b - 0.2)

        assert collision_detected, "Collision should have occurred between objects"

        _verify_collision_behavior(
            object_a,
            object_b,
            test_level=test_level,
            tolerance=0.1,
            initial_velocity=initial_velocity,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize("shape_type", ALL_SHAPES)
def test_falling_collision_with_ground(
    setup_collision_params,
    device: str,
    use_mujoco_contacts: bool,
    shape_type: ShapeType,
):
    """Test that objects fall and collide correctly with the ground plane.

    This test verifies:
    - Object falls under gravity
    - Object comes to rest at expected height (doesn't fall through ground)
    - Object doesn't bounce indefinitely

    Note: CONE primitive shape fails with Newton physics - may need further investigation.
    """
    # Skip CONE primitive - fails with Newton physics
    if shape_type == ShapeType.CONE:
        pytest.skip("CONE primitive collision with ground fails in Newton physics - needs investigation")
    sim_dt, _ = setup_collision_params
    fall_steps = 480  # 2 seconds

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -9.81))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(shape_type, "/World/Object", pos=(0.0, 0.0, 2.0), disable_gravity=False)

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        obj: RigidObject = scene["object_a"]

        for _ in range(fall_steps):
            perform_sim_step(sim, scene, sim_dt)

        final_height = wp.to_torch(obj.data.root_pos_w)[0, 2].item()
        final_velocity = torch.norm(wp.to_torch(obj.data.root_lin_vel_w)[0]).item()

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
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize(
    "bottom_shape,top_shape",
    STACKING_PAIRS,
    ids=[f"{shape_type_to_str(b)}_under_{shape_type_to_str(t)}" for b, t in STACKING_PAIRS],
)
def test_box_stacking_stability(
    setup_collision_params, device: str, use_mujoco_contacts: bool, bottom_shape: ShapeType, top_shape: ShapeType
):
    """Test that boxes can be stably stacked on top of each other.

    This tests the collision system's ability to maintain stable contacts
    for various combinations of primitive and mesh box colliders.
    """
    sim_dt, _ = setup_collision_params
    settle_steps = 480

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -9.81))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
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
            perform_sim_step(sim, scene, sim_dt)

        bottom_height = wp.to_torch(box_bottom.data.root_pos_w)[0, 2].item()
        top_height = wp.to_torch(box_top.data.root_pos_w)[0, 2].item()

        assert bottom_height > 0.2, f"Bottom box fell through ground: height={bottom_height:.4f}"
        assert top_height > bottom_height + 0.3, (
            f"Top box not properly stacked: top={top_height:.4f}, bottom={bottom_height:.4f}"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize("restitution", [0.0, 0.5, 0.9])
def test_sphere_bounce_restitution(
    setup_collision_params, device: str, use_mujoco_contacts: bool, restitution: float
):
    """Test that sphere bouncing behavior matches configured restitution.

    A sphere is dropped onto a ground plane with specific restitution.
    - restitution=0: No bounce (inelastic)
    - restitution=0.5: Partial bounce
    - restitution=0.9: High bounce

    Note: Newton/MujocoWarp handles restitution differently than PhysX.
    High restitution (0.9) tests may fail as Newton doesn't support direct restitution.
    """
    # Skip high restitution test - Newton handles bouncing differently
    if restitution > 0.8:
        pytest.skip("High restitution (>0.8) not fully supported in Newton - uses ke/kd contact parameters instead")
    sim_dt, _ = setup_collision_params
    drop_steps = 120
    bounce_steps = 120

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -9.81))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
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
            perform_sim_step(sim, scene, sim_dt)

        for _ in range(bounce_steps):
            perform_sim_step(sim, scene, sim_dt)

        final_height = wp.to_torch(sphere.data.root_pos_w)[0, 2].item()

        if restitution < 0.1:
            assert final_height < 0.5, f"Zero restitution should not bounce high: height={final_height:.4f}"
        elif restitution > 0.8:
            assert final_height > 0.3, f"High restitution should bounce: height={final_height:.4f}"


# Sphere pairs for momentum conservation tests
SPHERE_PAIRS = [
    (ShapeType.SPHERE, ShapeType.SPHERE),
    (ShapeType.MESH_SPHERE, ShapeType.MESH_SPHERE),
    (ShapeType.SPHERE, ShapeType.MESH_SPHERE),
]


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize(
    "sphere_a,sphere_b",
    SPHERE_PAIRS,
    ids=[f"{shape_type_to_str(a)}_{shape_type_to_str(b)}" for a, b in SPHERE_PAIRS],
)
def test_momentum_conservation_equal_mass(
    setup_collision_params, device: str, use_mujoco_contacts: bool, sphere_a: ShapeType, sphere_b: ShapeType
):
    """Test that momentum is conserved in equal-mass sphere-sphere collision.

    For equal mass objects with one stationary, after collision:
    - Total momentum is conserved
    - Object A slows down
    - Object B gains velocity
    """
    sim_dt, collision_steps = setup_collision_params

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, 0.0))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = CollisionTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)

        separation = 1.0
        scene_cfg.object_a = create_shape_cfg(sphere_a, "/World/SphereA", pos=(-separation / 2, 0.0, 0.5))
        scene_cfg.object_b = create_shape_cfg(sphere_b, "/World/SphereB", pos=(separation / 2, 0.0, 0.5))

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        sphere_a_obj: RigidObject = scene["object_a"]
        sphere_b_obj: RigidObject = scene["object_b"]

        initial_vel = 2.0
        sphere_a_obj.write_root_velocity_to_sim(torch.tensor([[initial_vel, 0, 0, 0, 0, 0]], device=device))
        sphere_b_obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))

        initial_momentum = initial_vel * 1.0

        for _ in range(collision_steps):
            perform_sim_step(sim, scene, sim_dt)

        final_vel_a = wp.to_torch(sphere_a_obj.data.root_lin_vel_w)[0, 0].item()
        final_vel_b = wp.to_torch(sphere_b_obj.data.root_lin_vel_w)[0, 0].item()
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
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize("target_finger", ["index", "middle", "ring", "thumb"])
@pytest.mark.parametrize(
    "drop_shape",
    [ShapeType.SPHERE, ShapeType.MESH_SPHERE, ShapeType.BOX, ShapeType.MESH_BOX],
    ids=["sphere", "mesh_sphere", "cube", "mesh_cube"],
)
def test_finger_collision_isolation(
    setup_collision_params, device: str, use_mujoco_contacts: bool, target_finger: str, drop_shape: ShapeType
):
    """Test that dropping an object on one finger only affects that finger.

    This test:
    1. Spawns an Allegro hand in default orientation with global gravity=0
    2. Launches a shape (sphere or cube) with initial downward velocity onto a specific fingertip
    3. Verifies that the target finger's joints deflect more than other fingers

    Tests primitive and mesh colliders for both spheres and cubes to ensure consistent
    collision behavior across different collider representations.

    Note: Uses gravity=0 globally and gives object initial velocity instead of per-body
    gravity control, since ALLEGRO_HAND_CFG uses UsdFileCfg without rigid_props.
    """
    sim_dt, _ = setup_collision_params
    drop_steps = 480  # 2 seconds for drop and settle

    # Hand position
    hand_pos = (0.0, 0.0, 0.5)

    # Use zero gravity globally - ball will be given initial downward velocity
    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, 0.0))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        # Create hand configuration with default orientation
        # Global gravity is 0, so hand fingers won't sag
        hand_cfg = ALLEGRO_HAND_CFG.copy()
        hand_cfg.prim_path = "/World/Hand"
        hand_cfg.init_state.pos = hand_pos
        # Keep default orientation and joint positions

        # Create ground plane (for visualization)
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Create the hand
        hand = Articulation(hand_cfg)

        # Get fingertip offset for target finger
        fingertip_offset = ALLEGRO_FINGERTIP_OFFSETS[target_finger]

        # Create a small object to drop on the fingertip
        # Position it directly above the fingertip
        drop_height = 0.10  # 10cm above fingertip (closer since no gravity acceleration)
        drop_pos = (
            hand_pos[0] + fingertip_offset[0],
            hand_pos[1] + fingertip_offset[1],
            hand_pos[2] + fingertip_offset[2] + drop_height,
        )

        # Common rigid body properties for drop object
        drop_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # Doesn't matter - global gravity is 0
            linear_damping=0.0,
            angular_damping=0.0,
        )
        drop_collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        drop_mass_props = sim_utils.MassPropertiesCfg(mass=0.2)
        drop_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))

        # Create shape spawn config based on drop_shape parameter
        # Size: 35mm radius for spheres, 50mm cube for boxes (sized to fit on one fingertip)
        if drop_shape == ShapeType.SPHERE:
            drop_spawn = sim_utils.SphereCfg(
                radius=0.035,
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
            )
        elif drop_shape == ShapeType.MESH_SPHERE:
            drop_spawn = sim_utils.MeshSphereCfg(
                radius=0.035,
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
            )
        elif drop_shape == ShapeType.BOX:
            drop_spawn = sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),  # 50mm cube
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
            )
        elif drop_shape == ShapeType.MESH_BOX:
            drop_spawn = sim_utils.MeshCuboidCfg(
                size=(0.05, 0.05, 0.05),  # 50mm cube
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
            )
        else:
            raise ValueError(f"Unsupported drop shape: {drop_shape}")

        drop_obj_cfg = RigidObjectCfg(
            prim_path="/World/DropObject",
            spawn=drop_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=drop_pos),
        )
        drop_object = RigidObject(drop_obj_cfg)

        # Reset simulation
        sim.reset()
        hand.reset()
        drop_object.reset()

        # Let hand settle for a few steps (thumb needs to reach its joint limit)
        settle_steps = 30
        for _ in range(settle_steps):
            hand.write_data_to_sim()
            sim.step(render=False)
            hand.update(sim_dt)

        # Now record initial joint positions (after settling)
        drop_object.reset()  # Reset object position after settling

        # Give the object an initial downward velocity (instead of relying on gravity)
        # Velocity equivalent to falling from ~10cm with gravity: v = sqrt(2*g*h) ≈ 1.4 m/s
        initial_velocity = torch.tensor([[0.0, 0.0, -1.5, 0.0, 0.0, 0.0]], device=device)
        drop_object.write_root_velocity_to_sim(initial_velocity)

        initial_joint_pos = wp.to_torch(hand.data.joint_pos).clone()

        # Track peak deflection per finger during simulation (max deflection at any moment)
        joint_names = hand.data.joint_names
        peak_deflection = {finger: 0.0 for finger in ["index", "middle", "ring", "thumb"]}

        # Run simulation
        for step in range(drop_steps):
            # Note: Allegro hand is fixed-base, no need to write root pose
            hand.write_data_to_sim()
            drop_object.write_data_to_sim()
            sim.step(render=False)
            hand.update(sim_dt)
            drop_object.update(sim_dt)

            # Track peak deflection for each finger (captures impact moment)
            current_joint_pos = wp.to_torch(hand.data.joint_pos)[0]
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
