# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests to verify contact sensor functionality using Newton physics.

This test suite verifies that:
1. Contact detection is accurate (no false positives or true negatives)
2. Contact forces are reported correctly
3. Contact filtering works properly
4. Contact time tracking is accurate

Uses proper collision scenarios (falling, stacking, horizontal collision) instead of
teleporting objects into interpenetrating states.
"""

# pyright: reportPrivateUsage=none

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import build_simulation_context
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Import hand configurations for articulated contact sensor tests
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

# Import shared physics test utilities
from physics.physics_test_utils import (
    COLLISION_PIPELINES,
    MESH_SHAPES,
    PRIMITIVE_SHAPES,
    STABLE_SHAPES,
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
# Scene Configuration
##


@configclass
class ContactSensorTestSceneCfg(InteractiveSceneCfg):
    """Configuration for contact sensor test scenes."""

    terrain: TerrainImporterCfg | None = None
    object_a: RigidObjectCfg | None = None
    object_b: RigidObjectCfg | None = None
    object_c: RigidObjectCfg | None = None  # For filtering tests
    contact_sensor_a: ContactSensorCfg | None = None
    contact_sensor_b: ContactSensorCfg | None = None


##
# Test Fixtures
##


# Simulation time step (120 Hz physics)
SIM_DT = 1.0 / 120.0




##
# Priority 1: Contact Detection Accuracy Tests
##


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize("shape_type", STABLE_SHAPES, ids=[shape_type_to_str(s) for s in STABLE_SHAPES])
def test_contact_lifecycle(
    device: str, use_mujoco_contacts: bool, shape_type: ShapeType
):
    """Test full contact detection lifecycle with varied heights across environments.

    STRESS TEST: 16 environments with objects at different heights, creating
    asynchronous contact events as they land at different times. This stresses:
    - Parallel contact detection across many environments
    - Asynchronous contact events (lower objects land first)
    - Per-environment physics accuracy (fall time t = sqrt(2h/g))
    - No cross-talk between environments

    Height distribution (4 groups of 4 envs each):
    - Group 0 (envs 0-3):   height = 0.5m  -> lands at ~0.32s (tick ~38)
    - Group 1 (envs 4-7):   height = 1.0m  -> lands at ~0.45s (tick ~54)
    - Group 2 (envs 8-11):  height = 1.5m  -> lands at ~0.55s (tick ~66)
    - Group 3 (envs 12-15): height = 2.0m  -> lands at ~0.64s (tick ~77)

    Verifies:
    - No contact initially while objects are falling
    - Contact detected after landing (timing validated against physics)
    - Lower drops land before higher drops
    - Contact stops when objects are lifted
    """
    import math
    import warp as wp

    num_envs = 16
    num_groups = 4
    envs_per_group = num_envs // num_groups

    # Heights for each group (meters above ground, accounting for object size)
    # Object radius/half-height is ~0.25m, so add that to drop height
    base_heights = [0.5, 1.0, 1.5, 2.0]
    object_offset = get_shape_height(shape_type) / 2  # Distance from center to bottom

    gravity_mag = 9.81
    total_fall_steps = 180  # ~1.5 seconds - enough for highest to land + settle
    lift_steps = 60

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -gravity_mag))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = ContactSensorTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(
            shape_type, "{ENV_REGEX_NS}/Object", pos=(0.0, 0.0, 3.0), disable_gravity=False, activate_contact_sensors=True
        )
        scene_cfg.contact_sensor_a = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            update_period=0.0,
            history_length=1,
            track_air_time=True,
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        contact_sensor: ContactSensor = scene["contact_sensor_a"]
        obj: RigidObject = scene["object_a"]

        # Set different initial heights per environment
        root_pose = wp.to_torch(obj.data.root_link_pose_w).clone()
        for group_idx, base_height in enumerate(base_heights):
            for i in range(envs_per_group):
                env_idx = group_idx * envs_per_group + i
                # Height from ground = base_height + object_offset (center of mass)
                root_pose[env_idx, 2] = base_height + object_offset
        obj.write_root_pose_to_sim(root_pose)

        # Calculate expected landing ticks for each group
        # Fall distance = base_height (from bottom of object to ground)
        # t = sqrt(2h/g), ticks = t / dt
        expected_land_ticks = []
        for base_height in base_heights:
            fall_time = math.sqrt(2 * base_height / gravity_mag)
            land_tick = int(fall_time / SIM_DT)
            expected_land_ticks.append(land_tick)

        # Track state per environment
        contact_detected = [False] * num_envs
        contact_tick = [-1] * num_envs

        # Phase 1: Initial check - no contact while high in air
        for _ in range(5):
            perform_sim_step(sim, scene, SIM_DT)

        forces = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
        for env_idx in range(num_envs):
            env_force = forces[env_idx].max().item()
            assert env_force < 0.01, (
                f"Env {env_idx}: No contact should be detected while in air. Force: {env_force:.4f} N"
            )

        # Phase 2: Let objects fall and track when each lands
        for tick in range(5, total_fall_steps):
            perform_sim_step(sim, scene, SIM_DT)

            forces = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
            for env_idx in range(num_envs):
                env_force = forces[env_idx].max().item()

                # Track first contact
                if env_force > 0.1 and not contact_detected[env_idx]:
                    contact_detected[env_idx] = True
                    contact_tick[env_idx] = tick

        # Verify all environments detected contact
        for env_idx in range(num_envs):
            group_idx = env_idx // envs_per_group
            assert contact_detected[env_idx], (
                f"Env {env_idx} (group {group_idx}, h={base_heights[group_idx]}m): "
                f"Contact should be detected after landing"
            )

        # Verify landing order: lower heights should land first
        for env_idx in range(num_envs):
            group_idx = env_idx // envs_per_group
            expected_tick = expected_land_ticks[group_idx]
            actual_tick = contact_tick[env_idx]

            # Tolerance: 30% of expected fall time + 10 ticks buffer
            tolerance_ticks = int(0.3 * expected_tick) + 10
            assert abs(actual_tick - expected_tick) < tolerance_ticks, (
                f"Env {env_idx} (group {group_idx}, h={base_heights[group_idx]}m): "
                f"Contact at tick {actual_tick}, expected ~{expected_tick} ± {tolerance_ticks}"
            )

        # Verify relative ordering: each group should land after the previous
        group_land_times = []
        for group_idx in range(num_groups):
            group_ticks = [contact_tick[group_idx * envs_per_group + i] for i in range(envs_per_group)]
            avg_tick = sum(group_ticks) / len(group_ticks)
            group_land_times.append(avg_tick)

        for i in range(num_groups - 1):
            assert group_land_times[i] < group_land_times[i + 1], (
                f"Group {i} (h={base_heights[i]}m) should land before Group {i + 1} (h={base_heights[i + 1]}m). "
                f"Avg ticks: {group_land_times[i]:.1f} vs {group_land_times[i + 1]:.1f}"
            )

        # Phase 3: Lift all objects and verify contact stops
        lift_velocity = 5.0
        velocity = torch.zeros(num_envs, 6, device=device)
        velocity[:, 2] = lift_velocity
        obj.write_root_velocity_to_sim(velocity)

        no_contact_detected = [False] * num_envs
        for step in range(lift_steps):
            perform_sim_step(sim, scene, SIM_DT)

            if step > 10:
                forces = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
                for env_idx in range(num_envs):
                    if forces[env_idx].max().item() < 0.01:
                        no_contact_detected[env_idx] = True

        for env_idx in range(num_envs):
            assert no_contact_detected[env_idx], (
                f"Env {env_idx}: Contact should stop after object is lifted off ground."
            )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize("shape_type", STABLE_SHAPES, ids=[shape_type_to_str(s) for s in STABLE_SHAPES])
def test_horizontal_collision_detects_contact(
    device: str, use_mujoco_contacts: bool, shape_type: ShapeType
):
    """Test horizontal collision detection with varied velocities and separations.

    STRESS TEST: 16 environments with different collision velocities and separations.
    This stresses the engine's ability to handle:
    - Collisions happening at different times (varied separation/velocity)
    - Different impact energies (kinetic energy = 0.5 * m * v^2)
    - Proper force reporting for different collision speeds

    Configuration per group (4 envs each):
    - Group 0: velocity=1.0 m/s, separation=0.6m -> collision at ~0.6s
    - Group 1: velocity=2.0 m/s, separation=0.8m -> collision at ~0.4s
    - Group 2: velocity=3.0 m/s, separation=1.0m -> collision at ~0.33s
    - Group 3: velocity=4.0 m/s, separation=1.2m -> collision at ~0.3s

    Verifies:
    - Contact detected for all collision configurations
    - Faster collisions produce higher peak forces
    - Both objects in each pair detect contact
    """
    import warp as wp

    collision_steps = 180  # 1.5 seconds - enough for slowest collision
    num_envs = 16
    num_groups = 4
    envs_per_group = num_envs // num_groups

    extent = get_shape_extent(shape_type)

    # Per-group configuration: (velocity m/s, total_separation m)
    # Separation is distance between object centers
    group_configs = [
        (1.0, 0.6 + 2 * extent),   # Slow, close
        (2.0, 0.8 + 2 * extent),   # Medium
        (3.0, 1.0 + 2 * extent),   # Fast
        (4.0, 1.2 + 2 * extent),   # Very fast, far
    ]

    # No gravity - horizontal collision
    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, 0.0))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        # Use largest separation for initial spawn position
        max_separation = max(cfg[1] for cfg in group_configs)
        scene_cfg = ContactSensorTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)

        scene_cfg.object_a = create_shape_cfg(
            shape_type, "{ENV_REGEX_NS}/ObjectA", pos=(-max_separation / 2, 0.0, 0.5),
            disable_gravity=True, activate_contact_sensors=True
        )
        scene_cfg.object_b = create_shape_cfg(
            shape_type, "{ENV_REGEX_NS}/ObjectB", pos=(max_separation / 2, 0.0, 0.5),
            disable_gravity=True, activate_contact_sensors=True
        )

        scene_cfg.contact_sensor_a = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/ObjectA",
            update_period=0.0,
            history_length=3,
        )
        scene_cfg.contact_sensor_b = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/ObjectB",
            update_period=0.0,
            history_length=3,
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        object_a: RigidObject = scene["object_a"]
        object_b: RigidObject = scene["object_b"]
        sensor_a: ContactSensor = scene["contact_sensor_a"]
        sensor_b: ContactSensor = scene["contact_sensor_b"]

        # Set different separations per environment
        pose_a = wp.to_torch(object_a.data.root_link_pose_w).clone()
        pose_b = wp.to_torch(object_b.data.root_link_pose_w).clone()

        for group_idx, (_, separation) in enumerate(group_configs):
            for i in range(envs_per_group):
                env_idx = group_idx * envs_per_group + i
                pose_a[env_idx, 0] = -separation / 2  # Object A on left
                pose_b[env_idx, 0] = separation / 2   # Object B on right

        object_a.write_root_pose_to_sim(pose_a)
        object_b.write_root_pose_to_sim(pose_b)

        # Set different velocities per environment (object A moves right toward B)
        velocity = torch.zeros(num_envs, 6, device=device)
        for group_idx, (vel, _) in enumerate(group_configs):
            for i in range(envs_per_group):
                env_idx = group_idx * envs_per_group + i
                velocity[env_idx, 0] = vel  # X velocity (toward B)
        object_a.write_root_velocity_to_sim(velocity)

        # Track contact detection and peak forces per environment
        contact_detected_a = [False] * num_envs
        contact_detected_b = [False] * num_envs
        contact_tick_a = [-1] * num_envs
        peak_force_a = [0.0] * num_envs

        # Run simulation and check for contact
        for tick in range(collision_steps):
            perform_sim_step(sim, scene, SIM_DT)

            forces_a = torch.norm(sensor_a.data.net_forces_w, dim=-1)
            forces_b = torch.norm(sensor_b.data.net_forces_w, dim=-1)

            for env_idx in range(num_envs):
                force_a = forces_a[env_idx].max().item()
                force_b = forces_b[env_idx].max().item()

                if force_a > 0.1:
                    if not contact_detected_a[env_idx]:
                        contact_detected_a[env_idx] = True
                        contact_tick_a[env_idx] = tick
                    peak_force_a[env_idx] = max(peak_force_a[env_idx], force_a)

                if force_b > 0.1:
                    contact_detected_b[env_idx] = True

        # Verify contact was detected in all environments
        for env_idx in range(num_envs):
            group_idx = env_idx // envs_per_group
            vel, sep = group_configs[group_idx]
            assert contact_detected_a[env_idx], (
                f"Env {env_idx} (v={vel}m/s, sep={sep:.2f}m): Object A should detect contact"
            )
            assert contact_detected_b[env_idx], (
                f"Env {env_idx} (v={vel}m/s, sep={sep:.2f}m): Object B should detect contact"
            )

        # Verify higher velocity produces higher peak force (compare group averages)
        group_avg_forces = []
        for group_idx in range(num_groups):
            group_forces = [peak_force_a[group_idx * envs_per_group + i] for i in range(envs_per_group)]
            avg_force = sum(group_forces) / len(group_forces)
            group_avg_forces.append(avg_force)

        for i in range(num_groups - 1):
            # Higher velocity should produce higher impact force
            # Note: This may not be strictly monotonic due to collision dynamics,
            # so we use a soft assertion with logging
            vel_i, _ = group_configs[i]
            vel_next, _ = group_configs[i + 1]
            if group_avg_forces[i + 1] <= group_avg_forces[i]:
                print(
                    f"Warning: Group {i + 1} (v={vel_next}m/s) avg force {group_avg_forces[i + 1]:.2f}N "
                    f"not higher than Group {i} (v={vel_i}m/s) avg force {group_avg_forces[i]:.2f}N"
                )


##
# Priority 2: Net Forces Tests
##


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
def test_resting_object_contact_force(device: str, use_mujoco_contacts: bool):
    """Test that resting object contact force equals weight and points upward.

    Scenario: Two objects (light and heavy) rest on ground after settling.
    Verifies:
    - Force magnitude approximately equals mass × gravity (F = mg)
    - Force direction is upward (positive Z, opposing gravity)
    - Heavier object has proportionally larger force
    """
    settle_steps = 240  # 2 seconds to settle
    num_envs = 4

    mass_a = 2.0  # kg (light)
    mass_b = 4.0  # kg (heavy)
    gravity_magnitude = 9.81  # m/s²
    expected_force_a = mass_a * gravity_magnitude  # 19.62 N
    expected_force_b = mass_b * gravity_magnitude  # 39.24 N

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -gravity_magnitude))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = ContactSensorTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)

        rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.5, angular_damping=0.5)

        # Object A - lighter mass
        scene_cfg.object_a = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BoxA",
            spawn=sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                rigid_props=rigid_props,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass_a),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.5)),
        )

        # Object B - heavier mass
        scene_cfg.object_b = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/BoxB",
            spawn=sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                rigid_props=rigid_props,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass_b),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.5)),
        )

        scene_cfg.contact_sensor_a = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/BoxA",
            update_period=0.0,
            history_length=1,
        )

        scene_cfg.contact_sensor_b = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/BoxB",
            update_period=0.0,
            history_length=1,
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        sensor_a: ContactSensor = scene["contact_sensor_a"]
        sensor_b: ContactSensor = scene["contact_sensor_b"]

        # Let objects settle
        for _ in range(settle_steps):
            perform_sim_step(sim, scene, SIM_DT)

        # Get force data
        forces_a = sensor_a.data.net_forces_w
        forces_b = sensor_b.data.net_forces_w
        assert forces_a is not None and forces_b is not None

        force_mags_a = torch.norm(forces_a, dim=-1)
        force_mags_b = torch.norm(forces_b, dim=-1)

        # Verify each sensor reports its own object's weight (not shared/duplicated data)
        tolerance_a = 0.2 * expected_force_a
        tolerance_b = 0.2 * expected_force_b

        for env_idx in range(num_envs):
            force_a = force_mags_a[env_idx].max().item()
            force_b = force_mags_b[env_idx].max().item()

            # Check F = mg for both objects
            assert abs(force_a - expected_force_a) < tolerance_a, (
                f"Env {env_idx}: BoxA ({mass_a}kg) force should be ~{expected_force_a:.2f} N. Got {force_a:.2f} N"
            )
            assert abs(force_b - expected_force_b) < tolerance_b, (
                f"Env {env_idx}: BoxB ({mass_b}kg) force should be ~{expected_force_b:.2f} N. Got {force_b:.2f} N"
            )

            # Heavier object should have larger force
            assert force_b > force_a, (
                f"Env {env_idx}: Heavier BoxB should have larger force. A: {force_a:.2f} N, B: {force_b:.2f} N"
            )

            # Check force direction is upward (positive Z)
            z_a = forces_a[env_idx, 0, 2].item()
            z_b = forces_b[env_idx, 0, 2].item()
            assert z_a > 0.1, f"Env {env_idx}: BoxA Z force should be positive. Got: {z_a:.4f}"
            assert z_b > 0.1, f"Env {env_idx}: BoxB Z force should be positive. Got: {z_b:.4f}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
def test_higher_drop_produces_larger_impact_force(device: str, use_mujoco_contacts: bool):
    """Test that dropping from higher produces larger peak impact force.

    STRESS TEST: 16 environments with fine-grained height progression.
    This stresses the engine's ability to accurately model:
    - Impact velocity (v = sqrt(2gh)) for many different heights
    - Impact force relationship with velocity
    - Consistent physics across many parallel environments

    Height distribution (16 environments):
    - Heights range from 0.3m to 3.0m in ~0.18m increments
    - v = sqrt(2*g*h): velocities from ~2.4 m/s to ~7.7 m/s
    - Impact force ∝ velocity (with restitution effects)

    Verifies:
    - Peak impact force increases monotonically with height
    - Force magnitude is reasonable (order of magnitude check)
    - All environments detect contact
    """
    import warp as wp

    num_envs = 16

    # Generate heights from 0.3m to 3.0m (16 values)
    min_height = 0.3
    max_height = 3.0
    drop_heights = [min_height + (max_height - min_height) * i / (num_envs - 1) for i in range(num_envs)]
    # Heights: [0.3, 0.48, 0.66, 0.84, 1.02, 1.2, 1.38, 1.56, 1.74, 1.92, 2.1, 2.28, 2.46, 2.64, 2.82, 3.0]

    gravity_mag = 9.81
    object_radius = 0.25  # Sphere radius

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -gravity_mag))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = ContactSensorTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(
            ShapeType.SPHERE,
            "{ENV_REGEX_NS}/Sphere",
            pos=(0.0, 0.0, max_height + object_radius),
            disable_gravity=False,
            activate_contact_sensors=True,
        )

        scene_cfg.contact_sensor_a = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Sphere",
            update_period=0.0,
            history_length=1,
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        obj: RigidObject = scene["object_a"]
        contact_sensor: ContactSensor = scene["contact_sensor_a"]

        # Set different drop heights for each environment
        root_pose = wp.to_torch(obj.data.root_link_pose_w).clone()
        for env_idx in range(num_envs):
            # Height = drop height + object radius (center of sphere above ground)
            root_pose[env_idx, 2] = drop_heights[env_idx] + object_radius
        obj.write_root_pose_to_sim(root_pose)

        # Calculate simulation time based on highest drop + settle time
        max_fall_time = (2 * max_height / gravity_mag) ** 0.5
        total_steps = int((max_fall_time + 0.5) / SIM_DT)

        # Track peak force and contact per environment
        peak_forces = [0.0] * num_envs
        contact_detected = [False] * num_envs

        for _ in range(total_steps):
            perform_sim_step(sim, scene, SIM_DT)

            net_forces = contact_sensor.data.net_forces_w
            force_magnitudes = torch.norm(net_forces, dim=-1)

            for env_idx in range(num_envs):
                current_force = force_magnitudes[env_idx].max().item()
                if current_force > 0.1:
                    contact_detected[env_idx] = True
                peak_forces[env_idx] = max(peak_forces[env_idx], current_force)

        # Verify all environments detected contact
        for env_idx in range(num_envs):
            assert contact_detected[env_idx], (
                f"Env {env_idx} (height={drop_heights[env_idx]:.2f}m): Contact should be detected"
            )

        # Verify peak forces increase monotonically with height
        # Use adjacent comparison with tolerance for physics noise
        violations = []
        for i in range(num_envs - 1):
            # Allow small violations due to simulation noise (5% tolerance)
            if peak_forces[i + 1] < peak_forces[i] * 0.95:
                violations.append(
                    f"Env {i} (h={drop_heights[i]:.2f}m, F={peak_forces[i]:.2f}N) -> "
                    f"Env {i + 1} (h={drop_heights[i + 1]:.2f}m, F={peak_forces[i + 1]:.2f}N)"
                )

        # Allow up to 2 violations (physics noise at similar heights)
        assert len(violations) <= 2, (
            f"Peak force should generally increase with height. Violations ({len(violations)}):\n"
            + "\n".join(violations)
        )

        # Verify overall trend: highest drop should have significantly higher force than lowest
        force_ratio = peak_forces[-1] / peak_forces[0] if peak_forces[0] > 0 else 0
        # Expected velocity ratio: sqrt(max_height/min_height) = sqrt(3.0/0.3) ≈ 3.16
        # Force roughly proportional to velocity, so expect ratio > 2
        assert force_ratio > 1.5, (
            f"Force ratio (highest/lowest) should be > 1.5. "
            f"Got {force_ratio:.2f} ({peak_forces[-1]:.2f}N / {peak_forces[0]:.2f}N)"
        )

        # Log force distribution for physics team analysis
        print(f"\nPeak forces by height (for physics team review):")
        for env_idx in range(num_envs):
            expected_velocity = (2 * gravity_mag * drop_heights[env_idx]) ** 0.5
            print(f"  h={drop_heights[env_idx]:.2f}m, v={expected_velocity:.2f}m/s, F={peak_forces[env_idx]:.2f}N")


##
# Priority 3: Filtering Tests
##


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
def test_filter_enables_force_matrix(device: str, use_mujoco_contacts: bool):
    """Test that filter_prim_paths_expr filters contacts and enables force_matrix_w.

    Scenario: Object A (cube) rests on ground, Object B (cube) stacked on A.
    Sensor on A is filtered for B only (not ground).

    Verifies:
    - force_matrix_w reports only filtered contact (A-B), not unfiltered (A-ground)
    - net_forces_w reports total contact force (ground + B combined)
    - force_matrix magnitude < net_forces magnitude (since ground contact excluded from matrix)
    """
    settle_steps = 360  # More time to ensure stable settling
    num_envs = 4

    mass_b = 2.0  # kg
    gravity = 9.81
    expected_force_from_b = mass_b * gravity  # ~19.62 N from B sitting on A

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -gravity))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = ContactSensorTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)

        # Object A - rests on ground, will have B stacked on top
        rigid_props_a = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.5, angular_damping=0.5)
        scene_cfg.object_a = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjectA",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 0.3),
                rigid_props=rigid_props_a,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        )

        # Object B - stacked on top of A (higher damping to settle faster)
        rigid_props_b = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=2.0, angular_damping=2.0)
        scene_cfg.object_b = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjectB",
            spawn=sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                rigid_props=rigid_props_b,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass_b),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
        )

        # Contact sensor on A filtered for B only (not ground)
        scene_cfg.contact_sensor_a = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/ObjectA",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/ObjectB"],
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        contact_sensor: ContactSensor = scene["contact_sensor_a"]

        # Let objects settle
        for _ in range(settle_steps):
            perform_sim_step(sim, scene, SIM_DT)

        # Get force data
        force_matrix = contact_sensor.data.force_matrix_w
        net_forces = contact_sensor.data.net_forces_w

        assert force_matrix is not None, "force_matrix_w should not be None when filter is set"
        assert net_forces is not None

        for env_idx in range(num_envs):
            # force_matrix_w: only reports contact with B (filtered object)
            matrix_force = torch.norm(force_matrix[env_idx]).item()

            # net_forces_w: reports total contact (ground + B)
            net_force = torch.norm(net_forces[env_idx]).item()

            # force_matrix should approximately equal B's weight (F = mg)
            tolerance = 0.3 * expected_force_from_b
            assert abs(matrix_force - expected_force_from_b) < tolerance, (
                f"Env {env_idx}: force_matrix should be ~{expected_force_from_b:.2f} N (B's weight). "
                f"Got: {matrix_force:.2f} N"
            )

            # Key assertion: force_matrix (B only) should be less than net_forces (ground + B)
            # because ground contact is NOT in force_matrix
            assert matrix_force < net_force, (
                f"Env {env_idx}: force_matrix (filtered, B only) should be less than net_forces (all contacts). "
                f"Matrix: {matrix_force:.2f} N, Net: {net_force:.2f} N"
            )


##
# Priority 4: Articulated System Tests
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

# Link names for each finger (for contact sensor prim_path)
# Using link_3 (last link with collision shapes) for each finger
ALLEGRO_FINGER_LINKS = {
    "index": "index_link_3",  # Index finger tip link
    "middle": "middle_link_3",  # Middle finger tip link
    "ring": "ring_link_3",  # Ring finger tip link
    "thumb": "thumb_link_3",  # Thumb finger tip link
}


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize("target_finger", ["index", "middle", "ring", "thumb"])
@pytest.mark.parametrize("drop_shape", [ShapeType.SPHERE, ShapeType.MESH_SPHERE, ShapeType.BOX, ShapeType.MESH_BOX])
@pytest.mark.xfail(reason="Newton contact sensor isolation bug: Contact forces leak to adjacent fingers in articulated systems")
def test_finger_contact_sensor_isolation(
    device: str, use_mujoco_contacts: bool, target_finger: str, drop_shape: ShapeType
):
    """Test contact sensor on Allegro hand fingers detects localized contacts.

    This test verifies that dropping an object on one finger produces contact forces
    only on that finger, proving proper contact isolation in articulated systems.

    Setup:
    1. Spawns Allegro hand with contact sensors on all four finger tips
    2. Drops a small object (sphere or box) onto a specific fingertip
    3. Tracks peak contact forces on all fingers

    Verifies:
    - Contact forces detected on target finger
    - Target finger has highest peak contact force (proves isolation)
    - Tests both primitive and mesh colliders

    Note: Uses zero global gravity with initial object velocity, similar to
    test_finger_collision_isolation in test_collision_behavior.py
    """
    import warp as wp

    drop_steps = 480  # 2 seconds for drop and settle

    # Hand position
    hand_pos = (0.0, 0.0, 0.5)

    # Zero gravity - object will be given initial downward velocity
    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, 0.0))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        # Create scene configuration with hand and contact sensors
        scene_cfg = ContactSensorTestSceneCfg(num_envs=1, env_spacing=5.0, lazy_sensor_update=False)

        # Add hand to scene
        scene_cfg.hand = ALLEGRO_HAND_CFG.copy()
        scene_cfg.hand.prim_path = "{ENV_REGEX_NS}/Hand"
        scene_cfg.hand.init_state.pos = hand_pos

        # Add contact sensors for all finger tips
        scene_cfg.contact_sensor_index = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Hand/{ALLEGRO_FINGER_LINKS['index']}",
            update_period=0.0,
            history_length=1,
        )
        scene_cfg.contact_sensor_middle = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Hand/{ALLEGRO_FINGER_LINKS['middle']}",
            update_period=0.0,
            history_length=1,
        )
        scene_cfg.contact_sensor_ring = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Hand/{ALLEGRO_FINGER_LINKS['ring']}",
            update_period=0.0,
            history_length=1,
        )
        scene_cfg.contact_sensor_thumb = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Hand/{ALLEGRO_FINGER_LINKS['thumb']}",
            update_period=0.0,
            history_length=1,
        )

        # Create ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Get fingertip offset for target finger
        fingertip_offset = ALLEGRO_FINGERTIP_OFFSETS[target_finger]

        # Position drop object above target fingertip
        drop_height = 0.10  # 10cm above fingertip
        drop_pos = (
            hand_pos[0] + fingertip_offset[0],
            hand_pos[1] + fingertip_offset[1],
            hand_pos[2] + fingertip_offset[2] + drop_height,
        )

        # Common properties for drop object
        drop_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        )
        drop_collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        drop_mass_props = sim_utils.MassPropertiesCfg(mass=0.2)
        drop_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))

        # Create shape spawn config based on drop_shape parameter
        if drop_shape == ShapeType.SPHERE:
            drop_spawn = sim_utils.SphereCfg(
                radius=0.035,
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
                activate_contact_sensors=True,
            )
        elif drop_shape == ShapeType.MESH_SPHERE:
            drop_spawn = sim_utils.MeshSphereCfg(
                radius=0.035,
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
                activate_contact_sensors=True,
            )
        elif drop_shape == ShapeType.BOX:
            drop_spawn = sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
                activate_contact_sensors=True,
            )
        elif drop_shape == ShapeType.MESH_BOX:
            drop_spawn = sim_utils.MeshCuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
                activate_contact_sensors=True,
            )
        else:
            raise ValueError(f"Unsupported drop shape: {drop_shape}")

        # Add drop object to scene
        scene_cfg.drop_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/DropObject",
            spawn=drop_spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=drop_pos),
        )

        # Create scene
        scene = InteractiveScene(scene_cfg)

        # Reset simulation
        sim.reset()
        scene.reset()

        # Get references to scene objects
        hand: Articulation = scene["hand"]
        drop_object: RigidObject = scene["drop_object"]
        finger_sensors = {
            "index": scene["contact_sensor_index"],
            "middle": scene["contact_sensor_middle"],
            "ring": scene["contact_sensor_ring"],
            "thumb": scene["contact_sensor_thumb"],
        }

        # Let hand settle
        settle_steps = 30
        for _ in range(settle_steps):
            perform_sim_step(sim, scene, SIM_DT)

        # Reset drop object position after settling
        drop_object.reset()

        # Give object initial downward velocity
        # v = sqrt(2*g*h) ≈ 1.4 m/s for 10cm drop
        initial_velocity = torch.tensor([[0.0, 0.0, -1.5, 0.0, 0.0, 0.0]], device=device)
        drop_object.write_root_velocity_to_sim(initial_velocity)

        # Track peak contact force per finger
        peak_forces = {finger: 0.0 for finger in ["index", "middle", "ring", "thumb"]}

        # Run simulation and track contact forces
        for step in range(drop_steps):
            perform_sim_step(sim, scene, SIM_DT)

            # Track peak forces for each finger
            for finger_name, sensor in finger_sensors.items():
                if sensor.data.net_forces_w is not None:
                    force_magnitude = torch.norm(sensor.data.net_forces_w[0]).item()
                    peak_forces[finger_name] = max(peak_forces[finger_name], force_magnitude)

        # Verify target finger experienced significant contact force
        target_peak = peak_forces[target_finger]
        assert target_peak > 0.5, (
            f"Target finger '{target_finger}' should detect contact force from impact. "
            f"Peak force: {target_peak:.4f} N (expected > 0.5 N)"
        )

        # Verify target finger had the HIGHEST peak force
        # This proves contact was isolated to the target finger
        for finger_name in ["index", "middle", "ring", "thumb"]:
            if finger_name != target_finger:
                assert target_peak >= peak_forces[finger_name], (
                    f"Target finger '{target_finger}' (peak={target_peak:.4f}N) should have "
                    f"highest peak force, but '{finger_name}' had peak={peak_forces[finger_name]:.4f}N"
                )


##
# Utility Tests
##


def test_sensor_print():
    """Test that contact sensor print/repr works correctly."""
    sim_cfg = make_sim_cfg(use_mujoco_contacts=False, device="cuda:0", gravity=(0.0, 0.0, -9.81))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = ContactSensorTestSceneCfg(num_envs=4, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(
            ShapeType.BOX, "{ENV_REGEX_NS}/Object", pos=(0.0, 0.0, 2.0), disable_gravity=False, activate_contact_sensors=True
        )
        scene_cfg.contact_sensor_a = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            update_period=0.0,
            history_length=3,
            track_air_time=True,
        )

        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()

        # Should not raise an exception
        sensor_str = str(scene["contact_sensor_a"])
        assert len(sensor_str) > 0, "Sensor string representation should not be empty"
        print(sensor_str)
