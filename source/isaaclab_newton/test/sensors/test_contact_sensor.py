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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pytest
import torch
import warp as wp
from flaky import flaky
from physics.physics_test_utils import (
    COLLISION_PIPELINES,
    STABLE_SHAPES,
    ShapeType,
    create_shape_cfg,
    get_shape_extent,
    get_shape_height,
    make_sim_cfg,
    perform_sim_step,
    shape_type_to_str,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import build_simulation_context
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

##
# Scene Configuration
##


@configclass
class ContactSensorTestSceneCfg(InteractiveSceneCfg):
    """Configuration for contact sensor test scenes."""

    terrain: TerrainImporterCfg | None = None
    object_a: RigidObjectCfg | None = None
    object_b: RigidObjectCfg | None = None
    object_c: RigidObjectCfg | None = None
    contact_sensor_a: ContactSensorCfg | None = None
    contact_sensor_b: ContactSensorCfg | None = None


SIM_DT = 1.0 / 120.0


# ===================================================================
# Priority 1: Contact Detection Accuracy
# ===================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize("shape_type", STABLE_SHAPES, ids=[shape_type_to_str(s) for s in STABLE_SHAPES])
def test_contact_lifecycle(device: str, use_mujoco_contacts: bool, shape_type: ShapeType):
    """Test full contact detection lifecycle with varied heights across environments.

    8 environments (2 groups x 4 envs) with objects at different heights.

    Verifies:
    - No contact initially while objects are falling
    - Contact detected after landing (timing validated against physics)
    - Lower drops land before higher drops
    - Contact stops when objects are lifted
    """
    num_envs = 8
    num_groups = 2
    envs_per_group = num_envs // num_groups

    base_heights = [0.5, 1.5]
    object_offset = get_shape_height(shape_type) / 2

    gravity_mag = 9.81
    total_fall_steps = 100
    lift_steps = 30

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -gravity_mag))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        scene_cfg = ContactSensorTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(
            shape_type,
            "{ENV_REGEX_NS}/Object",
            pos=(0.0, 0.0, 3.0),
            disable_gravity=False,
            activate_contact_sensors=True,
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

        root_pose = wp.to_torch(obj.data.root_link_pose_w).clone()
        for group_idx, base_height in enumerate(base_heights):
            for i in range(envs_per_group):
                env_idx = group_idx * envs_per_group + i
                root_pose[env_idx, 2] = base_height + object_offset
        obj.write_root_pose_to_sim_index(root_pose=root_pose)

        expected_land_ticks = [int(math.sqrt(2 * h / gravity_mag) / SIM_DT) for h in base_heights]

        contact_detected = [False] * num_envs
        contact_tick = [-1] * num_envs

        for _ in range(5):
            perform_sim_step(sim, scene, SIM_DT)

        forces = torch.norm(wp.to_torch(contact_sensor.data.net_forces_w), dim=-1)
        for env_idx in range(num_envs):
            assert forces[env_idx].max().item() < 0.01, f"Env {env_idx}: No contact should be detected while in air."

        for tick in range(5, total_fall_steps):
            perform_sim_step(sim, scene, SIM_DT)
            forces = torch.norm(wp.to_torch(contact_sensor.data.net_forces_w), dim=-1)
            for env_idx in range(num_envs):
                if forces[env_idx].max().item() > 0.1 and not contact_detected[env_idx]:
                    contact_detected[env_idx] = True
                    contact_tick[env_idx] = tick

        for env_idx in range(num_envs):
            group_idx = env_idx // envs_per_group
            assert contact_detected[env_idx], (
                f"Env {env_idx} (group {group_idx}, h={base_heights[group_idx]}m): Contact should be detected"
            )

        for env_idx in range(num_envs):
            group_idx = env_idx // envs_per_group
            expected_tick = expected_land_ticks[group_idx]
            tolerance_ticks = int(0.3 * expected_tick) + 10
            assert abs(contact_tick[env_idx] - expected_tick) < tolerance_ticks, (
                f"Env {env_idx}: Contact at tick {contact_tick[env_idx]}, expected ~{expected_tick} ± {tolerance_ticks}"
            )

        group_land_times = []
        for group_idx in range(num_groups):
            group_ticks = [contact_tick[group_idx * envs_per_group + i] for i in range(envs_per_group)]
            group_land_times.append(sum(group_ticks) / len(group_ticks))

        for i in range(num_groups - 1):
            assert group_land_times[i] < group_land_times[i + 1], (
                f"Group {i} should land before Group {i + 1}. "
                f"Avg ticks: {group_land_times[i]:.1f} vs {group_land_times[i + 1]:.1f}"
            )

        velocity = torch.zeros(num_envs, 6, device=device)
        velocity[:, 2] = 5.0
        obj.write_root_velocity_to_sim_index(root_velocity=velocity)

        no_contact_detected = [False] * num_envs
        for step in range(lift_steps):
            perform_sim_step(sim, scene, SIM_DT)
            if step > 10:
                forces = torch.norm(wp.to_torch(contact_sensor.data.net_forces_w), dim=-1)
                for env_idx in range(num_envs):
                    if forces[env_idx].max().item() < 0.01:
                        no_contact_detected[env_idx] = True

        for env_idx in range(num_envs):
            assert no_contact_detected[env_idx], f"Env {env_idx}: Contact should stop after lift."


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
@pytest.mark.parametrize("shape_type", STABLE_SHAPES, ids=[shape_type_to_str(s) for s in STABLE_SHAPES])
def test_horizontal_collision_detects_contact(device: str, use_mujoco_contacts: bool, shape_type: ShapeType):
    """Test horizontal collision detection with varied velocities and separations.

    8 environments (2 groups x 4 envs) with different collision speeds.

    Verifies:
    - Contact detected for all collision configurations
    - Both objects in each pair detect contact
    """
    collision_steps = 90
    num_envs = 8
    num_groups = 2
    envs_per_group = num_envs // num_groups
    extent = get_shape_extent(shape_type)

    group_configs = [
        (2.0, 0.6 + 2 * extent),
        (4.0, 0.8 + 2 * extent),
    ]

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, 0.0))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        max_separation = max(cfg[1] for cfg in group_configs)
        scene_cfg = ContactSensorTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(
            shape_type,
            "{ENV_REGEX_NS}/ObjectA",
            pos=(-max_separation / 2, 0.0, 0.5),
            disable_gravity=True,
            activate_contact_sensors=True,
        )
        scene_cfg.object_b = create_shape_cfg(
            shape_type,
            "{ENV_REGEX_NS}/ObjectB",
            pos=(max_separation / 2, 0.0, 0.5),
            disable_gravity=True,
            activate_contact_sensors=True,
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

        pose_a = wp.to_torch(object_a.data.root_link_pose_w).clone()
        pose_b = wp.to_torch(object_b.data.root_link_pose_w).clone()
        for group_idx, (_, separation) in enumerate(group_configs):
            for i in range(envs_per_group):
                env_idx = group_idx * envs_per_group + i
                pose_a[env_idx, 0] = -separation / 2
                pose_b[env_idx, 0] = separation / 2
        object_a.write_root_pose_to_sim_index(root_pose=pose_a)
        object_b.write_root_pose_to_sim_index(root_pose=pose_b)

        velocity = torch.zeros(num_envs, 6, device=device)
        for group_idx, (vel, _) in enumerate(group_configs):
            for i in range(envs_per_group):
                velocity[group_idx * envs_per_group + i, 0] = vel
        object_a.write_root_velocity_to_sim_index(root_velocity=velocity)

        contact_detected_a = [False] * num_envs
        contact_detected_b = [False] * num_envs

        for tick in range(collision_steps):
            perform_sim_step(sim, scene, SIM_DT)
            forces_a = torch.norm(wp.to_torch(sensor_a.data.net_forces_w), dim=-1)
            forces_b = torch.norm(wp.to_torch(sensor_b.data.net_forces_w), dim=-1)
            for env_idx in range(num_envs):
                if forces_a[env_idx].max().item() > 0.1:
                    contact_detected_a[env_idx] = True
                if forces_b[env_idx].max().item() > 0.1:
                    contact_detected_b[env_idx] = True

        for env_idx in range(num_envs):
            group_idx = env_idx // envs_per_group
            vel, sep = group_configs[group_idx]
            assert contact_detected_a[env_idx], f"Env {env_idx} (v={vel}m/s): Object A should detect contact"
            assert contact_detected_b[env_idx], f"Env {env_idx} (v={vel}m/s): Object B should detect contact"


# ===================================================================
# Priority 2: Net Forces
# ===================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
def test_resting_object_contact_force(device: str, use_mujoco_contacts: bool):
    """Test that resting object contact force equals weight and points upward.

    Two objects (light=2kg and heavy=4kg) rest on ground.

    Verifies:
    - Force magnitude ~ mass x gravity
    - Force direction is upward (positive Z)
    - Heavier object has proportionally larger force
    """
    settle_steps = 120
    num_envs = 4
    mass_a, mass_b = 2.0, 4.0
    gravity_magnitude = 9.81
    expected_force_a = mass_a * gravity_magnitude
    expected_force_b = mass_b * gravity_magnitude

    sim_cfg = make_sim_cfg(
        use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -gravity_magnitude)
    )

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = ContactSensorTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)
        rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.5, angular_damping=0.5)

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
            prim_path="{ENV_REGEX_NS}/BoxA", update_period=0.0, history_length=1
        )
        scene_cfg.contact_sensor_b = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/BoxB", update_period=0.0, history_length=1
        )

        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        sensor_a: ContactSensor = scene["contact_sensor_a"]
        sensor_b: ContactSensor = scene["contact_sensor_b"]

        for _ in range(settle_steps):
            perform_sim_step(sim, scene, SIM_DT)

        forces_a = wp.to_torch(sensor_a.data.net_forces_w)
        forces_b = wp.to_torch(sensor_b.data.net_forces_w)
        force_mags_a = torch.norm(forces_a, dim=-1)
        force_mags_b = torch.norm(forces_b, dim=-1)

        for env_idx in range(num_envs):
            fa = force_mags_a[env_idx].max().item()
            fb = force_mags_b[env_idx].max().item()

            assert abs(fa - expected_force_a) < 0.2 * expected_force_a, (
                f"Env {env_idx}: BoxA ({mass_a}kg) force should be ~{expected_force_a:.2f} N. Got {fa:.2f} N"
            )
            assert abs(fb - expected_force_b) < 0.2 * expected_force_b, (
                f"Env {env_idx}: BoxB ({mass_b}kg) force should be ~{expected_force_b:.2f} N. Got {fb:.2f} N"
            )
            assert fb > fa, f"Env {env_idx}: Heavier BoxB should have larger force. A: {fa:.2f}, B: {fb:.2f}"

            assert forces_a[env_idx, 0, 2].item() > 0.1, f"Env {env_idx}: BoxA Z force should be positive"
            assert forces_b[env_idx, 0, 2].item() > 0.1, f"Env {env_idx}: BoxB Z force should be positive"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_mujoco_contacts", COLLISION_PIPELINES)
def test_higher_drop_produces_larger_impact_force(device: str, use_mujoco_contacts: bool):
    """Test that dropping from higher produces larger peak impact force.

    8 environments with heights from 0.3m to 3.0m.

    Verifies:
    - Peak impact force generally increases with height
    - Overall trend: highest/lowest force ratio > 1.5
    """
    num_envs = 8
    min_height, max_height = 0.3, 3.0
    drop_heights = [min_height + (max_height - min_height) * i / (num_envs - 1) for i in range(num_envs)]
    gravity_mag = 9.81
    object_radius = 0.25

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

        root_pose = wp.to_torch(obj.data.root_link_pose_w).clone()
        for env_idx in range(num_envs):
            root_pose[env_idx, 2] = drop_heights[env_idx] + object_radius
        obj.write_root_pose_to_sim_index(root_pose=root_pose)

        total_steps = int(((2 * max_height / gravity_mag) ** 0.5 + 0.5) / SIM_DT)
        peak_forces = [0.0] * num_envs
        contact_detected = [False] * num_envs

        for _ in range(total_steps):
            perform_sim_step(sim, scene, SIM_DT)
            force_magnitudes = torch.norm(wp.to_torch(contact_sensor.data.net_forces_w), dim=-1)
            for env_idx in range(num_envs):
                f = force_magnitudes[env_idx].max().item()
                if f > 0.1:
                    contact_detected[env_idx] = True
                peak_forces[env_idx] = max(peak_forces[env_idx], f)

        for env_idx in range(num_envs):
            assert contact_detected[env_idx], f"Env {env_idx} (h={drop_heights[env_idx]:.2f}m): No contact"

        violations = []
        for i in range(num_envs - 1):
            if peak_forces[i + 1] < peak_forces[i] * 0.95:
                violations.append(
                    f"Env {i} (h={drop_heights[i]:.2f}m, F={peak_forces[i]:.2f}N) -> "
                    f"Env {i + 1} (h={drop_heights[i + 1]:.2f}m, F={peak_forces[i + 1]:.2f}N)"
                )
        assert len(violations) <= 2, "Peak force should increase with height. Violations:\n" + "\n".join(violations)

        force_ratio = peak_forces[-1] / peak_forces[0] if peak_forces[0] > 0 else 0
        assert force_ratio > 1.5, f"Force ratio (highest/lowest) should be > 1.5. Got {force_ratio:.2f}"


# ===================================================================
# Priority 3: Filtering
# ===================================================================


@flaky(max_runs=4, min_passes=1)
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize(
    "use_mujoco_contacts",
    [
        pytest.param(False, id="newton_contacts"),
        pytest.param(True, id="mujoco_contacts"),
    ],
)
def test_filter_enables_force_matrix(device: str, use_mujoco_contacts: bool):
    """Test that filter_prim_paths_expr filters contacts and enables force_matrix_w.

    Object A rests on ground, Object B stacked on A.
    Sensor on A is filtered for B only (not ground).

    Verifies:
    - force_matrix_w reports only filtered contact (A-B)
    - net_forces_w reports total contact (ground + B)
    - force_matrix < net_forces (ground contact excluded from matrix)
    """
    settle_steps = 180
    num_envs = 4
    mass_b = 2.0
    gravity = 9.81
    expected_force_from_b = mass_b * gravity

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, -gravity))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = ContactSensorTestSceneCfg(num_envs=num_envs, env_spacing=5.0, lazy_sensor_update=False)

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

        for _ in range(settle_steps):
            perform_sim_step(sim, scene, SIM_DT)

        force_matrix_raw = contact_sensor.data.force_matrix_w
        net_forces_raw = contact_sensor.data.net_forces_w

        assert force_matrix_raw is not None, "force_matrix_w should not be None when filter is set"

        force_matrix = wp.to_torch(force_matrix_raw)
        net_forces = wp.to_torch(net_forces_raw)

        for env_idx in range(num_envs):
            matrix_force = torch.norm(force_matrix[env_idx]).item()
            net_force = torch.norm(net_forces[env_idx]).item()

            tolerance = 0.3 * expected_force_from_b
            assert abs(matrix_force - expected_force_from_b) < tolerance, (
                f"Env {env_idx}: force_matrix should be ~{expected_force_from_b:.2f} N. Got: {matrix_force:.2f} N"
            )
            assert matrix_force < net_force, (
                f"Env {env_idx}: force_matrix (B only) should be < net_forces (all). "
                f"Matrix: {matrix_force:.2f} N, Net: {net_force:.2f} N"
            )


# ===================================================================
# Priority 4: Articulated System
# ===================================================================

ALLEGRO_FINGERTIP_OFFSETS = {
    "index": (-0.052, -0.252, 0.052),
    "middle": (-0.001, -0.252, 0.052),
    "ring": (0.054, -0.252, 0.052),
    "thumb": (-0.168, -0.039, 0.080),
}

ALLEGRO_FINGER_LINKS = {
    "index": "index_link_3",
    "middle": "middle_link_3",
    "ring": "ring_link_3",
    "thumb": "thumb_link_3",
}


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize(
    "use_mujoco_contacts",
    [
        pytest.param(
            False,
            id="newton_contacts",
            marks=pytest.mark.xfail(
                reason="Newton contact pipeline reports inaccurate per-finger forces in articulated systems"
            ),
        ),
        pytest.param(True, id="mujoco_contacts"),
    ],
)
@pytest.mark.parametrize(
    "drop_shape",
    [
        pytest.param(ShapeType.SPHERE, id="sphere"),
        pytest.param(ShapeType.MESH_BOX, id="mesh_box"),
    ],
)
def test_finger_contact_sensor_isolation(device: str, use_mujoco_contacts: bool, drop_shape: ShapeType):
    """Test contact sensor on Allegro hand fingers detects localized contacts.

    Uses 4 environments, each dropping a ball on a different finger
    (env 0 -> index, env 1 -> middle, env 2 -> ring, env 3 -> thumb).
    Verifies that in each env the target finger has the highest peak force.
    """
    drop_steps = 120
    num_envs = 4
    hand_pos = (0.0, 0.0, 0.5)
    finger_names = ["index", "middle", "ring", "thumb"]

    sim_cfg = make_sim_cfg(use_mujoco_contacts=use_mujoco_contacts, device=device, gravity=(0.0, 0.0, 0.0))

    with build_simulation_context(sim_cfg=sim_cfg, add_ground_plane=True, add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = ContactSensorTestSceneCfg(num_envs=num_envs, env_spacing=1.0, lazy_sensor_update=False)

        scene_cfg.hand = ALLEGRO_HAND_CFG.copy()
        scene_cfg.hand.prim_path = "{ENV_REGEX_NS}/Hand"
        scene_cfg.hand.init_state.pos = hand_pos

        for finger in finger_names:
            setattr(
                scene_cfg,
                f"contact_sensor_{finger}",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Hand/{ALLEGRO_FINGER_LINKS[finger]}",
                    update_period=0.0,
                    history_length=1,
                ),
            )

        drop_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, linear_damping=0.0, angular_damping=0.0
        )
        drop_collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        drop_mass_props = sim_utils.MassPropertiesCfg(mass=1.0)
        drop_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))

        spawn_map = {
            ShapeType.SPHERE: lambda: sim_utils.SphereCfg(
                radius=0.035,
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
                activate_contact_sensors=True,
            ),
            ShapeType.MESH_BOX: lambda: sim_utils.MeshCuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=drop_rigid_props,
                collision_props=drop_collision_props,
                mass_props=drop_mass_props,
                visual_material=drop_visual,
                activate_contact_sensors=True,
            ),
        }

        default_offset = ALLEGRO_FINGERTIP_OFFSETS["index"]
        default_drop_pos = (
            hand_pos[0] + default_offset[0],
            hand_pos[1] + default_offset[1],
            hand_pos[2] + default_offset[2] + 0.10,
        )

        scene_cfg.drop_object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/DropObject",
            spawn=spawn_map[drop_shape](),
            init_state=RigidObjectCfg.InitialStateCfg(pos=default_drop_pos),
        )

        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        hand: Articulation = scene["hand"]
        drop_object: RigidObject = scene["drop_object"]
        finger_sensors = {f: scene[f"contact_sensor_{f}"] for f in finger_names}

        # Newton's articulation.reset() doesn't write default_joint_pos to sim (unlike
        # ManagerBasedEnv's reset_scene_to_default event). Without this, joints start at 0.0
        # which is below thumb_joint_0's lower limit (0.279 rad), causing violent oscillation.
        default_jpos = wp.to_torch(hand.data.default_joint_pos).clone()
        default_jvel = wp.to_torch(hand.data.default_joint_vel).clone()
        hand.write_joint_position_to_sim_index(position=default_jpos)
        hand.write_joint_velocity_to_sim_index(velocity=default_jvel)
        hand.set_joint_position_target_index(target=default_jpos)

        hand_world_pos = wp.to_torch(hand.data.root_link_pose_w)[:, :3]
        drop_pose = wp.to_torch(drop_object.data.root_link_pose_w).clone()
        for env_idx, finger in enumerate(finger_names):
            offset = ALLEGRO_FINGERTIP_OFFSETS[finger]
            drop_pose[env_idx, 0] = hand_world_pos[env_idx, 0] + offset[0]
            drop_pose[env_idx, 1] = hand_world_pos[env_idx, 1] + offset[1]
            drop_pose[env_idx, 2] = hand_world_pos[env_idx, 2] + offset[2] + 0.10
        drop_object.write_root_pose_to_sim_index(root_pose=drop_pose)

        for _ in range(30):
            perform_sim_step(sim, scene, SIM_DT)

        hand_world_pos = wp.to_torch(hand.data.root_link_pose_w)[:, :3]
        drop_object.reset()
        drop_pose = wp.to_torch(drop_object.data.root_link_pose_w).clone()
        for env_idx, finger in enumerate(finger_names):
            offset = ALLEGRO_FINGERTIP_OFFSETS[finger]
            drop_pose[env_idx, 0] = hand_world_pos[env_idx, 0] + offset[0]
            drop_pose[env_idx, 1] = hand_world_pos[env_idx, 1] + offset[1]
            drop_pose[env_idx, 2] = hand_world_pos[env_idx, 2] + offset[2] + 0.10
        drop_object.write_root_pose_to_sim_index(root_pose=drop_pose)

        initial_velocity = torch.zeros(num_envs, 6, device=device)
        initial_velocity[:, 2] = -1.5
        drop_object.write_root_velocity_to_sim_index(root_velocity=initial_velocity)

        peak_forces = {f: [0.0] * num_envs for f in finger_names}

        for _ in range(drop_steps):
            perform_sim_step(sim, scene, SIM_DT)
            for finger_name, sensor in finger_sensors.items():
                if sensor.data.net_forces_w is not None:
                    forces = wp.to_torch(sensor.data.net_forces_w)
                    for env_idx in range(num_envs):
                        f = torch.norm(forces[env_idx]).item()
                        peak_forces[finger_name][env_idx] = max(peak_forces[finger_name][env_idx], f)

        for env_idx, target_finger in enumerate(finger_names):
            target_peak = peak_forces[target_finger][env_idx]
            assert target_peak > 0.5, (
                f"Env {env_idx}: Target finger '{target_finger}' peak force: {target_peak:.4f} N (expected > 0.5)"
            )

            for other_finger in finger_names:
                if other_finger != target_finger:
                    assert target_peak >= peak_forces[other_finger][env_idx], (
                        f"Env {env_idx}: '{target_finger}' (peak={target_peak:.4f}N) should have "
                        f"highest force, but '{other_finger}' had "
                        f"{peak_forces[other_finger][env_idx]:.4f}N"
                    )


# ===================================================================
# Utility
# ===================================================================


def test_sensor_print():
    """Test that contact sensor print/repr works correctly."""
    sim_cfg = make_sim_cfg(use_mujoco_contacts=False, device="cuda:0", gravity=(0.0, 0.0, -9.81))

    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        scene_cfg = ContactSensorTestSceneCfg(num_envs=4, env_spacing=5.0, lazy_sensor_update=False)
        scene_cfg.object_a = create_shape_cfg(
            ShapeType.BOX,
            "{ENV_REGEX_NS}/Object",
            pos=(0.0, 0.0, 2.0),
            disable_gravity=False,
            activate_contact_sensors=True,
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

        sensor_str = str(scene["contact_sensor_a"])
        assert len(sensor_str) > 0
        print(sensor_str)
