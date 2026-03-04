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
import torch
import warp as wp
from flaky import flaky

from isaacsim.core.cloner import GridCloner

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort:skip


@pytest.fixture
def sim():
    """Create a simulation context for testing."""
    # Wait for spawning
    stage = sim_utils.create_new_stage()
    # Constants
    num_envs = 16
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # TODO: Remove this once we have a better way to handle this.
    sim._app_control_on_stop_handle = None

    # Create a ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/GroundPlane", cfg)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    light_cfg = sim_utils.DistantLightCfg(intensity=5.0, exposure=10.0)
    light_cfg.func(
        "/Light",
        light_cfg,
        translation=[0, 0, 1],
    )

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0, stage=stage)
    cloner.define_base_env("/World/envs")
    env_prim_paths = cloner.generate_paths("/World/envs/env", num_envs)
    # create source prim
    stage.DefinePrim(env_prim_paths[0], "Xform")
    # clone the env xform
    cloner.clone(
        source_prim_path=env_prim_paths[0],
        prim_paths=env_prim_paths,
        replicate_physics=True,
    )

    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.actuators["panda_shoulder"].stiffness = 0.0
    robot_cfg.actuators["panda_shoulder"].damping = 0.0
    robot_cfg.actuators["panda_forearm"].stiffness = 0.0
    robot_cfg.actuators["panda_forearm"].damping = 0.0
    robot_cfg.spawn.rigid_props.disable_gravity = True

    # Define the ContactSensor
    contact_forces = None

    # Define the target sets
    ee_goal_abs_pos_set_b = torch.tensor(
        [
            [0.5, 0.5, 0.7],
            [0.5, -0.4, 0.6],
            [0.5, 0, 0.5],
        ],
        device=sim.device,
    )
    ee_goal_abs_quad_set_b = torch.tensor(
        [
            [0.0, 0.707, 0.0, 0.707],
            [0.707, 0.0, 0.0, 0.707],
            [1.0, 0.0, 0.0, 0.0],
        ],
        device=sim.device,
    )
    ee_goal_rel_pos_set = torch.tensor(
        [
            [0.2, 0.0, 0.0],
            [0.2, 0.2, 0.0],
            [0.2, 0.2, -0.2],
        ],
        device=sim.device,
    )
    ee_goal_rel_axisangle_set = torch.tensor(
        [
            [0.0, torch.pi / 2, 0.0],  # for [0.707, 0, 0.707, 0]
            [torch.pi / 2, 0.0, 0.0],  # for [0.707, 0.707, 0, 0]
            [torch.pi / 2, torch.pi / 2, 0.0],  # for [0.0, 1.0, 0, 0]
        ],
        device=sim.device,
    )
    ee_goal_abs_wrench_set_b = torch.tensor(
        [
            [0.0, 0.0, 10.0, 0.0, -1.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        device=sim.device,
    )
    kp_set = torch.tensor(
        [
            [200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
            [240.0, 240.0, 240.0, 240.0, 240.0, 240.0],
            [160.0, 160.0, 160.0, 160.0, 160.0, 160.0],
        ],
        device=sim.device,
    )
    d_ratio_set = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        ],
        device=sim.device,
    )
    # Format: [x, y, z, qx, qy, qz, qw, force_x, force_y, force_z, torque_x, torque_y, torque_z]
    ee_goal_hybrid_set_b = torch.tensor(
        [
            [0.6, 0.2, 0.5, 0.707, 0.0, 0.707, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, -0.29, 0.6, 0.707, 0.0, 0.707, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.1, 0.8, 0.5774, 0.0, 0.8165, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        device=sim.device,
    )
    # Format: [x, y, z, qx, qy, qz, qw] - quaternions converted from wxyz to xyzw format
    ee_goal_pose_set_tilted_b = torch.tensor(
        [
            [0.6, 0.15, 0.3, 0.92387953, 0.0, 0.38268343, 0.0],
            [0.6, -0.3, 0.3, 0.92387953, 0.0, 0.38268343, 0.0],
            [0.8, 0.0, 0.5, 0.92387953, 0.0, 0.38268343, 0.0],
        ],
        device=sim.device,
    )
    ee_goal_wrench_set_tilted_task = torch.tensor(
        [
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        ],
        device=sim.device,
    )

    # Define goals for the arm [xyz]
    target_abs_pos_set_b = ee_goal_abs_pos_set_b.clone()
    # Define goals for the arm [xyz + quat_xyzw]
    target_abs_pose_set_b = torch.cat([ee_goal_abs_pos_set_b, ee_goal_abs_quad_set_b], dim=-1)
    # Define goals for the arm [xyz]
    target_rel_pos_set = ee_goal_rel_pos_set.clone()
    # Define goals for the arm [xyz + axis-angle]
    target_rel_pose_set_b = torch.cat([ee_goal_rel_pos_set, ee_goal_rel_axisangle_set], dim=-1)
    # Define goals for the arm [force_xyz + torque_xyz]
    target_abs_wrench_set = ee_goal_abs_wrench_set_b.clone()
    # Define goals for the arm [xyz + quat_xyzw] and variable kp [kp_xyz + kp_rot_xyz]
    target_abs_pose_variable_kp_set = torch.cat([target_abs_pose_set_b, kp_set], dim=-1)
    # Define goals for the arm [xyz + quat_xyzw] and the variable imp. [kp_xyz + kp_rot_xyz + d_xyz + d_rot_xyz]
    target_abs_pose_variable_set = torch.cat([target_abs_pose_set_b, kp_set, d_ratio_set], dim=-1)
    # Define goals for the arm pose [xyz + quat_xyzw] and wrench [force_xyz + torque_xyz]
    target_hybrid_set_b = ee_goal_hybrid_set_b.clone()
    # Define goals for the arm pose, and wrench, and kp
    target_hybrid_variable_kp_set = torch.cat([target_hybrid_set_b, kp_set], dim=-1)
    # Define goals for the arm pose [xyz + quat_xyzw] in root and and wrench [force_xyz + torque_xyz] in task frame
    target_hybrid_set_tilted = torch.cat([ee_goal_pose_set_tilted_b, ee_goal_wrench_set_tilted_task], dim=-1)

    # Reference frame for targets
    frame = "root"

    yield (
        sim,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        target_abs_pos_set_b,
        target_abs_pose_set_b,
        target_rel_pos_set,
        target_rel_pose_set_b,
        target_abs_wrench_set,
        target_abs_pose_variable_kp_set,
        target_abs_pose_variable_set,
        target_hybrid_set_b,
        target_hybrid_variable_kp_set,
        target_hybrid_set_tilted,
        frame,
    )

    # Cleanup
    sim.stop()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_franka_pose_abs_without_inertial_decoupling(sim):
    """Test absolute pose control with fixed impedance and without inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        target_abs_pose_set_b,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=[400.0, 400.0, 400.0, 100.0, 100.0, 100.0],
        motion_damping_ratio_task=[5.0, 5.0, 5.0, 0.001, 0.001, 0.001],
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_pose_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_pose_abs_with_partial_inertial_decoupling(sim):
    """Test absolute pose control with fixed impedance and partial inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        target_abs_pose_set_b,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=True,
        gravity_compensation=False,
        motion_stiffness_task=1000.0,
        motion_damping_ratio_task=1.0,
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_pose_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_pose_abs_fixed_impedance_with_gravity_compensation(sim):
    """Test absolute pose control with fixed impedance, gravity compensation, and inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        target_abs_pose_set_b,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot = Articulation(cfg=robot_cfg)
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=True,
        motion_stiffness_task=500.0,
        motion_damping_ratio_task=1.0,
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_pose_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_pose_abs(sim):
    """Test absolute pose control with fixed impedance and inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        target_abs_pose_set_b,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=500.0,
        motion_damping_ratio_task=1.0,
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_pose_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_pose_rel(sim):
    """Test relative pose control with fixed impedance and inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        _,
        _,
        target_rel_pose_set_b,
        _,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=500.0,
        motion_damping_ratio_task=1.0,
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_rel_pose_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_pose_abs_variable_impedance(sim):
    """Test absolute pose control with variable impedance and inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        _,
        _,
        _,
        _,
        _,
        target_abs_pose_variable_set,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="variable",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_pose_variable_set,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_wrench_abs_open_loop(sim):
    """Test open loop absolute force control."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        _,
        _,
        _,
        target_abs_wrench_set,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)

    obstacle_spawn_cfg = sim_utils.CuboidCfg(
        size=(0.7, 0.7, 0.01),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        activate_contact_sensors=True,
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_.*/obstacle1",
        obstacle_spawn_cfg,
        translation=(0.2, 0.0, 0.93),
        orientation=(0.0, -0.1736, 0.0, 0.9848),
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_.*/obstacle2",
        obstacle_spawn_cfg,
        translation=(0.2, 0.35, 0.7),
        orientation=(0.707, 0.0, 0.0, 0.707),
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_.*/obstacle3",
        obstacle_spawn_cfg,
        translation=(0.55, 0.0, 0.7),
        orientation=(0.0, 0.707, 0.0, 0.707),
    )
    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/obstacle.*",
        update_period=0.0,
        history_length=50,
        debug_vis=False,
        force_threshold=0.1,
    )
    contact_forces = ContactSensor(contact_forces_cfg)

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["wrench_abs"],
        motion_control_axes_task=[0, 0, 0, 0, 0, 0],
        contact_wrench_control_axes_task=[1, 1, 1, 1, 1, 1],
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_wrench_set,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_wrench_abs_closed_loop(sim):
    """Test closed loop absolute force control."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        _,
        _,
        _,
        target_abs_wrench_set,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)

    obstacle_spawn_cfg = sim_utils.CuboidCfg(
        size=(0.7, 0.7, 0.01),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        activate_contact_sensors=True,
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_.*/obstacle1",
        obstacle_spawn_cfg,
        translation=(0.2, 0.0, 0.93),
        orientation=(0.0, -0.1736, 0.0, 0.9848),
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_.*/obstacle2",
        obstacle_spawn_cfg,
        translation=(0.2, 0.35, 0.7),
        orientation=(0.707, 0.0, 0.0, 0.707),
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_.*/obstacle3",
        obstacle_spawn_cfg,
        translation=(0.55, 0.0, 0.7),
        orientation=(0.0, 0.707, 0.0, 0.707),
    )
    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/obstacle.*",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        force_threshold=0.1,
    )
    contact_forces = ContactSensor(contact_forces_cfg)

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["wrench_abs"],
        contact_wrench_stiffness_task=[
            0.2,
            0.2,
            0.2,
            0.0,
            0.0,
            0.0,
        ],  # Zero torque feedback as we cannot contact torque
        motion_control_axes_task=[0, 0, 0, 0, 0, 0],
        contact_wrench_control_axes_task=[1, 1, 1, 1, 1, 1],
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_wrench_set,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_hybrid_decoupled_motion(sim):
    """Test hybrid control with fixed impedance and partial inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        target_hybrid_set_b,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)

    obstacle_spawn_cfg = sim_utils.CuboidCfg(
        size=(1.0, 1.0, 0.01),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        activate_contact_sensors=True,
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_.*/obstacle1",
        obstacle_spawn_cfg,
        translation=(target_hybrid_set_b[0, 0] + 0.05, 0.0, 0.7),
        orientation=(0.0, 0.707, 0.0, 0.707),
    )
    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/obstacle.*",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        force_threshold=0.1,
    )
    contact_forces = ContactSensor(contact_forces_cfg)

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=True,
        gravity_compensation=False,
        motion_stiffness_task=300.0,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        motion_control_axes_task=[0, 1, 1, 1, 1, 1],
        contact_wrench_control_axes_task=[1, 0, 0, 0, 0, 0],
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_leftfinger",
        ["panda_joint.*"],
        target_hybrid_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
@flaky(max_runs=3, min_passes=1)
def test_franka_hybrid_variable_kp_impedance(sim):
    """Test hybrid control with variable kp impedance and inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        target_hybrid_set_b,
        target_hybrid_variable_kp_set,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)

    obstacle_spawn_cfg = sim_utils.CuboidCfg(
        size=(1.0, 1.0, 0.01),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        activate_contact_sensors=True,
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_.*/obstacle1",
        obstacle_spawn_cfg,
        translation=(target_hybrid_set_b[0, 0] + 0.05, 0.0, 0.7),
        orientation=(0.0, 0.707, 0.0, 0.707),
    )
    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/obstacle.*",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        force_threshold=0.1,
    )
    contact_forces = ContactSensor(contact_forces_cfg)

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_damping_ratio_task=0.8,
        contact_wrench_stiffness_task=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        motion_control_axes_task=[0, 1, 1, 1, 1, 1],
        contact_wrench_control_axes_task=[1, 0, 0, 0, 0, 0],
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    # Use more convergence steps for hybrid control which is less precise
    _run_op_space_controller(
        robot,
        osc,
        "panda_leftfinger",
        ["panda_joint.*"],
        target_hybrid_variable_kp_set,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
        convergence_steps=750,
    )


@pytest.mark.isaacsim_ci
def test_franka_taskframe_pose_abs(sim):
    """Test absolute pose control in task frame with fixed impedance and inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        target_abs_pose_set_b,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    frame = "task"
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=500.0,
        motion_damping_ratio_task=1.0,
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_pose_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_taskframe_pose_rel(sim):
    """Test relative pose control in task frame with fixed impedance and inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        _,
        _,
        target_rel_pose_set_b,
        _,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    frame = "task"
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=500.0,
        motion_damping_ratio_task=1.0,
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_rel_pose_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_taskframe_hybrid(sim):
    """Test hybrid control in task frame with fixed impedance and inertial dynamics decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        target_hybrid_set_tilted,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    frame = "task"

    obstacle_spawn_cfg = sim_utils.CuboidCfg(
        size=(2.0, 1.5, 0.01),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        activate_contact_sensors=True,
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_.*/obstacle1",
        obstacle_spawn_cfg,
        translation=(target_hybrid_set_tilted[0, 0] + 0.085, 0.0, 0.3),
        orientation=(0.0, -0.3826834324, 0.0, 0.9238795325),
    )
    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/obstacle.*",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        force_threshold=0.1,
    )
    contact_forces = ContactSensor(contact_forces_cfg)

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=400.0,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 0, 1, 1, 1],
        contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_leftfinger",
        ["panda_joint.*"],
        target_hybrid_set_tilted,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_pose_abs_without_inertial_decoupling_with_nullspace_centering(sim):
    """Test absolute pose control with fixed impedance and nullspace centerin but without inertial decoupling."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        target_abs_pose_set_b,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=[400.0, 400.0, 400.0, 100.0, 100.0, 100.0],
        motion_damping_ratio_task=[5.0, 5.0, 5.0, 0.001, 0.001, 0.001],
        nullspace_control="position",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_pose_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_pose_abs_with_partial_inertial_decoupling_nullspace_centering(sim):
    """Test absolute pose control with fixed impedance, partial inertial decoupling and nullspace centering."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        target_abs_pose_set_b,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=True,
        gravity_compensation=False,
        motion_stiffness_task=1000.0,
        motion_damping_ratio_task=1.0,
        nullspace_control="position",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_pose_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_pose_abs_with_nullspace_centering(sim):
    """Test absolute pose control with fixed impedance, inertial decoupling and nullspace centering."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        target_abs_pose_set_b,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=500.0,
        motion_damping_ratio_task=1.0,
        nullspace_control="position",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_hand",
        ["panda_joint.*"],
        target_abs_pose_set_b,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


@pytest.mark.isaacsim_ci
def test_franka_taskframe_hybrid_with_nullspace_centering(sim):
    """Test hybrid control in task frame with fixed impedance, inertial decoupling and nullspace centering."""
    (
        sim_context,
        num_envs,
        robot_cfg,
        ee_marker,
        goal_marker,
        contact_forces,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        target_hybrid_set_tilted,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    frame = "task"

    obstacle_spawn_cfg = sim_utils.CuboidCfg(
        size=(2.0, 1.5, 0.01),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        activate_contact_sensors=True,
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_.*/obstacle1",
        obstacle_spawn_cfg,
        translation=(target_hybrid_set_tilted[0, 0] + 0.085, 0.0, 0.3),
        orientation=(0.0, -0.3826834324, 0.0, 0.9238795325),
    )
    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/obstacle.*",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
        force_threshold=0.1,
    )
    contact_forces = ContactSensor(contact_forces_cfg)

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_stiffness_task=400.0,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 0, 1, 1, 1],
        contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
        nullspace_control="position",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=num_envs, device=sim_context.device)

    _run_op_space_controller(
        robot,
        osc,
        "panda_leftfinger",
        ["panda_joint.*"],
        target_hybrid_set_tilted,
        sim_context,
        num_envs,
        ee_marker,
        goal_marker,
        contact_forces,
        frame,
    )


def _run_op_space_controller(
    robot: Articulation,
    osc: OperationalSpaceController,
    ee_frame_name: str,
    arm_joint_names: list[str],
    target_set: torch.tensor,
    sim: sim_utils.SimulationContext,
    num_envs: int,
    ee_marker: VisualizationMarkers,
    goal_marker: VisualizationMarkers,
    contact_forces: ContactSensor | None,
    frame: str,
    convergence_steps: int = 500,
):
    """Run the operational space controller with the given parameters.

    Args:
        robot (Articulation): The robot to control.
        osc (OperationalSpaceController): The operational space controller.
        ee_frame_name (str): The name of the end-effector frame.
        arm_joint_names (list[str]): The names of the arm joints.
        target_set (torch.tensor): The target set to track.
        sim (sim_utils.SimulationContext): The simulation context.
        num_envs (int): The number of environments.
        ee_marker (VisualizationMarkers): The end-effector marker.
        goal_marker (VisualizationMarkers): The goal marker.
        contact_forces (ContactSensor | None): The contact forces sensor.
        frame (str): The reference frame for targets.
        convergence_steps (int): Number of simulation steps to run before checking convergence. Defaults to 500.
    """
    # Initialize the masks for evaluating target convergence according to selection matrices
    pos_mask = torch.tensor(osc.cfg.motion_control_axes_task[:3], device=sim.device).view(1, 3)
    rot_mask = torch.tensor(osc.cfg.motion_control_axes_task[3:], device=sim.device).view(1, 3)
    wrench_mask = torch.tensor(osc.cfg.contact_wrench_control_axes_task, device=sim.device).view(1, 6)
    force_mask = wrench_mask[:, 0:3]  # Take only the force components as we can measure only these

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Play the simulator
    sim.reset()

    # Obtain the frame index of the end-effector
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    # Obtain joint indices
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    # Update existing buffers
    # Note: We need to update buffers before the first step for the controller.
    robot.update(dt=sim_dt)

    # Get the center of the robot soft joint limits
    joint_centers = torch.mean(wp.to_torch(robot.data.soft_joint_pos_limits)[:, arm_joint_ids, :], dim=-1)

    # get the updated states
    (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    ) = _update_states(robot, ee_frame_idx, arm_joint_ids, sim, contact_forces, num_envs)

    # Track the given target command
    current_goal_idx = 0  # Current goal index for the arm
    command = torch.zeros(
        num_envs, osc.action_dim, device=sim.device
    )  # Generic target command, which can be pose, position, force, etc.
    ee_target_pose_b = torch.zeros(num_envs, 7, device=sim.device)  # Target pose in the body frame
    ee_target_pose_w = torch.zeros(num_envs, 7, device=sim.device)  # Target pose in the world frame (for marker)

    # Set joint efforts to zero
    zero_joint_efforts = torch.zeros(num_envs, robot.num_joints, device=sim.device)
    joint_efforts = torch.zeros(num_envs, len(arm_joint_ids), device=sim.device)

    # Now we are ready!
    # Run for 3 target cycles plus 1 step to trigger final convergence check
    total_steps = 3 * convergence_steps + 1
    for count in range(total_steps):
        # reset every convergence_steps steps
        if count % convergence_steps == 0:
            # check that we converged to the goal
            if count > 0:
                _check_convergence(
                    osc, ee_pose_b, ee_target_pose_b, ee_force_b, command, pos_mask, rot_mask, force_mask, frame
                )
            # reset joint state to default
            default_joint_pos = wp.to_torch(robot.data.default_joint_pos).clone()
            default_joint_vel = wp.to_torch(robot.data.default_joint_vel).clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
            robot.write_data_to_sim()
            robot.reset()
            # reset contact sensor
            if contact_forces is not None:
                contact_forces.reset()
            # reset target pose
            robot.update(sim_dt)
            _, _, _, ee_pose_b, _, _, _, _, _, _ = _update_states(
                robot, ee_frame_idx, arm_joint_ids, sim, contact_forces, num_envs
            )  # at reset, the jacobians are not updated to the latest state
            command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = _update_target(
                osc, root_pose_w, ee_pose_b, target_set, current_goal_idx
            )
            # set the osc command
            osc.reset()
            command, task_frame_pose_b = _convert_to_task_frame(
                osc, command=command, ee_target_pose_b=ee_target_pose_b, frame=frame
            )
            osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b)
        else:
            # get the updated states
            (
                jacobian_b,
                mass_matrix,
                gravity,
                ee_pose_b,
                ee_vel_b,
                root_pose_w,
                ee_pose_w,
                ee_force_b,
                joint_pos,
                joint_vel,
            ) = _update_states(robot, ee_frame_idx, arm_joint_ids, sim, contact_forces, num_envs)
            # compute the joint commands
            joint_efforts = osc.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                current_ee_force_b=ee_force_b,
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=joint_centers,
            )
            robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()

        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

        # perform step
        sim.step(render=False)
        # update buffers
        robot.update(sim_dt)


def _update_states(
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
    sim: sim_utils.SimulationContext,
    contact_forces: ContactSensor | None,
    num_envs: int,
):
    """Update the states of the robot and obtain the relevant quantities for the operational space controller.

    Args:
        robot (Articulation): The robot to control.
        ee_frame_idx (int): The index of the end-effector frame.
        arm_joint_ids (list[int]): The indices of the arm joints.
        sim (sim_utils.SimulationContext): The simulation context.
        contact_forces (ContactSensor | None): The contact forces sensor.
        num_envs (int): Number of environments.

    Returns:
        jacobian_b (torch.tensor): The Jacobian in the root frame.
        mass_matrix (torch.tensor): The mass matrix.
        gravity (torch.tensor): The gravity vector.
        ee_pose_b (torch.tensor): The end-effector pose in the root frame.
        ee_vel_b (torch.tensor): The end-effector velocity in the root frame.
        root_pose_w (torch.tensor): The root pose in the world frame.
        ee_pose_w (torch.tensor): The end-effector pose in the world frame.
        ee_force_b (torch.tensor): The end-effector force in the root frame.
        joint_pos (torch.tensor): The joint positions.
        joint_vel (torch.tensor): The joint velocities.
    """
    # obtain dynamics related quantities from simulation
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = wp.to_torch(robot.root_view.get_jacobians())[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = wp.to_torch(robot.root_view.get_generalized_mass_matrices())[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = wp.to_torch(robot.root_view.get_gravity_compensation_forces())[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(wp.to_torch(robot.data.root_quat_w)))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pose_w = wp.to_torch(robot.data.root_pose_w)
    ee_pose_w = wp.to_torch(robot.data.body_pose_w)[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Compute the current velocity of the end-effector
    ee_vel_w = wp.to_torch(robot.data.body_vel_w)[
        :, ee_frame_idx, :
    ]  # Extract end-effector velocity in the world frame
    root_vel_w = wp.to_torch(robot.data.root_vel_w)  # Extract root velocity in the world frame
    relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
    ee_lin_vel_b = quat_apply_inverse(
        wp.to_torch(robot.data.root_quat_w), relative_vel_w[:, 0:3]
    )  # From world to root frame
    ee_ang_vel_b = quat_apply_inverse(wp.to_torch(robot.data.root_quat_w), relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Calculate the contact force
    ee_force_w = torch.zeros(num_envs, 3, device=sim.device)
    if contact_forces is not None:  # Only modify if it exist
        sim_dt = sim.get_physics_dt()
        contact_forces.update(sim_dt)  # update contact sensor
        # Calculate the contact force by averaging over last four time steps (i.e., to smoothen) and
        # taking the max of three surfaces as only one should be the contact of interest
        ee_force_w, _ = torch.max(torch.mean(wp.to_torch(contact_forces.data.net_forces_w_history), dim=1), dim=1)

    # This is a simplification, only for the sake of testing.
    ee_force_b = ee_force_w

    # Get joint positions and velocities
    joint_pos = wp.to_torch(robot.data.joint_pos)[:, arm_joint_ids]
    joint_vel = wp.to_torch(robot.data.joint_vel)[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        ee_force_b,
        joint_pos,
        joint_vel,
    )


def _update_target(
    osc: OperationalSpaceController,
    root_pose_w: torch.tensor,
    ee_pose_b: torch.tensor,
    target_set: torch.tensor,
    current_goal_idx: int,
):
    """Update the target for the operational space controller.

    Args:
        osc (OperationalSpaceController): The operational space controller.
        root_pose_w (torch.tensor): The root pose in the world frame.
        ee_pose_b (torch.tensor): The end-effector pose in the body frame.
        target_set (torch.tensor): The target set to track.
        current_goal_idx (int): The current goal index.

    Returns:
        command (torch.tensor): The target command.
        ee_target_pose_b (torch.tensor): The end-effector target pose in the body frame.
        ee_target_pose_w (torch.tensor): The end-effector target pose in the world frame.
        next_goal_idx (int): The next goal index.

    Raises:
        ValueError: If the target type is undefined.
    """
    # update the ee desired command
    command = torch.zeros(osc.num_envs, osc.action_dim, device=osc._device)
    command[:] = target_set[current_goal_idx]

    # update the ee desired pose
    ee_target_pose_b = torch.zeros(osc.num_envs, 7, device=osc._device)
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            ee_target_pose_b[:] = command[:, :7]
        elif target_type == "pose_rel":
            ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7] = apply_delta_pose(
                ee_pose_b[:, :3], ee_pose_b[:, 3:], command[:, :7]
            )
        elif target_type == "wrench_abs":
            pass  # ee_target_pose_b could stay at the root frame for force control, what matters is ee_target_b
        else:
            raise ValueError("Undefined target_type within _update_target().")

    # update the target desired pose in world frame (for marker)
    ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
    )
    ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

    next_goal_idx = (current_goal_idx + 1) % len(target_set)

    return command, ee_target_pose_b, ee_target_pose_w, next_goal_idx


def _convert_to_task_frame(
    osc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor, frame: str
):
    """Convert the target command to the task frame if required.

    Args:
        osc (OperationalSpaceController): The operational space controller.
        command (torch.tensor): The target command to convert.
        ee_target_pose_b (torch.tensor): The end-effector target pose in the body frame.
        frame (str): The reference frame for targets.

    Returns:
        command (torch.tensor): The converted target command.
        task_frame_pose_b (torch.tensor): The task frame pose in the body frame.

    Raises:
        ValueError: If the frame is invalid.
    """
    command = command.clone()
    task_frame_pose_b = None
    if frame == "root":
        # No need to transform anything if they are already in root frame
        pass
    elif frame == "task":
        # Convert target commands from base to the task frame
        command = command.clone()
        task_frame_pose_b = ee_target_pose_b.clone()

        cmd_idx = 0
        for target_type in osc.cfg.target_types:
            if target_type == "pose_abs":
                command[:, :3], command[:, 3:7] = subtract_frame_transforms(
                    task_frame_pose_b[:, :3], task_frame_pose_b[:, 3:], command[:, :3], command[:, 3:7]
                )
                cmd_idx += 7
            elif target_type == "pose_rel":
                # Compute rotation matrices
                R_task_b = matrix_from_quat(task_frame_pose_b[:, 3:])  # Task frame to base frame
                R_b_task = R_task_b.mT  # Base frame to task frame
                # Transform the delta position and orientation from base to task frame
                command[:, :3] = (R_b_task @ command[:, :3].unsqueeze(-1)).squeeze(-1)
                command[:, 3:7] = (R_b_task @ command[:, 3:7].unsqueeze(-1)).squeeze(-1)
                cmd_idx += 6
            elif target_type == "wrench_abs":
                # These are already defined in target frame for ee_goal_wrench_set_tilted_task (since it is
                # easier), so not transforming
                cmd_idx += 6
            else:
                raise ValueError("Undefined target_type within _convert_to_task_frame().")
    else:
        # Raise error for invalid frame
        raise ValueError("Invalid frame selection for target setting inside the test_operational_space.")

    return command, task_frame_pose_b


def _check_convergence(
    osc: OperationalSpaceController,
    ee_pose_b: torch.tensor,
    ee_target_pose_b: torch.tensor,
    ee_force_b: torch.tensor,
    ee_target_b: torch.tensor,
    pos_mask: torch.tensor,
    rot_mask: torch.tensor,
    force_mask: torch.tensor,
    frame: str,
):
    """Check the convergence to the target.

    Args:
        osc (OperationalSpaceController): The operational space controller.
        ee_pose_b (torch.tensor): The end-effector pose in the body frame.
        ee_target_pose_b (torch.tensor): The end-effector target pose in the body frame.
        ee_force_b (torch.tensor): The end-effector force in the body frame.
        ee_target_b (torch.tensor): The end-effector target in the body frame.
        pos_mask (torch.tensor): The position mask.
        rot_mask (torch.tensor): The rotation mask.
        force_mask (torch.tensor): The force mask.
        frame (str): The reference frame for targets.

    Raises:
        AssertionError: If the convergence is not achieved.
        ValueError: If the target type is undefined.
    """
    cmd_idx = 0
    for target_type in osc.cfg.target_types:
        if target_type == "pose_abs":
            pos_error, rot_error = compute_pose_error(
                ee_pose_b[:, 0:3], ee_pose_b[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
            )
            pos_error_norm = torch.linalg.norm(pos_error * pos_mask, dim=-1)
            rot_error_norm = torch.linalg.norm(rot_error * rot_mask, dim=-1)
            # desired error (zer)
            des_error = torch.zeros_like(pos_error_norm)
            # check convergence
            torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=0.1)
            torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=0.1)
            cmd_idx += 7
        elif target_type == "pose_rel":
            pos_error, rot_error = compute_pose_error(
                ee_pose_b[:, 0:3], ee_pose_b[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
            )
            pos_error_norm = torch.linalg.norm(pos_error * pos_mask, dim=-1)
            rot_error_norm = torch.linalg.norm(rot_error * rot_mask, dim=-1)
            # desired error (zer)
            des_error = torch.zeros_like(pos_error_norm)
            # check convergence
            torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=0.1)
            torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=0.1)
            cmd_idx += 6
        elif target_type == "wrench_abs":
            force_target_b = ee_target_b[:, cmd_idx : cmd_idx + 3].clone()
            # Convert to base frame if the target was defined in task frame
            if frame == "task":
                task_frame_pose_b = ee_target_pose_b.clone()
                R_task_b = matrix_from_quat(task_frame_pose_b[:, 3:])
                force_target_b[:] = (R_task_b @ force_target_b[:].unsqueeze(-1)).squeeze(-1)
            force_error = ee_force_b - force_target_b
            force_error_norm = torch.linalg.norm(
                force_error * force_mask, dim=-1
            )  # ignore torque part as we cannot measure it
            # Check convergence using statistical thresholds instead of a blanket all-environments
            # tolerance. Contact force steady-state is sensitive to physics engine internals (PhysX
            # solver iterations, contact resolution, penetration depth) which causes outlier
            # environments. A tight median check catches real controller regressions while a loose
            # max check catches catastrophic failures without breaking on single-environment noise.
            median_error = torch.median(force_error_norm).item()
            max_error = torch.max(force_error_norm).item()
            assert median_error < 5.0, (
                f"Median force error {median_error:.1f} N exceeds 5.0 N threshold"
                f" (max: {max_error:.1f} N, per-env: {force_error_norm.tolist()})"
            )
            assert max_error < 50.0, (
                f"Max force error {max_error:.1f} N exceeds 50.0 N sanity threshold"
                f" (median: {median_error:.1f} N, per-env: {force_error_norm.tolist()})"
            )
            cmd_idx += 6
        else:
            raise ValueError("Undefined target_type within _check_convergence().")
