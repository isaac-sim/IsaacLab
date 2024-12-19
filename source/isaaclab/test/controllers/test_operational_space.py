# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
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
    quat_inv,
    quat_rotate_inverse,
    subtract_frame_transforms,
)

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort:skip


class TestOperationalSpaceController(unittest.TestCase):
    """Test fixture for checking that Operational Space controller tracks commands properly."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Wait for spawning
        stage_utils.create_new_stage()
        # Constants
        self.num_envs = 16
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=0.01)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        # TODO: Remove this once we have a better way to handle this.
        self.sim._app_control_on_stop_handle = None

        # Create a ground plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/GroundPlane", cfg)

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        light_cfg = sim_utils.DistantLightCfg(intensity=5.0, exposure=10.0)
        light_cfg.func(
            "/Light",
            light_cfg,
            translation=[0, 0, 1],
        )

        # Create interface to clone the scene
        cloner = GridCloner(spacing=2.0)
        cloner.define_base_env("/World/envs")
        self.env_prim_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        # create source prim
        prim_utils.define_prim(self.env_prim_paths[0], "Xform")
        # clone the env xform
        self.env_origins = cloner.clone(
            source_prim_path=self.env_prim_paths[0],
            prim_paths=self.env_prim_paths,
            replicate_physics=True,
        )

        self.robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.robot_cfg.actuators["panda_shoulder"].stiffness = 0.0
        self.robot_cfg.actuators["panda_shoulder"].damping = 0.0
        self.robot_cfg.actuators["panda_forearm"].stiffness = 0.0
        self.robot_cfg.actuators["panda_forearm"].damping = 0.0
        self.robot_cfg.spawn.rigid_props.disable_gravity = True

        # Define the ContactSensor
        self.contact_forces = None

        # Define the target sets
        ee_goal_abs_pos_set_b = torch.tensor(
            [
                [0.5, 0.5, 0.7],
                [0.5, -0.4, 0.6],
                [0.5, 0, 0.5],
            ],
            device=self.sim.device,
        )
        ee_goal_abs_quad_set_b = torch.tensor(
            [
                [0.707, 0.0, 0.707, 0.0],
                [0.707, 0.707, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            device=self.sim.device,
        )
        ee_goal_rel_pos_set = torch.tensor(
            [
                [0.2, 0.0, 0.0],
                [0.2, 0.2, 0.0],
                [0.2, 0.2, -0.2],
            ],
            device=self.sim.device,
        )
        ee_goal_rel_axisangle_set = torch.tensor(
            [
                [0.0, torch.pi / 2, 0.0],  # for [0.707, 0, 0.707, 0]
                [torch.pi / 2, 0.0, 0.0],  # for [0.707, 0.707, 0, 0]
                [torch.pi, 0.0, 0.0],  # for [0.0, 1.0, 0, 0]
            ],
            device=self.sim.device,
        )
        ee_goal_abs_wrench_set_b = torch.tensor(
            [
                [0.0, 0.0, 10.0, 0.0, -1.0, 0.0],
                [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device=self.sim.device,
        )
        kp_set = torch.tensor(
            [
                [200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
                [240.0, 240.0, 240.0, 240.0, 240.0, 240.0],
                [160.0, 160.0, 160.0, 160.0, 160.0, 160.0],
            ],
            device=self.sim.device,
        )
        d_ratio_set = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            ],
            device=self.sim.device,
        )
        ee_goal_hybrid_set_b = torch.tensor(
            [
                [0.6, 0.2, 0.5, 0.0, 0.707, 0.0, 0.707, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.6, -0.29, 0.6, 0.0, 0.707, 0.0, 0.707, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.6, 0.1, 0.8, 0.0, 0.5774, 0.0, 0.8165, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device=self.sim.device,
        )
        ee_goal_pose_set_tilted_b = torch.tensor(
            [
                [0.6, 0.15, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
                [0.6, -0.3, 0.3, 0.0, 0.92387953, 0.0, 0.38268343],
                [0.8, 0.0, 0.5, 0.0, 0.92387953, 0.0, 0.38268343],
            ],
            device=self.sim.device,
        )
        ee_goal_wrench_set_tilted_task = torch.tensor(
            [
                [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            ],
            device=self.sim.device,
        )

        # Define goals for the arm [xyz]
        self.target_abs_pos_set_b = ee_goal_abs_pos_set_b.clone()
        # Define goals for the arm [xyz + quat_wxyz]
        self.target_abs_pose_set_b = torch.cat([ee_goal_abs_pos_set_b, ee_goal_abs_quad_set_b], dim=-1)
        # Define goals for the arm [xyz]
        self.target_rel_pos_set = ee_goal_rel_pos_set.clone()
        # Define goals for the arm [xyz + axis-angle]
        self.target_rel_pose_set_b = torch.cat([ee_goal_rel_pos_set, ee_goal_rel_axisangle_set], dim=-1)
        # Define goals for the arm [force_xyz + torque_xyz]
        self.target_abs_wrench_set = ee_goal_abs_wrench_set_b.clone()
        # Define goals for the arm [xyz + quat_wxyz] and variable kp [kp_xyz + kp_rot_xyz]
        self.target_abs_pose_variable_kp_set = torch.cat([self.target_abs_pose_set_b, kp_set], dim=-1)
        # Define goals for the arm [xyz + quat_wxyz] and the variable imp. [kp_xyz + kp_rot_xyz + d_xyz + d_rot_xyz]
        self.target_abs_pose_variable_set = torch.cat([self.target_abs_pose_set_b, kp_set, d_ratio_set], dim=-1)
        # Define goals for the arm pose [xyz + quat_wxyz] and wrench [force_xyz + torque_xyz]
        self.target_hybrid_set_b = ee_goal_hybrid_set_b.clone()
        # Define goals for the arm pose, and wrench, and kp
        self.target_hybrid_variable_kp_set = torch.cat([self.target_hybrid_set_b, kp_set], dim=-1)
        # Define goals for the arm pose [xyz + quat_wxyz] in root and and wrench [force_xyz + torque_xyz] in task frame
        self.target_hybrid_set_tilted = torch.cat([ee_goal_pose_set_tilted_b, ee_goal_wrench_set_tilted_task], dim=-1)

        # Reference frame for targets
        self.frame = "root"

    def tearDown(self):
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        # self.sim.clear()  # FIXME: This hangs the test for some reason when LIVESTREAM is not enabled.
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
        # Make contact_forces None after relevant tests otherwise other tests give warning
        self.contact_forces = None

    """
    Test fixtures.
    """

    def test_franka_pose_abs_without_inertial_decoupling(self):
        """Test absolute pose control with fixed impedance and without inertial dynamics decoupling."""
        robot = Articulation(cfg=self.robot_cfg)
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=[400.0, 400.0, 400.0, 100.0, 100.0, 100.0],
            motion_damping_ratio_task=[5.0, 5.0, 5.0, 0.001, 0.001, 0.001],
        )
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_pose_set_b)

    def test_franka_pose_abs_with_partial_inertial_decoupling(self):
        """Test absolute pose control with fixed impedance and partial inertial dynamics decoupling."""
        robot = Articulation(cfg=self.robot_cfg)
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=True,
            gravity_compensation=False,
            motion_stiffness_task=1000.0,
            motion_damping_ratio_task=1.0,
        )
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_pose_set_b)

    def test_franka_pose_abs_fixed_impedance_with_gravity_compensation(self):
        """Test absolute pose control with fixed impedance, gravity compensation, and inertial dynamics decoupling."""
        self.robot_cfg.spawn.rigid_props.disable_gravity = False
        robot = Articulation(cfg=self.robot_cfg)
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        )
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_pose_set_b)

    def test_franka_pose_abs(self):
        """Test absolute pose control with fixed impedance and inertial dynamics decoupling."""
        robot = Articulation(cfg=self.robot_cfg)
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        )
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_pose_set_b)

    def test_franka_pose_rel(self):
        """Test relative pose control with fixed impedance and inertial dynamics decoupling."""
        robot = Articulation(cfg=self.robot_cfg)
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        )
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_rel_pose_set_b)

    def test_franka_pose_abs_variable_impedance(self):
        """Test absolute pose control with variable impedance and inertial dynamics decoupling."""
        robot = Articulation(cfg=self.robot_cfg)
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="variable",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
        )
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_pose_variable_set)

    def test_franka_wrench_abs_open_loop(self):
        """Test open loop absolute force control."""
        robot = Articulation(cfg=self.robot_cfg)

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
            orientation=(0.9848, 0.0, -0.1736, 0.0),
        )
        obstacle_spawn_cfg.func(
            "/World/envs/env_.*/obstacle2",
            obstacle_spawn_cfg,
            translation=(0.2, 0.35, 0.7),
            orientation=(0.707, 0.707, 0.0, 0.0),
        )
        obstacle_spawn_cfg.func(
            "/World/envs/env_.*/obstacle3",
            obstacle_spawn_cfg,
            translation=(0.55, 0.0, 0.7),
            orientation=(0.707, 0.0, 0.707, 0.0),
        )
        contact_forces_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/obstacle.*",
            update_period=0.0,
            history_length=50,
            debug_vis=False,
            force_threshold=0.1,
        )
        self.contact_forces = ContactSensor(contact_forces_cfg)

        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["wrench_abs"],
            motion_control_axes_task=[0, 0, 0, 0, 0, 0],
            contact_wrench_control_axes_task=[1, 1, 1, 1, 1, 1],
        )
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_wrench_set)

    def test_franka_wrench_abs_closed_loop(self):
        """Test closed loop absolute force control."""
        robot = Articulation(cfg=self.robot_cfg)

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
            orientation=(0.9848, 0.0, -0.1736, 0.0),
        )
        obstacle_spawn_cfg.func(
            "/World/envs/env_.*/obstacle2",
            obstacle_spawn_cfg,
            translation=(0.2, 0.35, 0.7),
            orientation=(0.707, 0.707, 0.0, 0.0),
        )
        obstacle_spawn_cfg.func(
            "/World/envs/env_.*/obstacle3",
            obstacle_spawn_cfg,
            translation=(0.55, 0.0, 0.7),
            orientation=(0.707, 0.0, 0.707, 0.0),
        )
        contact_forces_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/obstacle.*",
            update_period=0.0,
            history_length=2,
            debug_vis=False,
            force_threshold=0.1,
        )
        self.contact_forces = ContactSensor(contact_forces_cfg)

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
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_wrench_set)

    def test_franka_hybrid_decoupled_motion(self):
        """Test hybrid control with fixed impedance and partial inertial dynamics decoupling."""
        robot = Articulation(cfg=self.robot_cfg)

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
            translation=(self.target_hybrid_set_b[0, 0] + 0.05, 0.0, 0.7),
            orientation=(0.707, 0.0, 0.707, 0.0),
        )
        contact_forces_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/obstacle.*",
            update_period=0.0,
            history_length=2,
            debug_vis=False,
            force_threshold=0.1,
        )
        self.contact_forces = ContactSensor(contact_forces_cfg)

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
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_leftfinger", ["panda_joint.*"], self.target_hybrid_set_b)

    def test_franka_hybrid_variable_kp_impedance(self):
        """Test hybrid control with variable kp impedance and inertial dynamics decoupling."""
        robot = Articulation(cfg=self.robot_cfg)

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
            translation=(self.target_hybrid_set_b[0, 0] + 0.05, 0.0, 0.7),
            orientation=(0.707, 0.0, 0.707, 0.0),
        )
        contact_forces_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/obstacle.*",
            update_period=0.0,
            history_length=2,
            debug_vis=False,
            force_threshold=0.1,
        )
        self.contact_forces = ContactSensor(contact_forces_cfg)

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
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(
            robot, osc, "panda_leftfinger", ["panda_joint.*"], self.target_hybrid_variable_kp_set
        )

    def test_franka_taskframe_pose_abs(self):
        """Test absolute pose control in task frame with fixed impedance and inertial dynamics decoupling."""
        robot = Articulation(cfg=self.robot_cfg)
        self.frame = "task"
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        )
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_pose_set_b)

    def test_franka_taskframe_pose_rel(self):
        """Test relative pose control in task frame with fixed impedance and inertial dynamics decoupling."""
        robot = Articulation(cfg=self.robot_cfg)
        self.frame = "task"
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        )
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_rel_pose_set_b)

    def test_franka_taskframe_hybrid(self):
        """Test hybrid control in task frame with fixed impedance and inertial dynamics decoupling."""
        robot = Articulation(cfg=self.robot_cfg)
        self.frame = "task"

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
            translation=(self.target_hybrid_set_tilted[0, 0] + 0.085, 0.0, 0.3),
            orientation=(0.9238795325, 0.0, -0.3826834324, 0.0),
        )
        contact_forces_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/obstacle.*",
            update_period=0.0,
            history_length=2,
            debug_vis=False,
            force_threshold=0.1,
        )
        self.contact_forces = ContactSensor(contact_forces_cfg)

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
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_leftfinger", ["panda_joint.*"], self.target_hybrid_set_tilted)

    def test_franka_pose_abs_without_inertial_decoupling_with_nullspace_centering(self):
        """Test absolute pose control with fixed impedance and nullspace centerin but without inertial decoupling."""
        robot = Articulation(cfg=self.robot_cfg)
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=[400.0, 400.0, 400.0, 100.0, 100.0, 100.0],
            motion_damping_ratio_task=[5.0, 5.0, 5.0, 0.001, 0.001, 0.001],
            nullspace_control="position",
        )
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_pose_set_b)

    def test_franka_pose_abs_with_partial_inertial_decoupling_nullspace_centering(self):
        """Test absolute pose control with fixed impedance, partial inertial decoupling and nullspace centering."""
        robot = Articulation(cfg=self.robot_cfg)
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
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_pose_set_b)

    def test_franka_pose_abs_with_nullspace_centering(self):
        """Test absolute pose control with fixed impedance, inertial decoupling and nullspace centering."""
        robot = Articulation(cfg=self.robot_cfg)
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
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_hand", ["panda_joint.*"], self.target_abs_pose_set_b)

    def test_franka_taskframe_hybrid_with_nullspace_centering(self):
        """Test hybrid control in task frame with fixed impedance, inertial decoupling and nullspace centering."""
        robot = Articulation(cfg=self.robot_cfg)
        self.frame = "task"

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
            translation=(self.target_hybrid_set_tilted[0, 0] + 0.085, 0.0, 0.3),
            orientation=(0.9238795325, 0.0, -0.3826834324, 0.0),
        )
        contact_forces_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/obstacle.*",
            update_period=0.0,
            history_length=2,
            debug_vis=False,
            force_threshold=0.1,
        )
        self.contact_forces = ContactSensor(contact_forces_cfg)

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
        osc = OperationalSpaceController(osc_cfg, num_envs=self.num_envs, device=self.sim.device)

        self._run_op_space_controller(robot, osc, "panda_leftfinger", ["panda_joint.*"], self.target_hybrid_set_tilted)

    """
    Helper functions
    """

    def _run_op_space_controller(
        self,
        robot: Articulation,
        osc: OperationalSpaceController,
        ee_frame_name: str,
        arm_joint_names: list[str],
        target_set: torch.tensor,
    ):
        """Run the operational space controller with the given parameters.

        Args:
            robot (Articulation): The robot to control.
            osc (OperationalSpaceController): The operational space controller.
            ee_frame_name (str): The name of the end-effector frame.
            arm_joint_names (list[str]): The names of the arm joints.
            target_set (torch.tensor): The target set to track.
        """
        # Initialize the masks for evaluating target convergence according to selection matrices
        self.pos_mask = torch.tensor(osc.cfg.motion_control_axes_task[:3], device=self.sim.device).view(1, 3)
        self.rot_mask = torch.tensor(osc.cfg.motion_control_axes_task[3:], device=self.sim.device).view(1, 3)
        self.wrench_mask = torch.tensor(osc.cfg.contact_wrench_control_axes_task, device=self.sim.device).view(1, 6)
        self.force_mask = self.wrench_mask[:, 0:3]  # Take only the force components as we can measure only these

        # Define simulation stepping
        sim_dt = self.sim.get_physics_dt()
        # Play the simulator
        self.sim.reset()

        # Obtain the frame index of the end-effector
        ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
        # Obtain joint indices
        arm_joint_ids = robot.find_joints(arm_joint_names)[0]

        # Update existing buffers
        # Note: We need to update buffers before the first step for the controller.
        robot.update(dt=sim_dt)

        # Get the center of the robot soft joint limits
        joint_centers = torch.mean(robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1)

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
        ) = self._update_states(robot, ee_frame_idx, arm_joint_ids)

        # Track the given target command
        current_goal_idx = 0  # Current goal index for the arm
        command = torch.zeros(
            self.num_envs, osc.action_dim, device=self.sim.device
        )  # Generic target command, which can be pose, position, force, etc.
        ee_target_pose_b = torch.zeros(self.num_envs, 7, device=self.sim.device)  # Target pose in the body frame
        ee_target_pose_w = torch.zeros(
            self.num_envs, 7, device=self.sim.device
        )  # Target pose in the world frame (for marker)

        # Set joint efforts to zero
        zero_joint_efforts = torch.zeros(self.num_envs, robot.num_joints, device=self.sim.device)
        joint_efforts = torch.zeros(self.num_envs, len(arm_joint_ids), device=self.sim.device)

        # Now we are ready!
        for count in range(1501):
            # reset every 500 steps
            if count % 500 == 0:
                # check that we converged to the goal
                if count > 0:
                    self._check_convergence(osc, ee_pose_b, ee_target_pose_b, ee_force_b, command)
                # reset joint state to default
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                robot.set_joint_effort_target(zero_joint_efforts)  # Set zero torques in the initial step
                robot.write_data_to_sim()
                robot.reset()
                # reset contact sensor
                if self.contact_forces is not None:
                    self.contact_forces.reset()
                # reset target pose
                robot.update(sim_dt)
                _, _, _, ee_pose_b, _, _, _, _, _, _ = self._update_states(
                    robot, ee_frame_idx, arm_joint_ids
                )  # at reset, the jacobians are not updated to the latest state
                command, ee_target_pose_b, ee_target_pose_w, current_goal_idx = self._update_target(
                    osc, root_pose_w, ee_pose_b, target_set, current_goal_idx
                )
                # set the osc command
                osc.reset()
                command, task_frame_pose_b = self._convert_to_task_frame(
                    osc, command=command, ee_target_pose_b=ee_target_pose_b
                )
                osc.set_command(
                    command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b
                )
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
                ) = self._update_states(robot, ee_frame_idx, arm_joint_ids)
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
            self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            self.goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

            # perform step
            self.sim.step(render=False)
            # update buffers
            robot.update(sim_dt)

    def _update_states(
        self,
        robot: Articulation,
        ee_frame_idx: int,
        arm_joint_ids: list[int],
    ):
        """Update the states of the robot and obtain the relevant quantities for the operational space controller.

        Args:
            robot (Articulation): The robot to control.
            ee_frame_idx (int): The index of the end-effector frame.
            arm_joint_ids (list[int]): The indices of the arm joints.

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
        jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
        mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
        gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
        # Convert the Jacobian from world to root frame
        jacobian_b = jacobian_w.clone()
        root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

        # Compute current pose of the end-effector
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pose_w = robot.data.body_state_w[:, ee_frame_idx, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        # Compute the current velocity of the end-effector
        ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]  # Extract end-effector velocity in the world frame
        root_vel_w = robot.data.root_vel_w  # Extract root velocity in the world frame
        relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
        ee_lin_vel_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
        ee_ang_vel_b = quat_rotate_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
        ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

        # Calculate the contact force
        ee_force_w = torch.zeros(self.num_envs, 3, device=self.sim.device)
        if self.contact_forces is not None:  # Only modify if it exist
            sim_dt = self.sim.get_physics_dt()
            self.contact_forces.update(sim_dt)  # update contact sensor
            # Calculate the contact force by averaging over last four time steps (i.e., to smoothen) and
            # taking the max of three surfaces as only one should be the contact of interest
            ee_force_w, _ = torch.max(torch.mean(self.contact_forces.data.net_forces_w_history, dim=1), dim=1)

        # This is a simplification, only for the sake of testing.
        ee_force_b = ee_force_w

        # Get joint positions and velocities
        joint_pos = robot.data.joint_pos[:, arm_joint_ids]
        joint_vel = robot.data.joint_vel[:, arm_joint_ids]

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
        self,
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
        command = torch.zeros(self.num_envs, osc.action_dim, device=self.sim.device)
        command[:] = target_set[current_goal_idx]

        # update the ee desired pose
        ee_target_pose_b = torch.zeros(self.num_envs, 7, device=self.sim.device)
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
        self, osc: OperationalSpaceController, command: torch.tensor, ee_target_pose_b: torch.tensor
    ):
        """Convert the target command to the task frame if required.

        Args:
            osc (OperationalSpaceController): The operational space controller.
            command (torch.tensor): The target command to convert.
            ee_target_pose_b (torch.tensor): The end-effector target pose in the body frame.

        Returns:
            command (torch.tensor): The converted target command.
            task_frame_pose_b (torch.tensor): The task frame pose in the body frame.

        Raises:
            ValueError: If the frame is invalid.
        """
        command = command.clone()
        task_frame_pose_b = None
        if self.frame == "root":
            # No need to transform anything if they are already in root frame
            pass
        elif self.frame == "task":
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
        self,
        osc: OperationalSpaceController,
        ee_pose_b: torch.tensor,
        ee_target_pose_b: torch.tensor,
        ee_force_b: torch.tensor,
        ee_target_b: torch.tensor,
    ):
        """Check the convergence to the target.

        Args:
            osc (OperationalSpaceController): The operational space controller.
            ee_pose_b (torch.tensor): The end-effector pose in the body frame.
            ee_target_pose_b (torch.tensor): The end-effector target pose in the body frame.
            ee_force_b (torch.tensor): The end-effector force in the body frame.
            ee_target_b (torch.tensor): The end-effector target in the body frame.

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
                pos_error_norm = torch.norm(pos_error * self.pos_mask, dim=-1)
                rot_error_norm = torch.norm(rot_error * self.rot_mask, dim=-1)
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
                pos_error_norm = torch.norm(pos_error * self.pos_mask, dim=-1)
                rot_error_norm = torch.norm(rot_error * self.rot_mask, dim=-1)
                # desired error (zer)
                des_error = torch.zeros_like(pos_error_norm)
                # check convergence
                torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=0.1)
                torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=0.1)
                cmd_idx += 6
            elif target_type == "wrench_abs":
                force_target_b = ee_target_b[:, cmd_idx : cmd_idx + 3].clone()
                # Convert to base frame if the target was defined in task frame
                if self.frame == "task":
                    task_frame_pose_b = ee_target_pose_b.clone()
                    R_task_b = matrix_from_quat(task_frame_pose_b[:, 3:])
                    force_target_b[:] = (R_task_b @ force_target_b[:].unsqueeze(-1)).squeeze(-1)
                force_error = ee_force_b - force_target_b
                force_error_norm = torch.norm(
                    force_error * self.force_mask, dim=-1
                )  # ignore torque part as we cannot measure it
                des_error = torch.zeros_like(force_error_norm)
                # check convergence: big threshold here as the force control is not precise when the robot moves
                torch.testing.assert_close(force_error_norm, des_error, rtol=0.0, atol=1.0)
                cmd_idx += 6
            else:
                raise ValueError("Undefined target_type within _check_convergence().")


if __name__ == "__main__":
    run_tests()
