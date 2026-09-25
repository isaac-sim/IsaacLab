# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.arm_ci

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

##
# Pre-defined configs
##
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass as lab_configclass
from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_apply_inverse,
    quat_from_matrix,
    quat_inv,
    subtract_frame_transforms,
)

from isaaclab_assets import FRANKA_PANDA_CFG, G1_29DOF_CFG  # isort:skip

pytestmark = pytest.mark.integration


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

    # Create environment clones using Isaac Lab's cloner utilities
    env_prim_paths = [f"/World/envs/env_{i}" for i in range(num_envs)]
    env_fmt = "/World/envs/env_{}"
    env_ids = np.arange(num_envs, dtype=np.int64)
    env_origins, _ = cloner.grid_transforms(num_envs, spacing=2.0)
    # create source prim
    stage.DefinePrim(env_prim_paths[0], "Xform")
    # clone the env xform
    cloner.usd_replicate(stage, [env_fmt.format(0)], [env_fmt], env_ids, positions=env_origins)

    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Explicit torque actuators enforce effort limits on the commands sent to the simulator.
    for actuator_name in ("panda_shoulder", "panda_forearm"):
        actuator_cfg = robot_cfg.actuators[actuator_name]
        robot_cfg.actuators[actuator_name] = IdealPDActuatorCfg(
            joint_names_expr=actuator_cfg.joint_names_expr,
            joint_effort_limit=actuator_cfg.joint_effort_limit,
            joint_velocity_limit=actuator_cfg.joint_velocity_limit,
            armature=actuator_cfg.armature,
            stiffness=0.0,
            damping=0.0,
        )
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
    # Define goals for the arm pose [xyz + quat_xyzw] and wrench [force_xyz + torque_xyz]
    target_hybrid_set_b = ee_goal_hybrid_set_b.clone()
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
        target_hybrid_set_b,
        target_hybrid_set_tilted,
        frame,
    )

    # Cleanup
    sim.stop()
    sim.clear_instance()


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
        motion_damping_ratio_task=2.0,
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
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)

    obstacle_spawn_cfg = sim_utils.CuboidCfg(
        size=(0.7, 0.7, 0.01),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        activate_contact_sensors=True,
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_[^/]+/obstacle1",
        obstacle_spawn_cfg,
        translation=(0.2, 0.0, 0.93),
        orientation=(0.0, -0.1736, 0.0, 0.9848),
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_[^/]+/obstacle2",
        obstacle_spawn_cfg,
        translation=(0.2, 0.35, 0.7),
        orientation=(0.707, 0.0, 0.0, 0.707),
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_[^/]+/obstacle3",
        obstacle_spawn_cfg,
        translation=(0.55, 0.0, 0.7),
        orientation=(0.0, 0.707, 0.0, 0.707),
    )
    contact_forces_cfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/obstacle[^/]*",
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
        target_hybrid_set_b,
        _,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)

    obstacle_spawn_cfg = sim_utils.CuboidCfg(
        size=(1.0, 1.0, 0.01),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        activate_contact_sensors=True,
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_[^/]+/obstacle1",
        obstacle_spawn_cfg,
        translation=(target_hybrid_set_b[0, 0] + 0.05, 0.0, 0.7),
        orientation=(0.0, 0.707, 0.0, 0.707),
    )
    contact_forces_cfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/obstacle[^/]*",
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
def test_task_frame_conversion_preserves_absolute_target():
    """A rounded pose command must resolve to the same target through either reference frame."""
    osc_cfg = OperationalSpaceControllerCfg(target_types=["pose_abs"])
    osc = OperationalSpaceController(osc_cfg, num_envs=1, device="cpu")
    target_b = torch.tensor([[0.5, -0.4, 0.6, 0.707, 0.0, 0.0, 0.707]])
    resolved_targets = []
    for frame in ("root", "task"):
        converted_command, task_frame_pose_b = _convert_to_task_frame(osc, target_b, target_b, frame)
        osc.set_command(converted_command, current_task_frame_pose_b=task_frame_pose_b)
        resolved_targets.append(osc.desired_ee_pose_b.clone())

    torch.testing.assert_close(resolved_targets[0], resolved_targets[1], atol=1e-6, rtol=0.0)


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
        target_hybrid_set_tilted,
        frame,
    ) = sim

    robot = Articulation(cfg=robot_cfg)
    frame = "task"

    obstacle_spawn_cfg = sim_utils.CuboidCfg(
        size=(2.0, 1.5, 0.01),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.1),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        activate_contact_sensors=True,
    )
    obstacle_spawn_cfg.func(
        "/World/envs/env_[^/]+/obstacle1",
        obstacle_spawn_cfg,
        translation=(target_hybrid_set_tilted[0, 0] + 0.085, 0.0, 0.3),
        orientation=(0.0, -0.3826834324, 0.0, 0.9238795325),
    )
    contact_forces_cfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/obstacle[^/]*",
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
        nullspace_stiffness=1.0,
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


##
# Floating-base regression test configs (PR #5107)
##

_G1_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]


@lab_configclass
class _FloatingBaseOscSceneCfg(InteractiveSceneCfg):
    """Minimal scene with a floating-base G1 humanoid."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.fix_root_link = False
        self.robot.spawn.rigid_props.disable_gravity = True


@lab_configclass
class _FloatingBaseOscActionsCfg:
    arm_action: OperationalSpaceControllerActionCfg = OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=_G1_ARM_JOINT_NAMES,
        body_name="left_elbow_link",
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            # Both flags enabled so the action term fetches mass matrix AND
            # gravity each step, exercising the floating-base +6 indexing on
            # both quantities.
            inertial_dynamics_decoupling=True,
            gravity_compensation=True,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        ),
    )


@lab_configclass
class _FloatingBaseOscObsCfg:
    @lab_configclass
    class _PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})

    policy: _PolicyCfg = _PolicyCfg()


@lab_configclass
class _FloatingBaseOscEnvCfg(ManagerBasedEnvCfg):
    scene: _FloatingBaseOscSceneCfg = _FloatingBaseOscSceneCfg(num_envs=4, env_spacing=4.0)
    actions: _FloatingBaseOscActionsCfg = _FloatingBaseOscActionsCfg()
    observations: _FloatingBaseOscObsCfg = _FloatingBaseOscObsCfg()
    decimation: int = 1
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.01)


@pytest.mark.isaacsim_ci
def test_franka_velocity_feedback_matches_jacobian(sim):
    """The OSC action term must measure velocity at the link origin used by the Jacobian."""
    sim_context, num_envs, robot_cfg, *_ = sim
    robot = Articulation(cfg=robot_cfg)
    sim_context.reset()
    arm_joint_ids, _ = robot.find_joints("panda_joint.*")
    ee_frame_idx = robot.find_bodies("panda_hand")[0][0]

    joint_vel = torch.zeros_like(robot.data.default_joint_vel.torch)
    joint_vel[:, arm_joint_ids] = torch.linspace(0.1, 0.7, len(arm_joint_ids), device=sim_context.device)
    robot.write_joint_state_to_sim_index(position=robot.data.default_joint_pos.torch, velocity=joint_vel)
    sim_context.step(render=False)
    robot.update(sim_context.get_physics_dt())

    # Angular motion and the hand's COM offset must expose the reference-point mismatch.
    assert not torch.allclose(
        robot.data.body_com_vel_w.torch[:, ee_frame_idx, :3],
        robot.data.body_link_vel_w.torch[:, ee_frame_idx, :3],
        atol=1e-4,
        rtol=1e-4,
    )
    env = SimpleNamespace(scene={"robot": robot}, sim=sim_context, num_envs=num_envs, device=sim_context.device)
    action_cfg = OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller_cfg=OperationalSpaceControllerCfg(target_types=["pose_abs"]),
    )
    action_term = OperationalSpaceControllerAction(action_cfg, env)
    action_term._compute_ee_jacobian()
    action_term._compute_ee_velocity()
    jacobian_b, ee_vel_b = action_term._jacobian_b, action_term._ee_vel_b
    joint_vel = robot.data.joint_vel.torch[:, arm_joint_ids]

    # With a stationary fixed base, the link twist must equal J(q) * q_dot.
    expected_vel_b = torch.bmm(jacobian_b, joint_vel.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(ee_vel_b, expected_vel_b, atol=1e-4, rtol=1e-4)


@pytest.mark.isaacsim_ci
def test_floating_base_osc_action_term_indexing():
    """Regression test for #4999 / PR #5107: verify OperationalSpaceControllerAction uses correct
    indices for mass matrix and gravity on floating-base robots.

    The Jacobian / mass-matrix / gravity-comp DoF axis prepends ``num_base_dofs``
    floating-base columns (``6`` for floating-base, ``0`` for fixed-base). The action
    term's ``_compute_dynamic_quantities()`` must use ``_jacobi_joint_idx`` (with the
    ``+ num_base_dofs`` shift) instead of ``_joint_ids``. This test instantiates the
    real action term via a ManagerBasedEnv, triggers ``_compute_dynamic_quantities()``,
    and verifies the extracted mass matrix and gravity match a manual extraction using
    the correct indices.

    If someone reverts ``_jacobi_joint_idx`` back to ``_joint_ids`` in
    ``_compute_dynamic_quantities``, this test will fail.
    """
    env_cfg = _FloatingBaseOscEnvCfg()
    env_cfg.sim.device = "cuda:0"
    env = ManagerBasedEnv(cfg=env_cfg)
    num_envs = env.num_envs

    try:
        robot: Articulation = env.scene["robot"]

        # --- 1. Verify the robot is floating-base ---
        assert not robot.is_fixed_base, "G1_29DOF_CFG must be floating-base for this test"

        # --- 2. Get the action term ---
        action_term = env.action_manager._terms["arm_action"]
        num_arm_joints = action_term._num_DoF

        # --- 3. Step the env to populate physics buffers ---
        zero_actions = torch.zeros(num_envs, action_term.action_dim, device=env.device)
        action_term.process_actions(zero_actions)
        action_term.apply_actions()

        # --- 4. The action term's _mass_matrix and _gravity are now populated ---
        term_mass = action_term._mass_matrix.clone()
        term_gravity = action_term._gravity.clone()

        # --- 5. Manually extract using the CORRECT indices: arm joints shifted past the floating-base DoFs ---
        jacobi_joint_idx = [joint_id + robot.num_base_dofs for joint_id in robot.find_joints(_G1_ARM_JOINT_NAMES)[0]]
        full_mass_matrix = robot.data.mass_matrix.torch
        full_gravity = robot.data.gravity_compensation_forces.torch

        manual_mass = full_mass_matrix[:, jacobi_joint_idx, :][:, :, jacobi_joint_idx]
        manual_gravity = full_gravity[:, jacobi_joint_idx]

        # --- 6. KEY ASSERTION: action term output must match manual extraction with correct indices ---
        torch.testing.assert_close(term_mass, manual_mass, atol=1e-5, rtol=0)
        torch.testing.assert_close(term_gravity, manual_gravity, atol=1e-5, rtol=0)

        # --- 7. Verify the data-layer tensor exposes the full DoF axis (J + num_base_dofs) ---
        expected_dofs = robot.num_joints + robot.num_base_dofs
        assert full_mass_matrix.shape[1] == expected_dofs, (
            f"Mass matrix should have {expected_dofs} DoFs, got {full_mass_matrix.shape[1]}"
        )

        # --- 8. Verify correct indices differ from raw joint_ids (the old bug) ---
        # Reconstruct the original joint_ids before any slice(None) optimization
        original_joint_ids, _ = robot.find_joints(_G1_ARM_JOINT_NAMES)
        buggy_mass = full_mass_matrix[:, original_joint_ids, :][:, :, original_joint_ids]
        assert not torch.allclose(term_mass, buggy_mass, atol=1e-6), (
            "Action term mass matrix should NOT match extraction with raw joint_ids (no num_base_dofs offset)"
        )

        # --- 9. Verify physically reasonable values ---
        diag = torch.diagonal(term_mass, dim1=-2, dim2=-1)
        assert (diag > 0).all(), f"Mass matrix diagonal must be positive, got min={diag.min().item():.6f}"
        assert diag.max().item() < 100.0, (
            f"Mass matrix diagonal too large ({diag.max().item():.1f}), possibly contaminated by base DOFs"
        )
        assert torch.allclose(term_mass, term_mass.transpose(-2, -1), atol=1e-5), "Mass matrix should be symmetric"

        # --- 10. Verify shapes ---
        assert term_mass.shape == (num_envs, num_arm_joints, num_arm_joints)
        assert term_gravity.shape == (num_envs, num_arm_joints)

    finally:
        env.close()


##
# Controller law: controller == reference
##

_NUM_ENVS = 4
_NUM_DOF = 7


def _random_quat(generator: torch.Generator) -> torch.Tensor:
    """Random unit quaternions in ``(x, y, z, w)`` order, shape (``_NUM_ENVS``, 4)."""
    quat = torch.randn(_NUM_ENVS, 4, generator=generator)
    return quat / quat.norm(dim=-1, keepdim=True)


def _reference_efforts(
    cfg: OperationalSpaceControllerCfg,
    command: torch.Tensor,
    task_frame_pose_b: torch.Tensor | None,
    jacobian_b: torch.Tensor,
    ee_pose_b: torch.Tensor,
    ee_vel_b: torch.Tensor,
    ee_force_b: torch.Tensor,
    mass_matrix: torch.Tensor,
    gravity: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    nullspace_joint_pos_target: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the operational-space law independently in Torch.

    Gains, selection axes, and targets are built here from the config and the raw command, and rotated from the
    task frame into the root frame, so the reference is independent of the controller's own state.
    """
    num_envs, _, num_dof = jacobian_b.shape
    joint_efforts = torch.zeros(num_envs, num_dof)

    if task_frame_pose_b is None:
        task_frame_pose_b = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).repeat(num_envs, 1)
    frame_pos_b = task_frame_pose_b[:, :3]
    rot_task_b = matrix_from_quat(task_frame_pose_b[:, 3:])

    def to_root_frame(axis_values) -> torch.Tensor:
        """Block-rotate per-axis task-frame values into a root-frame 6x6 matrix."""
        task = torch.diag_embed(torch.as_tensor(axis_values, dtype=torch.float).expand(num_envs, 6))
        root = torch.zeros_like(task)
        root[:, 0:3, 0:3] = rot_task_b @ task[:, 0:3, 0:3] @ rot_task_b.mT
        root[:, 3:6, 3:6] = rot_task_b @ task[:, 3:6, 3:6] @ rot_task_b.mT
        return root

    def to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
        """Rodrigues rotation through the matrix exponential of the skew-symmetric matrix."""
        skew = torch.zeros(num_envs, 3, 3)
        skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -axis_angle[:, 2], axis_angle[:, 1], -axis_angle[:, 0]
        return torch.linalg.matrix_exp(skew - skew.mT)

    # split the command into targets and impedance parameters
    target_sizes = [7 if target_type == "pose_abs" else 6 for target_type in cfg.target_types]
    impedance_size = {"fixed": 0, "variable_kp": 6, "variable": 12}[cfg.impedance_mode]
    *targets, impedance = torch.split(command, [*target_sizes, impedance_size], dim=-1)
    if cfg.impedance_mode == "fixed":
        stiffness = torch.as_tensor(cfg.motion_stiffness_task, dtype=torch.float).expand(num_envs, 6)
    else:
        stiffness = impedance[:, :6].clamp(*cfg.motion_stiffness_limits_task)
    if cfg.impedance_mode == "variable":
        damping_ratio = impedance[:, 6:].clamp(*cfg.motion_damping_ratio_limits_task)
    else:
        damping_ratio = torch.as_tensor(cfg.motion_damping_ratio_task, dtype=torch.float).expand(num_envs, 6)
    motion_axes = torch.tensor(cfg.motion_control_axes_task, dtype=torch.float)
    motion_p_gains = motion_axes * stiffness
    motion_d_gains = 2.0 * motion_p_gains.sqrt() * damping_ratio

    # resolve the targets in the root frame
    desired_ee_pose_b = None
    desired_ee_wrench_b = None
    for target_type, target in zip(cfg.target_types, targets):
        if target_type == "wrench_abs":
            force_b = (rot_task_b @ target[:, :3].unsqueeze(-1)).squeeze(-1)
            torque_b = (rot_task_b @ target[:, 3:].unsqueeze(-1)).squeeze(-1) + torch.cross(
                frame_pos_b, force_b, dim=-1
            )
            desired_ee_wrench_b = torch.cat([force_b, torque_b], dim=-1)
            continue
        if target_type == "pose_abs":
            desired_pos_task = target[:, :3]
            desired_rot_task = matrix_from_quat(target[:, 3:] / target[:, 3:].norm(dim=-1, keepdim=True))
        else:
            # pose_rel: displace the current end-effector pose expressed in the task frame
            ee_pos_task = (rot_task_b.mT @ (ee_pose_b[:, :3] - frame_pos_b).unsqueeze(-1)).squeeze(-1)
            ee_rot_task = rot_task_b.mT @ matrix_from_quat(ee_pose_b[:, 3:])
            desired_pos_task = ee_pos_task + target[:, :3]
            desired_rot_task = to_matrix(target[:, 3:]) @ ee_rot_task
        desired_pos_b = frame_pos_b + (rot_task_b @ desired_pos_task.unsqueeze(-1)).squeeze(-1)
        desired_quat_b = quat_from_matrix(rot_task_b @ desired_rot_task)
        desired_ee_pose_b = torch.cat([desired_pos_b, desired_quat_b], dim=-1)

    os_mass_matrix_b = torch.zeros(num_envs, 6, 6)
    mass_matrix_inv = None

    if desired_ee_pose_b is not None:
        pose_error_b = torch.cat(
            compute_pose_error(
                ee_pose_b[:, :3],
                ee_pose_b[:, 3:],
                desired_ee_pose_b[:, :3],
                desired_ee_pose_b[:, 3:],
                rot_error_type="axis_angle",
            ),
            dim=-1,
        )
        des_ee_acc_b = to_root_frame(motion_p_gains) @ pose_error_b.unsqueeze(-1) + to_root_frame(motion_d_gains) @ (
            -ee_vel_b
        ).unsqueeze(-1)
        selection_motion_b = to_root_frame(motion_axes)
        if cfg.inertial_dynamics_decoupling:
            mass_matrix_inv = torch.inverse(mass_matrix)
            if cfg.partial_inertial_dynamics_decoupling:
                os_mass_matrix_b[:, 0:3, 0:3] = torch.inverse(
                    jacobian_b[:, 0:3] @ mass_matrix_inv @ jacobian_b[:, 0:3].mT
                )
                os_mass_matrix_b[:, 3:6, 3:6] = torch.inverse(
                    jacobian_b[:, 3:6] @ mass_matrix_inv @ jacobian_b[:, 3:6].mT
                )
            else:
                os_mass_matrix_b[:] = torch.inverse(jacobian_b @ mass_matrix_inv @ jacobian_b.mT)
            os_command_forces_b = os_mass_matrix_b @ des_ee_acc_b
        else:
            os_command_forces_b = des_ee_acc_b
        os_command_forces_b = selection_motion_b @ os_command_forces_b
        joint_efforts += (jacobian_b.mT @ os_command_forces_b).squeeze(-1)

    if desired_ee_wrench_b is not None:
        force_axes = torch.tensor(cfg.contact_wrench_control_axes_task, dtype=torch.float)
        if cfg.contact_wrench_stiffness_task is not None:
            measured_wrench_b = torch.zeros(num_envs, 6)
            measured_wrench_b[:, 0:3] = ee_force_b
            measured_wrench_b[:, 3:6] = desired_ee_wrench_b[:, 3:6]
            contact_wrench_p_gains = force_axes * torch.as_tensor(cfg.contact_wrench_stiffness_task, dtype=torch.float)
            wrench_command_b = desired_ee_wrench_b.unsqueeze(-1) + to_root_frame(contact_wrench_p_gains) @ (
                desired_ee_wrench_b - measured_wrench_b
            ).unsqueeze(-1)
        else:
            wrench_command_b = desired_ee_wrench_b.unsqueeze(-1)
        selection_force_b = to_root_frame(force_axes)
        joint_efforts += (jacobian_b.mT @ selection_force_b @ wrench_command_b).squeeze(-1)

    if cfg.gravity_compensation:
        joint_efforts += gravity

    if cfg.nullspace_control == "position":
        if cfg.inertial_dynamics_decoupling and not cfg.partial_inertial_dynamics_decoupling:
            jacobian_pinv_transpose = os_mass_matrix_b @ jacobian_b @ mass_matrix_inv
        else:
            jacobian_pinv_transpose = torch.pinverse(jacobian_b).mT
        nullspace_jacobian_transpose = torch.eye(n=num_dof) - jacobian_b.mT @ jacobian_pinv_transpose
        nullspace_p_gain = cfg.nullspace_stiffness
        nullspace_d_gain = 2.0 * nullspace_p_gain**0.5 * cfg.nullspace_damping_ratio
        joint_acc_nullspace = (
            nullspace_p_gain * (nullspace_joint_pos_target - joint_pos) + nullspace_d_gain * (-joint_vel)
        ).unsqueeze(-1)
        joint_efforts += (nullspace_jacobian_transpose @ mass_matrix @ joint_acc_nullspace).squeeze(-1)

    return joint_efforts


_SCENARIOS = {
    "pose_abs": dict(target_types=["pose_abs"]),
    "pose_rel": dict(target_types=["pose_rel"]),
    "pose_rel_task_frame": dict(target_types=["pose_rel"], task_frame=True),
    "pose_abs_task_frame": dict(target_types=["pose_abs"], task_frame=True),
    "pose_abs_decoupled": dict(target_types=["pose_abs"], inertial_dynamics_decoupling=True, task_frame=True),
    "pose_abs_partial_decoupled": dict(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=True,
        task_frame=True,
    ),
    "pose_abs_gravity": dict(target_types=["pose_abs"], gravity_compensation=True, task_frame=True),
    "pose_abs_nullspace": dict(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        nullspace_control="position",
        task_frame=True,
    ),
    "pose_abs_nullspace_without_inertia": dict(target_types=["pose_abs"], nullspace_control="position"),
    "pose_abs_nullspace_partial_inertia": dict(
        target_types=["pose_abs"],
        nullspace_control="position",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=True,
    ),
    "wrench_only": dict(
        target_types=["wrench_abs"],
        motion_control_axes_task=(0, 0, 0, 0, 0, 0),
        contact_wrench_control_axes_task=(1, 1, 1, 1, 1, 1),
        task_frame=True,
    ),
    "wrench_open_loop": dict(
        target_types=["pose_abs", "wrench_abs"],
        motion_control_axes_task=(1, 1, 0, 1, 1, 1),
        contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
        task_frame=True,
    ),
    "wrench_closed_loop": dict(
        target_types=["pose_abs", "wrench_abs"],
        motion_control_axes_task=(1, 1, 0, 1, 1, 1),
        contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
        contact_wrench_stiffness_task=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        task_frame=True,
    ),
    "wrench_closed_loop_decoupled": dict(
        target_types=["pose_abs", "wrench_abs"],
        contact_wrench_stiffness_task=0.5,
        contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
        inertial_dynamics_decoupling=True,
        gravity_compensation=True,
        nullspace_control="position",
        task_frame=True,
    ),
    "wrench_decoupled_partial_axes": dict(
        target_types=["pose_abs", "wrench_abs"],
        motion_control_axes_task=(1, 1, 0, 1, 1, 1),
        contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
        contact_wrench_stiffness_task=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        inertial_dynamics_decoupling=True,
        task_frame=True,
    ),
    "variable_kp": dict(target_types=["pose_abs"], impedance_mode="variable_kp", task_frame=True),
    "variable": dict(target_types=["pose_abs"], impedance_mode="variable", task_frame=True),
    "hybrid_variable_kp_decoupled": dict(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="variable_kp",
        motion_control_axes_task=(0, 1, 1, 1, 1, 1),
        contact_wrench_control_axes_task=(1, 0, 0, 0, 0, 0),
        inertial_dynamics_decoupling=True,
        task_frame=True,
    ),
    "hybrid_variable_decoupled": dict(
        target_types=["pose_abs", "wrench_abs"],
        impedance_mode="variable",
        motion_control_axes_task=(0, 1, 1, 1, 1, 1),
        contact_wrench_control_axes_task=(1, 0, 0, 0, 0, 0),
        inertial_dynamics_decoupling=True,
        task_frame=True,
    ),
}


@pytest.mark.parametrize("scenario_name", list(_SCENARIOS))
def test_compute_matches_operational_space_law(scenario_name: str) -> None:
    """The controller matches an independent operational-space reference."""
    generator = torch.Generator().manual_seed(0)
    scenario = dict(_SCENARIOS[scenario_name])
    task_frame = scenario.pop("task_frame", False)
    cfg = OperationalSpaceControllerCfg(
        motion_stiffness_task=(120.0, 130.0, 140.0, 15.0, 16.0, 17.0),
        motion_damping_ratio_task=(1.0, 1.1, 0.9, 1.0, 1.2, 0.8),
        **scenario,
    )
    controller = OperationalSpaceController(cfg, num_envs=_NUM_ENVS, device="cpu")

    ee_pose_b = torch.cat([0.4 * torch.randn(_NUM_ENVS, 3, generator=generator), _random_quat(generator)], dim=-1)
    ee_vel_b = 0.2 * torch.randn(_NUM_ENVS, 6, generator=generator)
    ee_force_b = 3.0 * torch.randn(_NUM_ENVS, 3, generator=generator)
    jacobian_b = torch.randn(_NUM_ENVS, 6, _NUM_DOF, generator=generator)
    factor = torch.randn(_NUM_ENVS, _NUM_DOF, _NUM_DOF, generator=generator)
    mass_matrix = factor @ factor.mT + 3.0 * torch.eye(_NUM_DOF)  # SPD
    gravity = 0.5 * torch.randn(_NUM_ENVS, _NUM_DOF, generator=generator)
    joint_pos = 0.3 * torch.randn(_NUM_ENVS, _NUM_DOF, generator=generator)
    joint_vel = 0.2 * torch.randn(_NUM_ENVS, _NUM_DOF, generator=generator)
    nullspace_target = 0.1 * torch.randn(_NUM_ENVS, _NUM_DOF, generator=generator)
    task_frame_pose_b = (
        torch.cat([0.2 * torch.randn(_NUM_ENVS, 3, generator=generator), _random_quat(generator)], dim=-1)
        if task_frame
        else None
    )

    command = []
    for target_type in cfg.target_types:
        if target_type == "pose_abs":
            command.append(
                torch.cat([0.3 * torch.randn(_NUM_ENVS, 3, generator=generator), _random_quat(generator)], dim=-1)
            )
        elif target_type == "pose_rel":
            command.append(0.1 * torch.randn(_NUM_ENVS, 6, generator=generator))
        else:
            command.append(5.0 * torch.randn(_NUM_ENVS, 6, generator=generator))
    if cfg.impedance_mode in ("variable_kp", "variable"):
        command.append(torch.rand(_NUM_ENVS, 6, generator=generator) * 150.0 + 50.0)
    if cfg.impedance_mode == "variable":
        command.append(torch.rand(_NUM_ENVS, 6, generator=generator) * 2.0)
    command = torch.cat(command, dim=-1)

    controller.set_command(command.clone(), current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b)
    actual = controller.compute(
        jacobian_b=jacobian_b,
        current_ee_pose_b=ee_pose_b,
        current_ee_vel_b=ee_vel_b,
        current_ee_force_b=ee_force_b,
        mass_matrix=mass_matrix,
        gravity=gravity,
        current_joint_pos=joint_pos,
        current_joint_vel=joint_vel,
        nullspace_joint_pos_target=nullspace_target,
    )
    reference = _reference_efforts(
        cfg,
        command,
        task_frame_pose_b,
        jacobian_b,
        ee_pose_b,
        ee_vel_b,
        ee_force_b,
        mass_matrix,
        gravity,
        joint_pos,
        joint_vel,
        nullspace_target,
    )
    torch.testing.assert_close(actual, reference, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
    ],
)
@pytest.mark.parametrize("partial_inertial_decoupling", [False, True])
def test_inertial_decoupling_damps_near_singular_directions(device, partial_inertial_decoupling):
    """Finite near-singular modes must not amplify commands or couple retained tasks to posture control."""
    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=partial_inertial_decoupling,
        nullspace_control="position",
        nullspace_stiffness=1.0,
    )
    controller = OperationalSpaceController(cfg, num_envs=4, device=device)
    pose = torch.zeros(4, 7, device=device)
    pose[:, -1] = 1.0
    controller.set_command(pose)

    # Known task directions, rotated within each translation/rotation block.
    basis = torch.eye(6, device=device)
    basis[:2, :2] = basis[3:5, 3:5] = torch.tensor([[0.6, -0.8], [0.8, 0.6]], device=device)
    scales = torch.ones(4, 6, device=device)
    scales[0] = torch.tensor([0.5, 0.7, 1.0, 0.4, 0.6, 0.9], device=device)
    scales[1, 0] = scales[2, 3] = 0.002
    scales[3] = 0.0
    masses = torch.arange(2.0, 9.0, device=device)
    joint_basis = torch.eye(7, device=device)
    joint_basis[0, 0] = joint_basis[6, 6] = 0.6
    joint_basis[0, 6], joint_basis[6, 0] = -0.8, 0.8
    mass = (joint_basis @ torch.diag(masses) @ joint_basis.mT).repeat(4, 1, 1)
    jacobian = torch.zeros(4, 6, 7, device=device)
    jacobian[:, :, :6] = basis @ torch.diag_embed(scales * masses[:6].sqrt())
    jacobian = jacobian @ joint_basis.mT
    acceleration = torch.arange(1.0, 7.0, device=device).repeat(4, 1)
    inputs = dict(
        jacobian_b=jacobian,
        mass_matrix=mass,
        current_ee_pose_b=pose,
        current_ee_vel_b=-acceleration / 20.0,
        current_joint_pos=torch.zeros(4, 7, device=device),
        current_joint_vel=torch.zeros(4, 7, device=device),
    )
    efforts = controller.compute(**inputs)
    retained = scales > 0.1
    # All nonzero weak modes are below the lower threshold: the scalar damped response is s / (s² + d).
    damping = cfg.inertia_conditioning_thresholds[0]
    gains = torch.where(retained, scales.clamp_min(0.1).reciprocal(), scales / (scales.square() + damping))
    expected = torch.zeros_like(efforts)
    expected[:, :6] = masses[:6].sqrt() * (acceleration @ basis) * gains
    expected = expected @ joint_basis.mT
    torch.testing.assert_close(efforts, expected, atol=2e-4, rtol=2e-4)

    # Batch composition must not change the result when only some environments need damping.
    for env_id in range(4):
        single_controller = OperationalSpaceController(cfg, num_envs=1, device=device)
        single_controller.set_command(pose[env_id : env_id + 1])
        single_efforts = single_controller.compute(**{key: value[env_id : env_id + 1] for key, value in inputs.items()})
        torch.testing.assert_close(single_efforts[0], efforts[env_id], atol=2e-4, rtol=2e-4)

    # Full inertia decoupling must isolate every retained task direction from posture torques.
    if not partial_inertial_decoupling:
        with_posture = controller.compute(
            **inputs, nullspace_joint_pos_target=torch.ones_like(efforts) @ joint_basis.mT
        )
        null_acceleration = (jacobian @ torch.linalg.solve(mass, (with_posture - efforts).unsqueeze(-1))).squeeze(-1)
        torch.testing.assert_close(
            (null_acceleration @ basis) * retained, torch.zeros(4, 6, device=device), atol=2e-4, rtol=0.0
        )
        expected_null = masses.expand_as(efforts).clone()
        expected_null[:, :6] *= 1.0 - scales * gains
        expected_null = expected_null @ joint_basis.mT
        torch.testing.assert_close(with_posture - efforts, expected_null, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
    ],
)
def test_inertial_decoupling_smoothly_releases_weak_directions(device):
    """Task and posture efforts are continuous at both conditioning thresholds."""
    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        nullspace_control="position",
        nullspace_stiffness=1.0,
        inertia_conditioning_thresholds=(1e-4, 1e-3),
    )
    lower, upper = cfg.inertia_conditioning_thresholds
    offsets = torch.tensor([1 - 1e-5, 1.0, 1 + 1e-5], device=device)
    ratios = torch.cat((lower * offsets, torch.tensor([(lower + upper) / 2], device=device), upper * offsets))
    num_envs = len(ratios)
    controller = OperationalSpaceController(cfg, num_envs=num_envs, device=device)
    pose = torch.zeros(num_envs, 7, device=device)
    pose[:, -1] = 1.0
    controller.set_command(pose)
    jacobian = torch.eye(6, 7, device=device).repeat(num_envs, 1, 1)
    jacobian[:, 0, 0] = ratios.sqrt()
    inputs = dict(
        jacobian_b=jacobian,
        mass_matrix=torch.eye(7, device=device).repeat(num_envs, 1, 1),
        current_ee_pose_b=pose,
        current_ee_vel_b=torch.full((num_envs, 6), -0.05, device=device),
        current_joint_pos=torch.zeros(num_envs, 7, device=device),
        current_joint_vel=torch.zeros(num_envs, 7, device=device),
    )
    task_efforts = controller.compute(**inputs).clone()
    posture_efforts = controller.compute(**inputs, nullspace_joint_pos_target=torch.ones(num_envs, 7, device=device))
    posture_efforts = posture_efforts - task_efforts

    # Damping equals the lower threshold below the band, halves at its midpoint, and vanishes above it.
    reference_damping = torch.tensor([lower, lower / 2, 0.0], device=device)
    response = ratios[[1, 3, 5]] / (ratios[[1, 3, 5]] + reference_damping)
    torch.testing.assert_close(task_efforts[[1, 3, 5], 0], response / ratios[[1, 3, 5]].sqrt())
    torch.testing.assert_close(posture_efforts[[1, 3, 5], 0], 1.0 - response)
    for efforts in (task_efforts, posture_efforts):
        torch.testing.assert_close(efforts[:3, 0], efforts[1, 0].expand(3), atol=2e-4, rtol=1e-5)
        torch.testing.assert_close(efforts[-3:, 0], efforts[5, 0].expand(3), atol=2e-4, rtol=1e-5)
    torch.testing.assert_close(task_efforts[:, 1:6], torch.ones(num_envs, 5, device=device))
    torch.testing.assert_close(posture_efforts[:, 1:6], torch.zeros(num_envs, 5, device=device))
    torch.testing.assert_close(posture_efforts[:, 6], torch.ones(num_envs, device=device))


@pytest.mark.parametrize(
    "thresholds",
    [(0.0, 1e-4), (-1.0, 1e-4), (1e-4, 1e-4), (1e-3, 1e-4), (1e-4, 1.1), (1e-4, float("nan")), (1e-4, float("inf"))],
)
def test_inertial_decoupling_rejects_invalid_conditioning_thresholds(thresholds):
    cfg = OperationalSpaceControllerCfg(target_types=["pose_abs"], inertia_conditioning_thresholds=thresholds)
    with pytest.raises(ValueError, match="conditioning thresholds"):
        OperationalSpaceController(cfg, num_envs=1, device="cpu")


@pytest.mark.parametrize("partial_inertial_decoupling", [False, True])
def test_inertial_decoupling_handles_singular_task_inertia(partial_inertial_decoupling: bool):
    """Inertial decoupling produces finite efforts for rank-deficient Jacobians in a mixed batch."""
    num_envs = 3
    num_joints = 7
    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=partial_inertial_decoupling,
    )
    target_pose = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    jacobian = torch.zeros(num_envs, 6, num_joints)
    jacobian[:, :6, :6] = torch.eye(6)
    jacobian[1, 1] = 0.0  # singular translational task-space inertia
    jacobian[2, 5] = 0.0  # singular rotational task-space inertia
    current_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    expected = torch.zeros(num_envs, num_joints)
    expected[:, 0] = 10.0

    controller = OperationalSpaceController(cfg, num_envs=num_envs, device="cpu")
    controller.set_command(target_pose)
    joint_efforts = controller.compute(
        jacobian_b=jacobian,
        current_ee_pose_b=current_pose,
        current_ee_vel_b=torch.zeros(num_envs, 6),
        mass_matrix=torch.eye(num_joints).repeat(num_envs, 1, 1),
    )
    torch.testing.assert_close(joint_efforts, expected)


def test_reset_clears_the_task_space_targets() -> None:
    """After a reset no target is commanded, so only gravity compensation remains."""
    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        gravity_compensation=True,
        # closed-loop force control, so a stale measured wrench would surface as torque
        contact_wrench_stiffness_task=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
    )
    controller = OperationalSpaceController(cfg, num_envs=_NUM_ENVS, device="cpu")
    generator = torch.Generator().manual_seed(0)
    ee_pose_b = torch.cat([0.4 * torch.randn(_NUM_ENVS, 3, generator=generator), _random_quat(generator)], dim=-1)
    gravity = 0.5 * torch.randn(_NUM_ENVS, _NUM_DOF, generator=generator)
    command = torch.cat(
        [
            torch.cat([0.3 * torch.randn(_NUM_ENVS, 3, generator=generator), _random_quat(generator)], dim=-1),
            5.0 * torch.randn(_NUM_ENVS, 6, generator=generator),
        ],
        dim=-1,
    )
    controller.set_command(command, current_ee_pose_b=ee_pose_b)
    # step once while commanded to populate the measured-wrench state;
    # only then does the reset have stale state to clear
    controller.compute(
        jacobian_b=torch.randn(_NUM_ENVS, 6, _NUM_DOF, generator=generator),
        current_ee_pose_b=ee_pose_b,
        current_ee_vel_b=0.2 * torch.randn(_NUM_ENVS, 6, generator=generator),
        current_ee_force_b=3.0 * torch.randn(_NUM_ENVS, 3, generator=generator),
        gravity=gravity,
    )
    controller.reset()

    efforts = controller.compute(
        jacobian_b=torch.randn(_NUM_ENVS, 6, _NUM_DOF, generator=generator),
        current_ee_pose_b=ee_pose_b,
        current_ee_vel_b=0.2 * torch.randn(_NUM_ENVS, 6, generator=generator),
        current_ee_force_b=3.0 * torch.randn(_NUM_ENVS, 3, generator=generator),
        gravity=gravity,
    )
    torch.testing.assert_close(efforts, gravity, atol=1e-4, rtol=1e-4)


def _pose_abs_controller(num_envs: int) -> OperationalSpaceController:
    cfg = OperationalSpaceControllerCfg(target_types=["pose_abs"], inertial_dynamics_decoupling=False)
    return OperationalSpaceController(cfg, num_envs=num_envs, device="cpu")


def test_pose_abs_target_quaternion_is_normalized():
    """Scaling an absolute pose quaternion must not change the target orientation or the commanded efforts."""
    num_envs = 2
    unit_quat = torch.tensor([[0.0, 0.0, 0.3826834, 0.9238795]]).repeat(num_envs, 1)  # 45 deg about z
    unit_target = torch.cat([torch.tensor([[0.1, 0.0, 0.0]]).repeat(num_envs, 1), unit_quat], dim=-1)
    scaled_target = unit_target.clone()
    scaled_target[0, 3:7] *= 3.0
    scaled_target[1, 3:7] *= -0.25  # sign flip encodes the same rotation

    current_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    jacobian = torch.zeros(num_envs, 6, 7)
    jacobian[:, :6, :6] = torch.eye(6)

    efforts = []
    for target in (unit_target, scaled_target):
        controller = _pose_abs_controller(num_envs)
        controller.set_command(target, current_ee_pose_b=current_pose)
        torch.testing.assert_close(
            torch.linalg.norm(controller.desired_ee_pose_task[:, 3:7], dim=-1), torch.ones(num_envs)
        )
        efforts.append(
            controller.compute(
                jacobian_b=jacobian, current_ee_pose_b=current_pose, current_ee_vel_b=torch.zeros(num_envs, 6)
            )
        )

    torch.testing.assert_close(efforts[0], efforts[1])
    assert torch.isfinite(efforts[1]).all()


def test_pose_abs_degenerate_quaternion_falls_back_to_current_orientation():
    """Zero and non-finite quaternions keep the current orientation, or identity without a current pose."""
    num_envs = 3
    current_quat = torch.tensor([[0.0, 0.7071068, 0.0, 0.7071068]]).repeat(num_envs, 1)  # 90 deg about y
    current_pose = torch.cat([torch.zeros(num_envs, 3), current_quat], dim=-1)
    target = torch.zeros(num_envs, 7)
    target[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    target[1, 3:7] = 0.0
    target[2, 3:7] = torch.tensor([float("nan"), 0.0, 0.0, 1.0])

    controller = _pose_abs_controller(num_envs)
    controller.set_command(target, current_ee_pose_b=current_pose)
    torch.testing.assert_close(controller.desired_ee_pose_task[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(controller.desired_ee_pose_task[1:, 3:7], current_quat[1:])

    controller = _pose_abs_controller(num_envs)
    controller.set_command(target)
    torch.testing.assert_close(
        controller.desired_ee_pose_task[1:, 3:7], torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs - 1, 1)
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
    position_tolerance: float = 0.1,
    rotation_tolerance: float = 0.1,
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
        position_tolerance (float): Maximum position error norm. Defaults to 0.1.
        rotation_tolerance (float): Maximum rotation error norm. Defaults to 0.1.
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
    joint_centers = torch.mean(robot.data.soft_joint_pos_limits.torch[:, arm_joint_ids, :], dim=-1)

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
    task_frame_pose_b = None  # Task frame pose in the body frame, set with each new target

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
                    osc,
                    task_frame_pose_b,
                    ee_pose_b,
                    ee_target_pose_b,
                    ee_force_b,
                    command,
                    pos_mask,
                    rot_mask,
                    force_mask,
                    frame,
                    position_tolerance,
                    rotation_tolerance,
                )
            # reset joint state to default
            default_joint_pos = robot.data.default_joint_pos.torch.clone()
            default_joint_vel = robot.data.default_joint_vel.torch.clone()
            robot.write_joint_position_to_sim_index(position=default_joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=default_joint_vel)
            robot.set_joint_effort_target_index(target=zero_joint_efforts)  # Set zero torques in the initial step
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
            command, task_frame_pose_b = _convert_to_task_frame(
                osc, command=command, ee_target_pose_b=ee_target_pose_b, frame=frame
            )
            osc.reset()
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
            robot.set_joint_effort_target_index(target=joint_efforts, joint_ids=arm_joint_ids)
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
    jacobian_w = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.data.mass_matrix.torch[:, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.data.gravity_compensation_forces.torch[:, arm_joint_ids]
    # Convert the Jacobian from world to root frame
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w.torch))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # Compute current pose of the end-effector
    root_pose_w = robot.data.root_pose_w.torch
    ee_pose_w = robot.data.body_pose_w.torch[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    # Match the link-origin reference point used by the pose and Jacobian.
    ee_vel_w = robot.data.body_link_vel_w.torch[:, ee_frame_idx, :]
    root_vel_w = robot.data.root_link_vel_w.torch
    relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w.torch, relative_vel_w[:, 0:3])  # From world to root frame
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w.torch, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # Calculate the contact force
    ee_force_w = torch.zeros(num_envs, 3, device=sim.device)
    if contact_forces is not None:  # Only modify if it exist
        sim_dt = sim.get_physics_dt()
        contact_forces.update(sim_dt)  # update contact sensor
        # Calculate the contact force by averaging over last four time steps (i.e., to smoothen) and
        # taking the max of three surfaces as only one should be the contact of interest
        ee_force_w, _ = torch.max(torch.mean(contact_forces.data.net_normal_forces_w_history.torch, dim=1), dim=1)

    # This is a simplification, only for the sake of testing.
    ee_force_b = ee_force_w

    # Get joint positions and velocities
    joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel.torch[:, arm_joint_ids]

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
        # Rounded goal quaternions must define a unit rotation when used as a reference frame.
        task_frame_pose_b[:, 3:] /= torch.linalg.vector_norm(task_frame_pose_b[:, 3:], dim=-1, keepdim=True)

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
    task_frame_pose_b: torch.Tensor | None,
    ee_pose_b: torch.tensor,
    ee_target_pose_b: torch.tensor,
    ee_force_b: torch.tensor,
    ee_target_b: torch.tensor,
    pos_mask: torch.tensor,
    rot_mask: torch.tensor,
    force_mask: torch.tensor,
    frame: str,
    position_tolerance: float,
    rotation_tolerance: float,
):
    """Check the convergence to the target.

    Args:
        osc (OperationalSpaceController): The operational space controller.
        task_frame_pose_b (torch.Tensor | None): The task frame pose in the body frame, required when
            ``frame`` is ``"task"``.
        ee_pose_b (torch.tensor): The end-effector pose in the body frame.
        ee_target_pose_b (torch.tensor): The end-effector target pose in the body frame.
        ee_force_b (torch.tensor): The end-effector force in the body frame.
        ee_target_b (torch.tensor): The end-effector target in the body frame.
        pos_mask (torch.tensor): The position mask.
        rot_mask (torch.tensor): The rotation mask.
        force_mask (torch.tensor): The force mask.
        frame (str): The reference frame for targets.
        position_tolerance (float): Maximum position error norm.
        rotation_tolerance (float): Maximum rotation error norm.

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
            if frame == "task":
                pos_error = quat_apply_inverse(task_frame_pose_b[:, 3:], pos_error)
                rot_error = quat_apply_inverse(task_frame_pose_b[:, 3:], rot_error)
            pos_error_norm = torch.linalg.norm(pos_error * pos_mask, dim=-1)
            rot_error_norm = torch.linalg.norm(rot_error * rot_mask, dim=-1)
            # desired error (zer)
            des_error = torch.zeros_like(pos_error_norm)
            # check convergence
            torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=position_tolerance)
            torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=rotation_tolerance)
            cmd_idx += 7
        elif target_type == "pose_rel":
            pos_error, rot_error = compute_pose_error(
                ee_pose_b[:, 0:3], ee_pose_b[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
            )
            if frame == "task":
                pos_error = quat_apply_inverse(task_frame_pose_b[:, 3:], pos_error)
                rot_error = quat_apply_inverse(task_frame_pose_b[:, 3:], rot_error)
            pos_error_norm = torch.linalg.norm(pos_error * pos_mask, dim=-1)
            rot_error_norm = torch.linalg.norm(rot_error * rot_mask, dim=-1)
            # desired error (zer)
            des_error = torch.zeros_like(pos_error_norm)
            # check convergence
            torch.testing.assert_close(pos_error_norm, des_error, rtol=0.0, atol=position_tolerance)
            torch.testing.assert_close(rot_error_norm, des_error, rtol=0.0, atol=rotation_tolerance)
            cmd_idx += 6
        elif target_type == "wrench_abs":
            force_target = ee_target_b[:, cmd_idx : cmd_idx + 3]
            force = ee_force_b
            if frame == "task":
                force = quat_apply_inverse(task_frame_pose_b[:, 3:], force)
            force_error = force - force_target
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
