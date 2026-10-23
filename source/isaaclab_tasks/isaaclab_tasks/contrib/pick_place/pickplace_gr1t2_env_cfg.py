# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import DampingTaskCfg, FrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib import ISAACLAB_TELEOP_AVAILABLE
from isaaclab_tasks.contrib.steering_wheel import (
    condition_newton_steering_wheel_detail_collisions as _condition_newton_steering_wheel_detail_collisions,
)
from isaaclab_tasks.contrib.teleop_physics_cfg import TeleopPhysicsCfg

from . import mdp
from .gr1t2_kitless_action_cfg import GR1T2KitlessPinkInverseKinematicsActionCfg

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip

from isaaclab_tasks.contrib.robot_pov_camera_cfg import robot_pov_camera_cfg  # isort: skip
from isaaclab_tasks.contrib.packing_table import condition_packing_table_for_kitless_ovphysx  # isort: skip


def build_gr1t2_pickplace_pipeline():
    """Build an Isaac Capture retargeting pipeline for GR1T2 pick-place teleoperation.

    Creates two Se3AbsRetargeters for left and right wrist pose tracking and
    two DexHandRetargeters for left and right dexterous hand finger control
    from hand tracking data. All outputs are flattened into a single action
    tensor via TensorReorderer.
    """
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # Create input sources (trackers are auto-discovered from pipeline)
    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    # External input: world-to-anchor 4x4 transform matrix provided by IsaacTeleopDevice
    transform_input = ValueInput("world_T_anchor", TransformMatrix())

    # Apply the coordinate-frame transform to controller poses so that
    # downstream retargeters receive data in the simulation world frame.
    _transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # -------------------------------------------------------------------------
    # SE3 Absolute Pose Retargeters (left and right wrists)
    # -------------------------------------------------------------------------
    # Left wrist: identity rotation offset (passes through as-is in original retargeter)
    left_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {
            HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT),
        }
    )

    # Right wrist: 180-degree Z rotation offset
    # From GR1T2Retargeter._retarget_abs: the USD control frame is 180 degrees
    # rotated around the Z axis w.r.t. the OpenXR frame.
    right_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=180.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {
            HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # DexHand Retargeters (left and right hands)
    # -------------------------------------------------------------------------
    # Resolve dex-retargeting YAML config paths from IsaacLab's retargeter data directory
    import isaaclab_teleop.isaac_teleop_cfg as _teleop_cfg_mod

    _teleop_cfg_file = _teleop_cfg_mod.__file__
    if _teleop_cfg_file is None:
        raise RuntimeError("Could not resolve isaaclab_teleop package path for dex-retargeting configs.")
    _teleop_pkg_dir = os.path.dirname(_teleop_cfg_file)
    _data_dir = os.path.join(
        _teleop_pkg_dir,
        "deprecated",
        "openxr",
        "retargeters",
        "humanoid",
        "fourier",
        "data",
    )
    _config_dir = os.path.join(_data_dir, "configs", "dex-retargeting")
    left_yaml_path = os.path.join(_config_dir, "fourier_hand_left_dexpilot.yml")
    right_yaml_path = os.path.join(_config_dir, "fourier_hand_right_dexpilot.yml")

    # Resolve URDF paths (downloads from Omniverse if needed)
    local_left_urdf = retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_left_hand.urdf")
    local_right_urdf = retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_right_hand.urdf")

    # Hand-tracking to base-link frame transform (OPERATOR2MANO matrix)
    # From gr1_t2_dex_retargeting_utils: [[0,-1,0],[-1,0,0],[0,0,-1]]
    operator2mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)

    # Joint names for each hand (11 DOF per hand)
    left_hand_joint_names = [
        "L_index_proximal_joint",
        "L_index_intermediate_joint",
        "L_middle_proximal_joint",
        "L_middle_intermediate_joint",
        "L_pinky_proximal_joint",
        "L_pinky_intermediate_joint",
        "L_ring_proximal_joint",
        "L_ring_intermediate_joint",
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_distal_joint",
    ]

    right_hand_joint_names = [
        "R_index_proximal_joint",
        "R_index_intermediate_joint",
        "R_middle_proximal_joint",
        "R_middle_intermediate_joint",
        "R_pinky_proximal_joint",
        "R_pinky_intermediate_joint",
        "R_ring_proximal_joint",
        "R_ring_intermediate_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_distal_joint",
    ]

    left_dex_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=left_yaml_path,
        hand_urdf=local_left_urdf,
        hand_joint_names=left_hand_joint_names,
        hand_side="left",
        handtracking_to_baselink_frame_transform=operator2mano,
    )
    left_dex = DexHandRetargeter(left_dex_cfg, name="left_hand")
    connected_left_dex = left_dex.connect(
        {
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
        }
    )

    right_dex_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=right_yaml_path,
        hand_urdf=local_right_urdf,
        hand_joint_names=right_hand_joint_names,
        hand_side="right",
        handtracking_to_baselink_frame_transform=operator2mano,
    )
    right_dex = DexHandRetargeter(right_dex_cfg, name="right_hand")
    connected_right_dex = right_dex.connect(
        {
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # TensorReorderer: flatten into a 36D action tensor
    # -------------------------------------------------------------------------
    # Se3AbsRetargeter outputs 7D arrays: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]

    # Output order must match the PinkInverseKinematicsActionCfg expected tensor layout:
    #   [left_wrist(7), right_wrist(7), hand_joints(22)]
    # Hand joints follow hand_joint_names order from ActionsCfg.upper_body_ik.
    output_order = (
        left_ee_elements
        + right_ee_elements
        + [
            # hand_joint_names indices 0-4 (left proximal + thumb yaw)
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            # hand_joint_names indices 5-9 (right proximal + thumb yaw)
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            # hand_joint_names indices 10-14 (left intermediate + thumb pitch)
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            # hand_joint_names indices 15-19 (right intermediate + thumb pitch)
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            # hand_joint_names indices 20-21 (thumb distal)
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ]
    )

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": left_hand_joint_names,
            "right_hand_joints": right_hand_joint_names,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_hand_joints": connected_left_dex.output("hand_joints"),
            "right_hand_joints": connected_right_dex.output("hand_joints"),
        }
    )

    pipeline = OutputCombiner(
        {
            "action": connected_reorderer.output("output"),
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )
    return pipeline, [left_dex, right_dex]


##
# Scene definition
##

# The steering wheel USD authors its rigid body on a nested prim rather than at the
# spawned ``Object`` root, so contact filtering must target that actor: filtering
# against ``Object`` matches an empty Xform and normal_force_matrix_w always reads zero.
_STEERING_WHEEL_BODY = "{ENV_REGEX_NS}/Object/Geometry/sm_steeringwheel_a01_01"


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the GR1T2 Pick Place Base Scene."""

    # Table
    packing_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 0.9996], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        ),
    )

    # Humanoid robot configured for pick-place manipulation tasks
    robot: ArticulationCfg = replace(
        GR1T2_HIGH_PD_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.0, 0.0, 0.7071, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Per-finger contact sensors on all finger links of each hand, filtered against
    # the wheel body so normal_force_matrix_w reports each finger's grip force. This drives
    # the per-finger haptic glove feedback (see GloveHapticFeedbackCfg below).
    # Contact reporting is already enabled on the robot by GR1T2_HIGH_PD_CFG
    # (``spawn.activate_contact_sensors=True``), so it is not set again here.
    left_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*L_(index|middle|ring|pinky|thumb)[^/]*_link",
        filter_prim_paths_expr=[_STEERING_WHEEL_BODY],
        update_period=0.0,
        history_length=3,
    )
    right_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*R_(index|middle|ring|pinky|thumb)[^/]*_link",
        filter_prim_paths_expr=[_STEERING_WHEEL_BODY],
        update_period=0.0,
        history_length=3,
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class PickPlaceGR1T2SceneCfg(ObjectTableSceneCfg):
    """GR1T2 pick-place scene with the camera observation shown in XR PiP."""

    robot_pov_cam = robot_pov_camera_cfg(
        parent_prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset_pos=(0.11999996, -0.00000233, 0.74674994),
        offset_rot=(-0.69303199, 0.69304552, -0.14034840, 0.14034565),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = GR1T2KitlessPinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        hand_joint_names=[
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_hand_pitch_link",
            "right_wrist": "right_hand_pitch_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        # The frames names are the ones present in the URDF file
        # The urdf has to be generated from the USD that is being used in the scene
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=22,
            show_ik_warnings=False,
            # Determines whether Pink IK solver will fail due to a joint limit violation
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame="left_hand_pitch_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=12,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTaskCfg(
                    frame="right_hand_pitch_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=12,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                DampingTaskCfg(
                    cost=0.5,  # [cost] * [s] / [rad]
                ),
                NullSpacePostureTaskCfg(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "left_hand_pitch_link",
                        "right_hand_pitch_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                ),
            ],
            fixed_input_tasks=[],
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_hand_roll_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_hand_roll_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"})

        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})
        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]},
        )

        object = ObsTerm(
            func=mdp.object_obs,
            params={"left_eef_link_name": "left_hand_roll_link", "right_eef_link_name": "right_hand_roll_link"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class PickPlaceGR1T2ObservationsCfg(ObservationsCfg):
    """GR1T2 pick-place observations including the camera shown in XR PiP."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        robot_pov_cam = ObsTerm(
            func=base_mdp.image_rgb,
            params={
                "sensor_cfg": SceneEntityCfg("robot_pov_cam"),
                "normalize": False,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done_pick_place, params={"task_link_name": "right_hand_roll_link"})


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceGR1T2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    sim: SimulationCfg = SimulationCfg(physics=TeleopPhysicsCfg())

    # Scene settings
    scene: PickPlaceGR1T2SceneCfg = PickPlaceGR1T2SceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: PickPlaceGR1T2ObservationsCfg = PickPlaceGR1T2ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (11), right hand joint pos (11)]
    idle_action = [
        -0.22878,
        0.2536,
        1.0953,
        0.5,
        -0.5,
        0.5,
        0.5,
        0.22878,
        0.2536,
        1.0953,
        0.5,
        -0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # visualizer camera settings
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2
        self.num_rerenders_on_reset = 3

        # Defer USD→URDF conversion to controller initialization (requires Isaac Sim at runtime).
        self.actions.upper_body_ik.controller.usd_path = self.scene.robot.spawn.usd_path
        self.actions.upper_body_ik.controller.urdf_output_dir = self.temp_urdf_dir

        self.image_obs_list = ["robot_pov_cam"]

        if ISAACLAB_TELEOP_AVAILABLE:
            from isaaclab_teleop import (
                GloveHapticFeedbackCfg,
                IsaacTeleopCfg,
                XrCameraFeedCfg,
                XrCameraFeedLayoutCfg,
                XrCfg,
            )

            # Isaac Capture-based teleoperation pipeline.
            self.xr = XrCfg(
                anchor_pos=(0.0, 0.0, 0.0),
                anchor_rot=(0.0, 0.0, 0.0, 1.0),
                # The shared anchor places the default eye inside the robot head.
                # Exclude visual meshes from the eye partition, not source visibility.
                excluded_visual_meshes=["/World/envs/env_0/Robot/head_.*_link/visuals/.*"],
            )
            self.isaac_teleop = IsaacTeleopCfg(
                pipeline_builder=lambda: build_gr1t2_pickplace_pipeline()[0],
                sim_device=self.sim.device,
                xr_cfg=self.xr,
                xr_camera_feeds=[
                    XrCameraFeedCfg(
                        camera_name="robot_pov_cam",
                        enable_dlss_ray_reconstruction=True,
                        dlss_exec_mode="quality",
                        offset_m=(0.0, -0.15),
                        max_update_hz=0.0,
                    )
                ],
                xr_camera_feed_layout=XrCameraFeedLayoutCfg(
                    placement="world",
                    world_position_m=(0.0, 0.8, 1.6),
                    world_orientation_xyzw=(0.70710678, 0.0, 0.0, 0.70710678),
                ),
            )

            # Per-finger haptic glove feedback: vibrate each finger of the operator's
            # glove in proportion to how tightly it grips the object. The session
            # always requests the push-tensor extension the glove device needs, so
            # this stays inert (no glove connected) rather than failing.
            self.haptic_feedback = GloveHapticFeedbackCfg(
                left_sensor_name="left_hand_contact",
                right_sensor_name="right_hand_contact",
            )

    def stabilize_for_kitless_ovphysx(self) -> None:
        """Configure the GR1T2 asset and action boundaries for kitless OVPhysX."""
        action_cfg = self.actions.upper_body_ik
        action_cfg.controller.urdf_path = os.path.join(
            os.path.dirname(__file__), "assets", "GR1T2_fourier_hand_6dof_kinematics.urdf"
        )
        action_cfg.kitless_ovphysx_enabled = True
        # Do not apply retargeter defaults before hand tracking starts.
        self.isaac_teleop.require_initial_bimanual_hand_tracking = True
        _apply_kitless_ovphysx_drive_gains(self.scene.robot)
        _condition_kitless_ovphysx_robot_spawn_properties(self.scene.robot)
        _condition_kitless_ovphysx_gr1t2_stage(self.scene.robot, action_cfg)
        condition_packing_table_for_kitless_ovphysx(self.scene.packing_table)

    def stabilize_for_kitless_newton_mjwarp(self) -> None:
        """Configure GR1T2 actuators and contacts for kitless Newton."""
        action_cfg = self.actions.upper_body_ik
        action_cfg.controller.urdf_path = os.path.join(
            os.path.dirname(__file__), "assets", "GR1T2_fourier_hand_6dof_kinematics.urdf"
        )
        action_cfg.kitless_ovphysx_enabled = True
        self.isaac_teleop.require_initial_bimanual_hand_tracking = True
        _stabilize_gr1t2_for_kitless_newton_mjwarp(self)
        _condition_kitless_ovphysx_gr1t2_stage(self.scene.robot, action_cfg, widen_hand_limits=False)
        _wrap_packing_table_collision_for_newton(self.scene.packing_table)


def _stabilize_gr1t2_for_kitless_newton_mjwarp(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Apply Newton actuator, allocation-budget, and wheel-collision preparation."""
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg

    # Match the kitless OVPhysX task: internal finger contacts must not force
    # compliant thumb drives away from their commands. Object contacts remain.
    fragments = env_cfg.scene.robot.spawn.articulation_props
    for fragment in fragments if isinstance(fragments, list) else [fragments]:
        if isinstance(fragment, NewtonArticulationCfg):
            fragment.self_collision_enabled = False

    groups = env_cfg.scene.robot.actuators
    contracts = {
        "head": (["head_.*"], 80.0, 8.0, 0.03, 80.0),
        "trunk": (["waist_.*"], 500.0, 50.0, 0.10, 250.0),
        "legs": ([".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"], 300.0, 30.0, 0.125, 250.0),
        "right-arm": (["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"], 1200.0, 60.0, 0.30, 800.0),
        "left-arm": (["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"], 1200.0, 60.0, 0.30, 800.0),
    }
    for name, (joint_names, stiffness, damping, armature, effort_limit) in contracts.items():
        actuator = groups.get(name)
        if actuator is None:
            actuator = ImplicitActuatorCfg(joint_names_expr=joint_names)
            groups[name] = actuator
        actuator.joint_names_expr = joint_names
        actuator.stiffness = stiffness
        actuator.damping = damping
        actuator.armature = armature
        actuator.effort_limit_sim = effort_limit

    for side, name in (("L", "left-hand"), ("R", "right-hand")):
        # The action conditions all eleven targets to the asset's mimic
        # relation. Driving the whole hand gives Newton the same compliant
        # hand targets as the OVPhysX path.
        hand = groups[name]
        hand.joint_names_expr = [f"{side}_.*"]
        hand.stiffness = 5.0
        hand.damping = 0.5
        hand.armature = 0.02
        hand.effort_limit_sim = 20.0
        groups.pop(f"{name}-slaved", None)

    solver_cfg = env_cfg.sim.physics.solver_cfg
    solver_cfg.nconmax = 4096
    solver_cfg.njmax = 1536
    _condition_newton_steering_wheel_detail_collisions(env_cfg.scene.object)


def _wrap_packing_table_collision_for_newton(asset_cfg: AssetBaseCfg) -> None:
    """Use the shared USD collision preparation for the packing-table asset."""
    from isaaclab_tasks.contrib.packing_table import configure_packing_table_collisions

    configure_packing_table_collisions(asset_cfg)


def _apply_kitless_ovphysx_drive_gains(robot_cfg: ArticulationCfg) -> None:
    """Restore stable actuator coverage around the Kit-tuned arm drives."""
    from isaaclab.actuators import ImplicitActuatorCfg

    groups = {
        "head": (["head_.*"], 80.0, 8.0, 0.03, 80.0),
        "legs": ([".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"], 300.0, 30.0, 0.10, 250.0),
        "right-hand": (None, 5.0, 0.5, 0.001, 20.0),
        "left-hand": (None, 5.0, 0.5, 0.001, 20.0),
    }
    for group_name, (joint_expr, stiffness, damping, armature, effort_limit) in groups.items():
        actuator = robot_cfg.actuators.get(group_name)
        if actuator is None:
            assert joint_expr is not None
            actuator = ImplicitActuatorCfg(joint_names_expr=list(joint_expr))
            robot_cfg.actuators[group_name] = actuator
        actuator.stiffness = stiffness
        actuator.damping = damping
        actuator.armature = armature
        actuator.effort_limit_sim = effort_limit

    if robot_cfg.spawn is not None and robot_cfg.spawn.articulation_props is not None:
        fragments = robot_cfg.spawn.articulation_props
        for fragment in fragments if isinstance(fragments, list) else [fragments]:
            if isinstance(fragment, PhysxArticulationCfg):
                fragment.enabled_self_collisions = False


def _condition_kitless_ovphysx_robot_spawn_properties(robot_cfg: ArticulationCfg) -> None:
    """Limit robot contact depenetration for kitless OVPhysX."""
    if robot_cfg.spawn is not None and robot_cfg.spawn.rigid_props is not None:
        fragments = robot_cfg.spawn.rigid_props
        for fragment in fragments if isinstance(fragments, list) else [fragments]:
            if isinstance(fragment, PhysxRigidBodyCfg):
                fragment.max_depenetration_velocity = 0.1


def _author_kitless_ovphysx_contact_offsets(prim) -> bool:
    """Author Kit-parity offsets on a collision prim before OVPhysX parses it."""
    from pxr import Sdf

    schemas = prim.GetMetadata("apiSchemas")
    applied = [str(schema) for schema in (schemas.GetAddedOrExplicitItems() if schemas else [])]
    if not any(schema.startswith("PhysicsCollisionAPI") for schema in applied):
        return False
    if not any(schema.startswith("PhysxCollisionAPI") for schema in applied):
        list_op = Sdf.TokenListOp()
        list_op.explicitItems = [*applied, "PhysxCollisionAPI"]
        prim.SetMetadata("apiSchemas", list_op)
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    return True


def _condition_kitless_ovphysx_gr1t2_stage(
    robot_cfg: ArticulationCfg,
    action_cfg: GR1T2KitlessPinkInverseKinematicsActionCfg,
    *,
    widen_hand_limits: bool = True,
) -> None:
    """Condition the spawned GR1T2 hand description before kitless physics parses it.

    The raw asset has near-singular finger inertia, hard PhysX-only mimic
    constraints, and hand rest poses exactly on physical limits. Those inputs
    are stable after Kit's asset preparation but are ill-conditioned when read
    directly by OVPhysX. This spawn boundary floors inertia, removes the hard
    constraints that conflict with direct hand drives, and optionally keeps the
    physical limits inactive for OVPhysX. Newton retains the anatomical limits
    because bounding drive targets alone does not bound simulated positions. The
    original mimic relations and anatomical limits are recorded on the typed
    action config for target conditioning. The source USD and normal Kit path
    remain unchanged.
    """
    if robot_cfg.spawn is None or getattr(robot_cfg.spawn, "_kitless_ovphysx_doctored", False):
        return
    original_spawn = robot_cfg.spawn.func

    def _spawn_and_condition(prim_path, cfg, *args, **kwargs):
        import math

        from pxr import Gf, Sdf, Usd

        prim = original_spawn(prim_path, cfg, *args, **kwargs)
        action_cfg.hand_joint_limits.clear()
        action_cfg.hand_mimic_relations.clear()

        # Collision meshes below instanceable ``collisions`` scopes are
        # immutable proxies. De-instance those scopes so the parser-visible
        # collision prims can receive the OVPhysX contact offsets.
        for child in Usd.PrimRange(prim):
            if child.GetName() == "collisions" and child.IsInstanceable():
                child.SetInstanceable(False)

        for child in Usd.PrimRange(prim):
            _author_kitless_ovphysx_contact_offsets(child)
            inertia = child.GetAttribute("physics:diagonalInertia")
            if inertia and inertia.HasValue():
                value = inertia.Get()
                if min(value) < 1.0e-4:
                    inertia.Set(Gf.Vec3f(*(max(component, 1.0e-4) for component in value)))

            is_hand_joint = child.GetName()[:2] in ("L_", "R_") and child.GetTypeName() == "PhysicsRevoluteJoint"
            schemas = child.GetMetadata("apiSchemas")
            applied = list(schemas.GetAddedOrExplicitItems()) if schemas else []
            kept = [schema for schema in applied if not str(schema).startswith("PhysxMimicJointAPI")]
            if len(kept) != len(applied):
                gearing = child.GetAttribute("physxMimicJoint:rotZ:gearing")
                offset = child.GetAttribute("physxMimicJoint:rotZ:offset")
                reference = child.GetRelationship("physxMimicJoint:rotZ:referenceJoint")
                targets = reference.GetTargets() if reference else []
                if gearing and gearing.HasValue() and targets:
                    action_cfg.hand_mimic_relations[child.GetName()] = (
                        targets[0].name,
                        float(gearing.Get()),
                        float(offset.Get()) if offset and offset.HasValue() else 0.0,
                    )
                list_op = Sdf.TokenListOp()
                list_op.explicitItems = kept
                child.SetMetadata("apiSchemas", list_op)
                for property_name in [name for name in child.GetPropertyNames() if name.startswith("physxMimicJoint:")]:
                    if not child.RemoveProperty(property_name):
                        mimic_attr = child.GetAttribute(property_name)
                        if mimic_attr and mimic_attr.GetTypeName() == Sdf.ValueTypeNames.Float:
                            mimic_attr.Set(0.0)

            if is_hand_joint:
                lower = child.GetAttribute("physics:lowerLimit")
                upper = child.GetAttribute("physics:upperLimit")
                if lower and upper and lower.HasValue() and upper.HasValue():
                    action_cfg.hand_joint_limits[child.GetName()] = (
                        math.radians(float(lower.Get())),
                        math.radians(float(upper.Get())),
                    )
                    if widen_hand_limits:
                        lower.Set(-720.0)
                        upper.Set(720.0)
        return prim

    robot_cfg.spawn.func = _spawn_and_condition
    robot_cfg.spawn._kitless_ovphysx_doctored = True
