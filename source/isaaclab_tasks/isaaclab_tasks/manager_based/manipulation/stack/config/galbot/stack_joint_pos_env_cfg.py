# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import logging

from isaaclab_physx.assets import SurfaceGripperCfg

try:
    import isaacteleop  # noqa: F401  -- pipeline builders need isaacteleop at runtime
    from isaaclab_teleop import IsaacTeleopCfg

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation features will be disabled.")

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp.actions.actions_cfg import SurfaceGripperBinaryActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import ObservationsCfg, StackEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.galbot import GALBOT_ONE_CHARLIE_CFG  # isort: skip


def _build_se3_abs_gripper_pipeline(hand_side="left"):
    """Build an IsaacTeleop Se3Abs + Gripper pipeline for single-arm manipulator teleoperation.

    Creates a Se3AbsRetargeter for end-effector absolute pose tracking and
    a GripperRetargeter for pinch-based gripper control from hand tracking data.
    All outputs are flattened into a single 8D action tensor via TensorReorderer.
    """
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    hand_key = HandsSource.LEFT if hand_side == "left" else HandsSource.RIGHT

    # SE3 Absolute Pose Retargeter
    se3_cfg = Se3RetargeterConfig(
        input_device=hand_key,
        zero_out_xy_rotation=True,
        use_wrist_rotation=False,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    se3 = Se3AbsRetargeter(se3_cfg, name="ee_pose")
    connected_se3 = se3.connect({hand_key: transformed_hands.output(hand_key)})

    # Gripper Retargeter (pinch-based)
    gripper_cfg = GripperRetargeterConfig(hand_side=hand_side)
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    controller_key = ControllersSource.LEFT if hand_side == "left" else ControllersSource.RIGHT
    connected_gripper = gripper.connect(
        {
            f"hand_{hand_side}": hands.output(hand_key),
            f"controller_{hand_side}": controllers.output(controller_key),
        }
    )

    # TensorReorderer: flatten into an 8D action tensor [ee_pose(7), gripper(1)]
    ee_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    gripper_elements = ["gripper_cmd"]

    reorderer = TensorReorderer(
        input_config={
            "ee_pose": ee_elements,
            "gripper": gripper_elements,
        },
        output_order=ee_elements + gripper_elements,
        name="action_reorderer",
        input_types={
            "ee_pose": "array",
            "gripper": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_pose": connected_se3.output("ee_pose"),
            "gripper": connected_gripper.output("gripper_command"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.0),
                "y": (0.20, 0.40),
                "z": (0.0203, 0.0203),
                "yaw": (-1.0, 1.0, 0.0),
            },
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class ObservationGalbotLeftArmGripperCfg:
    """Observations for the Galbot Left Arm Gripper."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        object = ObsTerm(
            func=mdp.object_abs_obs_in_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        cube_positions = ObsTerm(
            func=mdp.cube_poses_in_base_frame, params={"robot_cfg": SceneEntityCfg("robot"), "return_key": "pos"}
        )
        cube_orientations = ObsTerm(
            func=mdp.cube_poses_in_base_frame, params={"robot_cfg": SceneEntityCfg("robot"), "return_key": "quat"}
        )

        eef_pos = ObsTerm(
            func=mdp.ee_frame_pose_in_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "return_key": "pos",
            },
        )
        eef_quat = ObsTerm(
            func=mdp.ee_frame_pose_in_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "return_key": "quat",
            },
        )
        gripper_pos = ObsTerm(
            func=mdp.gripper_pos,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObservationsCfg.SubtaskCfg):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            super().__post_init__()

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()


@configclass
class GalbotLeftArmCubeStackEnvCfg(StackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # MDP settings

        # Set events
        self.events = EventCfg()
        self.observations.policy = ObservationGalbotLeftArmGripperCfg().PolicyCfg()
        self.observations.subtask_terms = ObservationGalbotLeftArmGripperCfg().SubtaskCfg()

        # Set galbot as robot
        self.scene.robot = GALBOT_ONE_CHARLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (galbot)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["left_arm_joint.*"], scale=0.5, use_default_offset=True
        )
        # Enable Parallel Gripper
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_gripper_.*_joint"],
            open_command_expr={"left_gripper_.*_joint": 0.035},
            close_command_expr={"left_gripper_.*_joint": 0.0},
        )
        self.gripper_joint_names = ["left_gripper_.*_joint"]
        self.gripper_open_val = 0.035
        self.gripper_threshold = 0.010

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        cube_collision_properties = CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)

        # Set each stacking cube deterministically
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                collision_props=cube_collision_properties,
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                collision_props=cube_collision_properties,
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                collision_props=cube_collision_properties,
            ),
        )

        # Listens to the required transforms
        self.marker_cfg = FRAME_MARKER_CFG.copy()
        self.marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_gripper_tcp_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

        # IsaacTeleop-based teleoperation pipeline (left hand)
        if _TELEOP_AVAILABLE:
            pipeline = _build_se3_abs_gripper_pipeline(hand_side="left")
            self.isaac_teleop = IsaacTeleopCfg(
                pipeline_builder=lambda: pipeline,
                sim_device=self.sim.device,
                xr_cfg=self.xr,
            )


@configclass
class GalbotRightArmCubeStackEnvCfg(GalbotLeftArmCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Move to area below right hand (invert y-axis)
        left, right = self.events.randomize_cube_positions.params["pose_range"]["y"]
        self.events.randomize_cube_positions.params["pose_range"]["y"] = (-right, -left)

        # Set actions for the specific robot type (galbot)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["right_arm_joint.*"], scale=0.5, use_default_offset=True
        )

        # Set surface gripper: Ensure the SurfaceGripper prim has the required attributes
        self.scene.surface_gripper = SurfaceGripperCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_suction_cup_tcp_link/SurfaceGripper",
            max_grip_distance=0.0075,
            shear_force_limit=5000.0,
            coaxial_force_limit=5000.0,
            retry_interval=0.05,
        )

        # Set surface gripper action
        self.actions.gripper_action = SurfaceGripperBinaryActionCfg(
            asset_name="surface_gripper",
            open_command=-1.0,
            close_command=1.0,
        )

        self.scene.ee_frame.target_frames[0].prim_path = "{ENV_REGEX_NS}/Robot/right_suction_cup_tcp_link"

        # IsaacTeleop-based teleoperation pipeline (right hand)
        if _TELEOP_AVAILABLE:
            pipeline = _build_se3_abs_gripper_pipeline(hand_side="right")
            self.isaac_teleop = IsaacTeleopCfg(
                pipeline_builder=lambda: pipeline,
                sim_device=self.sim.device,
                xr_cfg=self.xr,
            )
