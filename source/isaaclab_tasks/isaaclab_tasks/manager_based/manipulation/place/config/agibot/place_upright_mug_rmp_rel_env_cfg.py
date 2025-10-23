# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp
from isaaclab_tasks.manager_based.manipulation.place.config.agibot import place_toy2box_rmp_rel_env_cfg
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.agibot import AGIBOT_A2D_CFG  # isort: skip
from isaaclab.controllers.config.rmp_flow import AGIBOT_LEFT_ARM_RMPFLOW_CFG  # isort: skip

##
# Event settings
##


@configclass
class EventCfgPlaceUprightMug:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    randomize_mug_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.2),
                "y": (-0.10, 0.10),
                "z": (0.75, 0.75),
                "roll": (-1.57, -1.57),
                "yaw": (-0.57, 0.57),
            },
            "asset_cfgs": [SceneEntityCfg("mug")],
        },
    )


#
# MDP settings
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        mug_positions = ObsTerm(
            func=place_mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("mug"), "return_key": "pos"}
        )
        mug_orientations = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("mug"), "return_key": "quat"},
        )
        eef_pos = ObsTerm(func=mdp.ee_frame_pose_in_base_frame, params={"return_key": "pos"})
        eef_quat = ObsTerm(func=mdp.ee_frame_pose_in_base_frame, params={"return_key": "quat"})
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp = ObsTerm(
            func=place_mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("mug"),
                "diff_threshold": 0.05,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    mug_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.85, "asset_cfg": SceneEntityCfg("mug")}
    )

    success = DoneTerm(
        func=place_mdp.object_placed_upright,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("mug"),
            "target_height": 0.6,
        },
    )


"""
Env to Place Upright Mug with AgiBot Left Arm using RMPFlow
"""


class RmpFlowAgibotPlaceUprightMugEnvCfg(place_toy2box_rmp_rel_env_cfg.PlaceToy2BoxEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events = EventCfgPlaceUprightMug()

        # Set Agibot as robot
        self.scene.robot = AGIBOT_A2D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (-0.60, 0.0, 0.0)

        # reset obs and termination terms
        self.observations = ObservationsCfg()
        self.terminations = TerminationsCfg()

        # Table
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.50, 0.0, 0.60], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                scale=(1.0, 1.0, 0.60),
            ),
        )

        # add contact force sensor for grasped checking
        self.scene.contact_grasp = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_.*_Pad_Link",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Mug"],
        )

        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (Agibot)
        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint.*"],
            body_name="gripper_center",
            controller=AGIBOT_LEFT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0], rot=[0.7071, 0.0, -0.7071, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )

        # Enable Parallel Gripper:
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_hand_joint1", "left_.*_Support_Joint"],
            open_command_expr={"left_hand_joint1": 0.994, "left_.*_Support_Joint": 0.994},
            close_command_expr={"left_hand_joint1": 0.0, "left_.*_Support_Joint": 0.0},
        )

        # find joint ids for grippers
        self.gripper_joint_names = ["left_hand_joint1", "left_Right_1_Joint"]
        self.gripper_open_val = 0.994
        self.gripper_threshold = 0.2

        # Rigid body properties of mug
        mug_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.mug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Mug",
            init_state=RigidObjectCfg.InitialStateCfg(),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=mug_properties,
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
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_center",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                        rot=[
                            0.7071,
                            0.0,
                            -0.7071,
                            0.0,
                        ],
                    ),
                ),
            ],
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        # Set the simulation parameters
        self.sim.dt = 1 / 60
        self.sim.render_interval = 6

        self.decimation = 3
        self.episode_length_s = 10.0
