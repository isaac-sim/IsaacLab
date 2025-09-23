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
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import ObjectTableSceneCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.agibot import AGIBOT_A2D_CFG  # isort: skip
from isaaclab.controllers.config.rmp_flow import AGIBOT_RIGHT_ARM_RMPFLOW_CFG  # isort: skip

##
# Event settings
##


@configclass
class EventCfgPlaceToy2Box:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    init_toy_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.15, 0.20),
                "y": (-0.3, -0.15),
                "z": (-0.65, -0.65),
                "yaw": (-3.14, 3.14),
            },
            "asset_cfgs": [SceneEntityCfg("toy_truck")],
        },
    )
    init_box_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.35),
                "y": (0.0, 0.10),
                "z": (-0.55, -0.55),
                "yaw": (-3.14, 3.14),
            },
            "asset_cfgs": [SceneEntityCfg("box")],
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
        toy_truck_positions = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("toy_truck"), "return_key": "pos"},
        )
        toy_truck_orientations = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("toy_truck"), "return_key": "quat"},
        )
        box_positions = ObsTerm(
            func=place_mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("box"), "return_key": "pos"}
        )
        box_orientations = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("box"), "return_key": "quat"},
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
                "object_cfg": SceneEntityCfg("toy_truck"),
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

    toy_truck_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.85, "asset_cfg": SceneEntityCfg("toy_truck")}
    )

    success = DoneTerm(
        func=place_mdp.object_a_is_into_b,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_a_cfg": SceneEntityCfg("toy_truck"),
            "object_b_cfg": SceneEntityCfg("box"),
            "xy_threshold": 0.10,
            "height_diff": 0.06,
            "height_threshold": 0.04,
        },
    )


@configclass
class PlaceToy2BoxEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""

        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # set viewer to see the whole scene
        self.viewer.eye = [1.5, -1.0, 1.5]
        self.viewer.lookat = [0.5, 0.0, 0.0]


"""
Env to Replay Sim2Lab Demonstrations with JointSpaceAction
"""


class RmpFlowAgibotPlaceToy2BoxEnvCfg(PlaceToy2BoxEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events = EventCfgPlaceToy2Box()

        # Set Agibot as robot
        self.scene.robot = AGIBOT_A2D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # add table
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.7], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                scale=(1.8, 1.0, 0.30),
            ),
        )

        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (Agibot)
        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            body_name="right_gripper_center",
            controller=AGIBOT_RIGHT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )

        # Enable Parallel Gripper:
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_hand_joint1", "right_.*_Support_Joint"],
            open_command_expr={"right_hand_joint1": 0.994, "right_.*_Support_Joint": 0.994},
            close_command_expr={"right_hand_joint1": 0.20, "right_.*_Support_Joint": 0.20},
        )

        # find joint ids for grippers
        self.gripper_joint_names = ["right_hand_joint1", "right_Right_1_Joint"]
        self.gripper_open_val = 0.994
        self.gripper_threshold = 0.2

        # Rigid body properties of toy_truck and box
        toy_truck_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        box_properties = toy_truck_properties.copy()

        # Notes: remember to add Physics/Mass properties to the toy_truck mesh to make grasping successful,
        # then you can use below MassPropertiesCfg to set the mass of the toy_truck
        toy_mass_properties = MassPropertiesCfg(
            mass=0.05,
        )

        self.scene.toy_truck = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ToyTruck",
            init_state=RigidObjectCfg.InitialStateCfg(),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/ToyTruck/toy_truck.usd",
                rigid_props=toy_truck_properties,
                mass_props=toy_mass_properties,
            ),
        )

        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Box/box.usd",
                rigid_props=box_properties,
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
                    prim_path="{ENV_REGEX_NS}/Robot/right_gripper_center",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

        # add contact force sensor for grasped checking
        self.scene.contact_grasp = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_.*_Pad_Link",
            update_period=0.05,
            history_length=6,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/ToyTruck"],
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
        self.episode_length_s = 30.0
