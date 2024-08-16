# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors import CameraCfg, TiledCameraCfg

from . import mdp

##
# Scene definition
##

MIN_HEIGHT_FOR_LIFT = 0.06 # 0.04 by default

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING
    # camera object: will be populated by agent env cfg
    camera: CameraCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), 
            pos_y=(-0.25, 0.25), 
            pos_z=(0.25, 0.5), 
            roll=(0.0, 0.0), 
            pitch=(0.0, 0.0), 
            yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        #object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        #target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        # camera
        cam_data = ObsTerm(func=mdp.rgb_camera, params={"sensor_cfg": SceneEntityCfg("camera")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True # default: True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # NOTE: Can be used to randomize the states or the robots attributes

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Rewards: ---------------------------------------------------------------------------------------------------------

    # Reaching
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=5.0)

    # Grasping
    # NOTE: EXPERIMENTAL
    approach_gripper_object = RewTerm(
        func=mdp.approach_gripper_object_to_grasp, 
        weight=5.0, 
        params={
            "offset": MISSING
        }
    )
    align_grasp_around_object = RewTerm(
        func=mdp.align_grasp_around_object, 
        weight=0.125
    )
    grasp_object = RewTerm(
        func=mdp.grasp_object,
        weight=10.0,
        params={
            "threshold": 0.04,
            "open_joint_pos": MISSING,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )

    # Lifting
    # lifting_object = RewTerm(
    #     func=mdp.object_is_lifted, 
    #     params={
    #         "minimal_height": MIN_HEIGHT_FOR_LIFT
    #     }, 
    #     weight=15.0,
    # )

    # Lift object above a certain height
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=5.0,
    # )

    # Penalties: -------------------------------------------------------------------------------------------------------
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "object_pose"},
    )

    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "object_pose"},
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # penalize staying in one place (not moving)
    # NOTE: EXPERIMENTAL
    # joint_vel_ee_stationary = RewTerm(
    #     func=mdp.is_ee_stationary,
    #     weight=-1e-4,
    #     params={
    #         "threshold": 0.01,
    #         "prev_joint_vel": MISSING,
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    # NOTE: EXPERIMENTAL
    succes = DoneTerm(
        func=mdp.is_object_is_lifted, params={"minimal_height": MIN_HEIGHT_FOR_LIFT}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # (1) reaching the object
    # NOTE: EXPERIMENTAL
    reaching_object = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={
            "term_name": "reaching_object", 
            "weight": 1.0,
            "num_steps": 1000 #10000
        })
    
    # (2) grasping the object
    # NOTE: EXPERIMENTAL
    grasp_object = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={
            "term_name": "grasp_object", 
            "weight": 1.0,
            "num_steps": 5000
        })

    # (3) lifting the object
    # NOTE: EXPERIMENTAL
    object_goal_tracking = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={
            "term_name": "object_goal_tracking", 
            "weight": 1.0,
            "num_steps": 10000
        })
    
    # object_goal_tracking = CurrTerm(
    #     func=mdp.modify_reward_weight, 
    #     params={
    #         "term_name": "object_goal_tracking", 
    #         "weight": 1.0,
    #         "num_steps": 50000
    #     })

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        #self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
