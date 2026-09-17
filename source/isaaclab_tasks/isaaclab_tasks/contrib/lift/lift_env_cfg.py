# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import REQUIRED
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.contrib.lift import mdp

##
# Scene definition
##


@dataclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = REQUIRED
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = REQUIRED
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = REQUIRED

    # table
    table: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        )
    )

    # plane
    plane: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
            spawn=GroundPlaneCfg(),
        )
    )

    # lights
    light: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
    )


##
# MDP settings
##


@dataclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose: Any = field(
        default_factory=lambda: mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name=REQUIRED,  # will be set by derived env cfg
            resampling_time_range=(5.0, 5.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.4, 0.6),
                pos_y=(-0.25, 0.25),
                pos_z=(0.25, 0.5),
                roll=(0.0, 0.0),
                pitch=(0.0, 0.0),
                yaw=(0.0, 0.0),
            ),
        )
    )


@dataclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by derived env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = REQUIRED
    gripper_action: mdp.BinaryJointPositionActionCfg = REQUIRED


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_pos_rel))
        joint_vel: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_vel_rel))
        object_position: Any = field(default_factory=lambda: ObsTerm(func=mdp.object_position_in_robot_root_frame))
        target_object_position: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        )
        actions: Any = field(default_factory=lambda: ObsTerm(func=mdp.last_action))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = field(default_factory=PolicyCfg)


@dataclass
class EventCfg:
    """Configuration for events."""

    reset_all: Any = field(default_factory=lambda: EventTerm(func=mdp.reset_scene_to_default, mode="reset"))

    reset_object_position: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            },
        )
    )


@dataclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object: Any = field(
        default_factory=lambda: RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    )

    lifting_object: Any = field(
        default_factory=lambda: RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)
    )

    object_goal_tracking: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.object_goal_distance,
            params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose", "success_threshold": 0.05},
            weight=16.0,
        )
    )

    object_goal_tracking_fine_grained: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.object_goal_distance,
            params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
            weight=5.0,
        )
    )

    # action penalty
    action_rate: Any = field(default_factory=lambda: RewTerm(func=mdp.action_rate_l2, weight=-1e-4))

    joint_vel: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.joint_vel_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    )


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = field(default_factory=lambda: DoneTerm(func=mdp.time_out, time_out=True))

    object_dropping: Any = field(
        default_factory=lambda: DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
        )
    )


@dataclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate: Any = field(
        default_factory=lambda: CurrTerm(
            func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
        )
    )

    joint_vel: Any = field(
        default_factory=lambda: CurrTerm(
            func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
        )
    )


##
# Environment configuration
##


@dataclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = field(default_factory=lambda: ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5))
    # Basic settings
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    commands: CommandsCfg = field(default_factory=CommandsCfg)
    # MDP settings
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    events: EventCfg = field(default_factory=EventCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
