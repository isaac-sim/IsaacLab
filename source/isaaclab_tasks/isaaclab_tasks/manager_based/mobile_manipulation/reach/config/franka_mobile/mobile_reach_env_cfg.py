# Copyright (c) 2022-2025, Elevate Robotics
# All rights reserved.

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTermCfg as CommandTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Scene definition
##

@configclass
class MobileReachSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with a mobile manipulator."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Mobile base with arm - to be populated by specific robot config
    robot: ArticulationCfg = MISSING

    # End-effector frames - to be populated by specific robot config
    ee_frame: FrameTransformerCfg = MISSING

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # Set by specific robot config
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 2.0),    # Larger workspace due to mobile base
            pos_y=(-1.0, 1.0),
            pos_z=(0.3, 1.0),    # Keep targets above ground
            roll=(0.0, 0.0),     # Start with just position control
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Will be set by specific robot config
    arm_action = MISSING
    base_action = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Base state
        base_pos = ObsTerm(func=mdp.base_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_vel = ObsTerm(func=mdp.base_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_rpy = ObsTerm(func=mdp.base_rpy, noise=Unoise(n_min=-0.01, n_max=0.01))

        # Arm state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        # Task state
        ee_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task objective
    ee_pos_tracking = RewTerm(
        func=mdp.ee_pos_tracking_error,
        weight=1.0,
        params={"command_name": "ee_pose", "std": 0.5}
    )

    # Costs/regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-0.0001)
    base_acc = RewTerm(func=mdp.base_acc_l2, weight=-0.0001)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

##
# Environment configuration
##

@configclass
class MobileReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the mobile reach environment."""

    # Scene settings
    scene: MobileReachSceneCfg = MobileReachSceneCfg(num_envs=4096, env_spacing=4.0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 10.0

        # Simulation settings
        self.sim.dt = 0.02  # 50 Hz
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False

        # Viewer settings
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)