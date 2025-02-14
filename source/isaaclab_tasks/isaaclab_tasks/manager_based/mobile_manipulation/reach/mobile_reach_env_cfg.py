# Copyright (c) 2022-2025, Elevate Robotics
# All rights reserved.

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.mobile_manipulation.reach.mdp as mdp

##
# Scene definition
##


@configclass
class MobileReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a mobile manipulator."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Robot - will be populated by specific robot configs
    robot: ArticulationCfg = MISSING

    # End-effector sensor - will be populated by specific robot configs
    # ee_frame: FrameTransformerCfg = MISSING

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
        body_name=MISSING,  # set by robot-specific config
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),  # Larger workspace due to mobile base
            pos_y=(-1.0, 1.0),
            pos_z=(0.3, 0.7),  # Reasonable heights for manipulation
            roll=(0.0, 0.0),  # Keep end-effector upright for now
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Will be set by robot-specific config
    arm_action: ActionTerm = MISSING
    base_action: ActionTerm = MISSING


def base_yaw(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Base orientation as yaw angle."""
    asset: RigidObject = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return yaw.unsqueeze(-1)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Arm state
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )

        # Base state
        base_pos = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_yaw = ObsTerm(func=base_yaw, noise=Unoise(n_min=-0.01, n_max=0.01))

        # Task
        ee_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task terms - tracking end-effector target
    ee_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_link7"]),
            "command_name": "ee_pose",
        },
    )

    ee_position_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_link7"]),
            "std": 0.2,
            "command_name": "ee_pose",
        },
    )

    base_heading_to_target = RewTerm(
        func=mdp.base_heading_to_target,
        weight=0.0001,
        params={"command_name": "ee_pose"},
    )

    # Energy costs
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_vel = RewTerm(func=mdp.base_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Optional early termination on reaching goal
    success = DoneTerm(
        func=mdp.reached_goal,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_link7"]),
            "command_name": "ee_pose",
            "threshold": 0.05,
        },
    )


##
# Environment configuration
##


@configclass
class MobileReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the mobile reach environment."""

    # Scene settings
    scene: MobileReachSceneCfg = MobileReachSceneCfg(num_envs=4096, env_spacing=3.0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 2
        self.episode_length_s = 10.0

        # Simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

        # Viewer settings - wider view for mobile base
        self.viewer.eye = (5.0, 5.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
