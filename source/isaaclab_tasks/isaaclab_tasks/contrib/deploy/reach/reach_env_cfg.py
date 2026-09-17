# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import REQUIRED
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.deploy.mdp as mdp

##
# Scene definition
##


@dataclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )
    )

    # robots
    robot: ArticulationCfg = REQUIRED

    # lights
    light: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
        )
    )

    table: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
            ),
        )
    )


##
# MDP settings
##


@dataclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose: Any = field(
        default_factory=lambda: mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name=REQUIRED,
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.35, 0.65),
                pos_y=(-0.2, 0.2),
                pos_z=(0.15, 0.5),
                roll=(0.0, 0.0),
                pitch=REQUIRED,  # depends on end-effector axis
                yaw=(-3.14, 3.14),
            ),
        )
    )


@dataclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = REQUIRED
    gripper_action: ActionTerm | None = None


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0)))
        joint_vel: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0)))
        pose_command: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = field(default_factory=PolicyCfg)


@dataclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.125, 0.125),
                "velocity_range": (0.0, 0.0),
            },
        )
    )

    robot_joint_stiffness_and_damping: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.randomize_actuator_gains,
            min_step_count_between_reset=200,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.75, 1.5),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
    )

    joint_friction: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.randomize_joint_parameters,
            min_step_count_between_reset=200,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "friction_distribution_params": (0.0, 0.1),
                "operation": "add",
                "distribution": "uniform",
            },
        )
    )


@dataclass
class RewardsCfg:
    """Reward terms for the MDP."""

    end_effector_keypoint_tracking: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.keypoint_command_error,
            weight=-1.5,
            params={
                "asset_cfg": SceneEntityCfg("ee_frame"),
                "command_name": "ee_pose",
                "keypoint_scale": 0.45,
            },
        )
    )
    end_effector_keypoint_tracking_exp: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.keypoint_command_error_exp,
            weight=1.5,
            params={
                "asset_cfg": SceneEntityCfg("ee_frame"),
                "command_name": "ee_pose",
                "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001), (5000, 0.0001)],
                "kp_use_sum_of_exps": False,
                "keypoint_scale": 0.45,
            },
        )
    )

    action_rate: Any = field(default_factory=lambda: RewTerm(func=mdp.action_rate_l2, weight=-0.005))
    action: Any = field(default_factory=lambda: RewTerm(func=mdp.action_l2, weight=-0.005))


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = field(default_factory=lambda: DoneTerm(func=mdp.time_out, time_out=True))


##
# Environment configuration
##


@dataclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the end-effector pose tracking environment that has been deployed on a real robot."""

    # Scene settings
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(num_envs=4096, env_spacing=2.5))
    # Basic settings
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    commands: CommandsCfg = field(default_factory=CommandsCfg)
    # MDP settings
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    events: EventCfg = field(default_factory=EventCfg)

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(3.5, 3.5, 3.5))
        # simulation settings
        self.sim.dt = 1.0 / 120.0
