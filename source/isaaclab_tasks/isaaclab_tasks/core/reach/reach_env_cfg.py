# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, dataclass
from typing import Any

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import config_field
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

##
# Physics backend presets
##


@dataclass
class ReachPhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = config_field(PhysxCfg(bounce_threshold_velocity=0.2))
    ovphysx: OvPhysxCfg = config_field(OvPhysxCfg())
    physx: PhysxAutoCfg = config_field(PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx))

    newton_mjwarp: NewtonCfg = config_field(
        NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=100,
                nconmax=20,
                cone="pyramidal",
                integrator="implicitfast",
                impratio=1,
                update_data_interval=2,
            ),
            num_substeps=2,
            debug_mode=False,
            use_cuda_graph=True,
        )
    )
    default: NewtonCfg = config_field(newton_mjwarp)


##
# Scene definition
##


@dataclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground: Any = config_field(
        AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )
    )

    table: Any = config_field(
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
            spawn=sim_utils.CuboidCfg(
                size=(0.9, 1.3, 1.0),
                collision_props=sim_utils.CollisionBaseCfg(),
            ),
        )
    )

    # robots
    robot: ArticulationCfg = config_field(MISSING)

    # lights
    light: Any = config_field(
        AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
        )
    )


##
# MDP settings
##


@dataclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose: Any = config_field(
        mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name=MISSING,
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            position_success_threshold=0.05,
            orientation_success_threshold=0.2,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(0.35, 0.65),
                pos_y=(-0.2, 0.2),
                pos_z=(0.15, 0.5),
                roll=(0.0, 0.0),
                pitch=MISSING,  # depends on end-effector axis
                yaw=(-3.14, 3.14),
            ),
        )
    )


@dataclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = config_field(MISSING)
    gripper_action: ActionTerm | None = config_field(None)


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos: Any = config_field(ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)))
        joint_vel: Any = config_field(ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)))
        pose_command: Any = config_field(ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"}))
        actions: Any = config_field(ObsTerm(func=mdp.last_action))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = config_field(PolicyCfg())


@dataclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints: Any = config_field(
        EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
            },
        )
    )


@dataclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking: Any = config_field(
        RewTerm(
            func=mdp.position_command_error,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
        )
    )
    end_effector_orientation_tracking: Any = config_field(
        RewTerm(
            func=mdp.orientation_command_error,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
        )
    )
    success: Any = config_field(RewTerm(func=mdp.is_terminated_term, weight=10.0, params={"term_keys": ["success"]}))

    # control and physical motion penalties
    action_rate: Any = config_field(RewTerm(func=mdp.action_rate_l2, weight=-0.0001))
    action_magnitude: Any = config_field(RewTerm(func=mdp.action_l2, weight=-0.005))
    joint_vel: Any = config_field(
        RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.0001,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    )


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    success: Any = config_field(
        DoneTerm(
            func=mdp.pose_command_success,
            params={"command_name": "ee_pose"},
        )
    )
    time_out: Any = config_field(DoneTerm(func=mdp.time_out, time_out=True))


@dataclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate: Any = config_field(
        CurrTerm(
            func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
        )
    )

    joint_vel: Any = config_field(
        CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500})
    )


##
# Environment configuration
##


@dataclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = config_field(ReachSceneCfg(num_envs=4096, env_spacing=2.5))
    # Basic settings
    observations: ObservationsCfg = config_field(ObservationsCfg())
    actions: ActionsCfg = config_field(ActionsCfg())
    commands: CommandsCfg = config_field(CommandsCfg())
    # MDP settings
    rewards: RewardsCfg = config_field(RewardsCfg())
    terminations: TerminationsCfg = config_field(TerminationsCfg())
    events: EventCfg = config_field(EventCfg())
    curriculum: CurriculumCfg = config_field(CurriculumCfg())

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(3.5, 3.5, 3.5))
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.physics = ReachPhysicsCfg()
