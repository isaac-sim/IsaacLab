# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import replace_config

import isaaclab_tasks.core.locomotion.mdp as mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.ant import ANT_CFG


@dataclass
class AntPhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = field(default_factory=lambda: PhysxCfg(bounce_threshold_velocity=0.2))
    ovphysx: OvPhysxCfg = field(default_factory=OvPhysxCfg)
    physx: PhysxAutoCfg = field(
        default_factory=lambda: PhysxAutoCfg(
            isaacsim_physx=PhysxCfg(bounce_threshold_velocity=0.2), ovphysx=OvPhysxCfg()
        )
    )
    newton_mjwarp: NewtonCfg = field(
        default_factory=lambda: NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=45,
                nconmax=25,
                cone="pyramidal",
                integrator="implicitfast",
                impratio=1,
            ),
            num_substeps=1,
            debug_mode=False,
        )
    )
    newton_kamino: NewtonCfg = field(
        default_factory=lambda: NewtonCfg(
            solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
            debug_mode=False,
            use_cuda_graph=True,
        )
    )
    default: NewtonCfg = field(
        default_factory=lambda: NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=45,
                nconmax=25,
                cone="pyramidal",
                integrator="implicitfast",
                impratio=1,
            ),
            num_substeps=1,
            debug_mode=False,
        )
    )


@dataclass
class AntSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an ant robot."""

    # terrain
    terrain: Any = field(
        default_factory=lambda: TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
    )

    # robot
    robot: Any = field(default_factory=lambda: replace_config(ANT_CFG, prim_path="{ENV_REGEX_NS}/Robot"))

    # sensors
    joint_wrench: Any = field(default_factory=lambda: JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot"))

    # lights
    light: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
    )


##
# MDP settings
##


@dataclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # the effort is clipped at the gear magnitude, i.e. to a unit action: unbounded joint efforts
    # drive the solver to NaN
    joint_effort: Any = field(
        default_factory=lambda: mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=[".*"], scale=7.5, clip={".*": (-7.5, 7.5)}
        )
    )


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height: Any = field(default_factory=lambda: ObsTerm(func=mdp.base_pos_z))
        base_lin_vel: Any = field(default_factory=lambda: ObsTerm(func=mdp.base_lin_vel))
        base_ang_vel: Any = field(default_factory=lambda: ObsTerm(func=mdp.base_ang_vel))
        base_yaw_roll: Any = field(default_factory=lambda: ObsTerm(func=mdp.base_yaw_roll))
        base_angle_to_target: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        )
        base_up_proj: Any = field(default_factory=lambda: ObsTerm(func=mdp.base_up_proj))
        base_heading_proj: Any = field(
            default_factory=lambda: ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        )
        joint_pos_norm: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_pos_limit_normalized))
        joint_vel_rel: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_vel_rel, scale=0.2))
        feet_body_forces: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.body_incoming_wrench,
                scale=0.1,
                params={
                    "sensor_cfg": SceneEntityCfg(
                        "joint_wrench",
                        body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"],
                    )
                },
            )
        )
        actions: Any = field(default_factory=lambda: ObsTerm(func=mdp.last_action))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = field(default_factory=PolicyCfg)


@dataclass
class AntObservationsCfg(PresetCfg):
    physx: ObservationsCfg = field(default_factory=ObservationsCfg)
    isaacsim_physx: ObservationsCfg = field(default_factory=ObservationsCfg)
    newton_mjwarp: ObservationsCfg = field(default_factory=ObservationsCfg)
    default: ObservationsCfg = field(default_factory=ObservationsCfg)


@dataclass
class EventCfg:
    """Configuration for events."""

    reset_base: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        )
    )

    reset_robot_joints: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.2, 0.2),
                "velocity_range": (-0.1, 0.1),
            },
        )
    )


@dataclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress: Any = field(
        default_factory=lambda: RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    )
    # (2) Stay alive bonus
    alive: Any = field(default_factory=lambda: RewTerm(func=mdp.is_alive, weight=0.5))
    # (3) Reward for upright posture
    upright: Any = field(
        default_factory=lambda: RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    )
    # (4) Reward for moving in the right direction
    move_to_target: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
        )
    )
    # (5) Penalty for large action commands
    action_l2: Any = field(default_factory=lambda: RewTerm(func=mdp.action_l2, weight=-0.005))
    # (6) Penalty for energy consumption
    energy: Any = field(
        default_factory=lambda: RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}})
    )
    # (7) Penalty for reaching close to joint limits
    joint_pos_limits: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.joint_pos_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.99, "gear_ratio": {".*": 15.0}}
        )
    )
    # (8) Penalty for falling over, applied once on the terminating step
    terminating: Any = field(default_factory=lambda: RewTerm(func=mdp.terminated_penalty, weight=-2.0))
    # (9) Survival rate metric (logged only, contributes no reward)
    success_rate: Any = field(default_factory=lambda: RewTerm(func=mdp.survival_success_rate, weight=0.0))


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out: Any = field(default_factory=lambda: DoneTerm(func=mdp.time_out, time_out=True))
    # (2) Terminate if the robot falls
    torso_height: Any = field(
        default_factory=lambda: DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.31})
    )


@dataclass
class AntEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Ant walking environment."""

    # Scene settings
    scene: AntSceneCfg = field(
        default_factory=lambda: AntSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)
    )
    # Basic settings
    observations: AntObservationsCfg = field(default_factory=AntObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    # MDP settings
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    events: EventCfg = field(default_factory=EventCfg)

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = AntPhysicsCfg()
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
