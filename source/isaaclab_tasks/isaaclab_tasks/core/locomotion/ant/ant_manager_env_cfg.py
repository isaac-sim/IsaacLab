# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
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
from isaaclab.utils import config_field, replace_config

import isaaclab_tasks.core.locomotion.mdp as mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.ant import ANT_CFG


@dataclass
class AntPhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = config_field(PhysxCfg(bounce_threshold_velocity=0.2))
    ovphysx: OvPhysxCfg = config_field(OvPhysxCfg())
    physx: PhysxAutoCfg = config_field(PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx))
    newton_mjwarp: NewtonCfg = config_field(
        NewtonCfg(
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
    newton_kamino: NewtonCfg = config_field(
        NewtonCfg(
            solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
            debug_mode=False,
            use_cuda_graph=True,
        )
    )
    default: NewtonCfg = config_field(newton_mjwarp)


@dataclass
class AntSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an ant robot."""

    # terrain
    terrain: Any = config_field(
        TerrainImporterCfg(
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
    robot: Any = config_field(replace_config(ANT_CFG, prim_path="{ENV_REGEX_NS}/Robot"))

    # sensors
    joint_wrench: Any = config_field(JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot"))

    # lights
    light: Any = config_field(
        AssetBaseCfg(
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
    joint_effort: Any = config_field(
        mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5, clip={".*": (-7.5, 7.5)})
    )


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height: Any = config_field(ObsTerm(func=mdp.base_pos_z))
        base_lin_vel: Any = config_field(ObsTerm(func=mdp.base_lin_vel))
        base_ang_vel: Any = config_field(ObsTerm(func=mdp.base_ang_vel))
        base_yaw_roll: Any = config_field(ObsTerm(func=mdp.base_yaw_roll))
        base_angle_to_target: Any = config_field(
            ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        )
        base_up_proj: Any = config_field(ObsTerm(func=mdp.base_up_proj))
        base_heading_proj: Any = config_field(
            ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        )
        joint_pos_norm: Any = config_field(ObsTerm(func=mdp.joint_pos_limit_normalized))
        joint_vel_rel: Any = config_field(ObsTerm(func=mdp.joint_vel_rel, scale=0.2))
        feet_body_forces: Any = config_field(
            ObsTerm(
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
        actions: Any = config_field(ObsTerm(func=mdp.last_action))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = config_field(PolicyCfg())


@dataclass
class AntObservationsCfg(PresetCfg):
    physx: ObservationsCfg = config_field(ObservationsCfg())
    isaacsim_physx: ObservationsCfg = config_field(physx)
    newton_mjwarp: ObservationsCfg = config_field(ObservationsCfg())
    default: ObservationsCfg = config_field(newton_mjwarp)


@dataclass
class EventCfg:
    """Configuration for events."""

    reset_base: Any = config_field(
        EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        )
    )

    reset_robot_joints: Any = config_field(
        EventTerm(
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
    progress: Any = config_field(
        RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    )
    # (2) Stay alive bonus
    alive: Any = config_field(RewTerm(func=mdp.is_alive, weight=0.5))
    # (3) Reward for upright posture
    upright: Any = config_field(RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93}))
    # (4) Reward for moving in the right direction
    move_to_target: Any = config_field(
        RewTerm(func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)})
    )
    # (5) Penalty for large action commands
    action_l2: Any = config_field(RewTerm(func=mdp.action_l2, weight=-0.005))
    # (6) Penalty for energy consumption
    energy: Any = config_field(RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}}))
    # (7) Penalty for reaching close to joint limits
    joint_pos_limits: Any = config_field(
        RewTerm(
            func=mdp.joint_pos_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.99, "gear_ratio": {".*": 15.0}}
        )
    )
    # (8) Penalty for falling over, applied once on the terminating step
    terminating: Any = config_field(RewTerm(func=mdp.terminated_penalty, weight=-2.0))
    # (9) Survival rate metric (logged only, contributes no reward)
    success_rate: Any = config_field(RewTerm(func=mdp.survival_success_rate, weight=0.0))


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out: Any = config_field(DoneTerm(func=mdp.time_out, time_out=True))
    # (2) Terminate if the robot falls
    torso_height: Any = config_field(DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.31}))


@dataclass
class AntEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Ant walking environment."""

    # Scene settings
    scene: AntSceneCfg = config_field(AntSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True))
    # Basic settings
    observations: AntObservationsCfg = config_field(AntObservationsCfg())
    actions: ActionsCfg = config_field(ActionsCfg())
    # MDP settings
    rewards: RewardsCfg = config_field(RewardsCfg())
    terminations: TerminationsCfg = config_field(TerminationsCfg())
    events: EventCfg = config_field(EventCfg())

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
