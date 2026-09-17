# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import dataclass, field

from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import replace_config
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.core.cartpole.mdp as mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from typing import Any

##
# Physics backend presets
##


@dataclass
class CartpolePhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = field(default_factory=PhysxCfg)
    ovphysx: OvPhysxCfg = field(default_factory=OvPhysxCfg)
    physx: PhysxAutoCfg = field(default_factory=lambda: PhysxAutoCfg(isaacsim_physx=PhysxCfg(), ovphysx=OvPhysxCfg()))
    newton_mjwarp: NewtonCfg = field(
        default_factory=lambda: NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=5,
                nconmax=3,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
        )
    )
    default: NewtonCfg = field(
        default_factory=lambda: NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=5,
                nconmax=3,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
        )
    )
    newton_kamino: NewtonCfg = field(
        default_factory=lambda: NewtonCfg(
            solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
            debug_mode=False,
            use_cuda_graph=True,
        )
    )


##
# Scene definition
##


@dataclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        )
    )

    # cartpole
    robot: ArticulationCfg = field(
        default_factory=lambda: replace_config(CARTPOLE_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    )

    # lights
    # rot quaternion for euler angles (roll, pitch, yaw) = (0, -45, -45) degrees
    distant_light: Any = field(
        default_factory=lambda: AssetBaseCfg(
            prim_path="/World/DistantLight",
            init_state=AssetBaseCfg.InitialStateCfg(
                rot=(-0.14644663035869598, -0.3535534143447876, -0.3535534143447876, 0.8535533547401428)
            ),
            spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0),
        )
    )


##
# MDP settings
##


@dataclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort: Any = field(
        default_factory=lambda: mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=["slider_to_cart"], scale=100.0
        )
    )


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_pos_rel))
        joint_vel_rel: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_vel_rel))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = field(default_factory=PolicyCfg)


@dataclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_cart_position: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
                "position_range": (-1.0, 1.0),
                "velocity_range": (-0.5, 0.5),
            },
        )
    )

    reset_pole_position: Any = field(
        default_factory=lambda: EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
                "position_range": (-0.25 * math.pi, 0.25 * math.pi),
                "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
            },
        )
    )


@dataclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive: Any = field(default_factory=lambda: RewTerm(func=mdp.is_alive, weight=1.0))
    # (2) Failure penalty
    terminating: Any = field(default_factory=lambda: RewTerm(func=mdp.is_terminated, weight=-2.0))
    # (3) Primary task: keep pole upright
    pole_pos: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.joint_pos_target_l2,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
        )
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.joint_vel_l1,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
        )
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel: Any = field(
        default_factory=lambda: RewTerm(
            func=mdp.joint_vel_l1,
            weight=-0.005,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
        )
    )
    # (6) Success rate tracking (zero-weight, metric only)
    success_rate: Any = field(default_factory=lambda: RewTerm(func=mdp.survival_success_rate, weight=0.0))


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out: Any = field(default_factory=lambda: DoneTerm(func=mdp.time_out, time_out=True))
    # (2) Cart out of bounds
    cart_out_of_bounds: Any = field(
        default_factory=lambda: DoneTerm(
            func=mdp.joint_pos_out_of_manual_limit,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
        )
    )


##
# Environment configuration
##


@dataclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: CartpoleSceneCfg = field(
        default_factory=lambda: CartpoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    )
    # Basic settings
    observations: ObservationsCfg = field(default_factory=ObservationsCfg)
    actions: ActionsCfg = field(default_factory=ActionsCfg)
    events: EventCfg = field(default_factory=EventCfg)
    # MDP settings
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # visualizer camera settings
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = CartpolePhysicsCfg()
