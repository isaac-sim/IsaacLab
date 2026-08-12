# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based configuration for the multi-agent cart-double-pendulum task."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedMARLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.pendulum.mdp as mdp
from isaaclab_tasks.core.pendulum.pendulum_marl_env_cfg import PendulumPhysicsCfg

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG  # isort: skip


@configclass
class PendulumMARLSceneCfg(InteractiveSceneCfg):
    """Scene configuration with the final direct-task pendulum asset settings."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["pendulum_actuator"].armature = 0.05
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


_CART_CFG = SceneEntityCfg("robot", joint_names=["slider_to_cart"])
_POLE_CFG = SceneEntityCfg("robot", joint_names=["cart_to_pole"])
_PENDULUM_CFG = SceneEntityCfg("robot", joint_names=["pole_to_pendulum"])
_JOINT_DATA_PARAMS = {"cart_cfg": _CART_CFG, "pole_cfg": _POLE_CFG, "pendulum_cfg": _PENDULUM_CFG}


@configclass
class CartActionsCfg:
    """Cart agent action configuration."""

    effort = base_mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class PendulumActionsCfg:
    """Lower-pendulum agent action configuration."""

    effort = base_mdp.JointEffortActionCfg(asset_name="robot", joint_names=["pole_to_pendulum"], scale=50.0)


@configclass
class CartObservationsCfg:
    """Cart agent observation configuration."""

    @configclass
    class PolicyCfg(ObsGroup):
        """The single cart policy observation group."""

        observations = ObsTerm(func=mdp.cart_observation, params=_JOINT_DATA_PARAMS)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PendulumObservationsCfg:
    """Lower-pendulum agent observation configuration."""

    @configclass
    class PolicyCfg(ObsGroup):
        """The single lower-pendulum policy observation group."""

        observations = ObsTerm(func=mdp.pendulum_observation, params=_JOINT_DATA_PARAMS)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class StateObservationsCfg:
    """Centralized state configuration."""

    @configclass
    class StateCfg(ObsGroup):
        """The single centralized-state group."""

        observations = ObsTerm(func=mdp.state, params=_JOINT_DATA_PARAMS)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    state: StateCfg = StateCfg()


@configclass
class ResetEventCfg:
    """Reset events matching the final direct Pendulum state writes."""

    reset_scene = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_pole_position = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": _POLE_CFG,
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_pendulum_position = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": _PENDULUM_CFG,
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TeamRewardsCfg:
    """Per-agent reward terms producing the shared final direct-task reward."""

    # RewardManager multiplies every weight by step_dt. The direct reward is
    # also step-dt scaled, so these are the raw direct coefficients, not dt-scaled.
    alive = RewTerm(func=mdp.alive, weight=1.0)
    terminating = RewTerm(func=mdp.terminated, weight=-2.0)
    cart_vel = RewTerm(func=mdp.cart_velocity_l1, weight=-0.01, params=_JOINT_DATA_PARAMS)
    pole_pos = RewTerm(func=mdp.pole_position, weight=1.0, params=_JOINT_DATA_PARAMS)
    pole_vel = RewTerm(func=mdp.pole_velocity_l1, weight=-0.01, params=_JOINT_DATA_PARAMS)
    pendulum_pos = RewTerm(func=mdp.lower_link_position, weight=1.0, params=_JOINT_DATA_PARAMS)
    pendulum_vel = RewTerm(func=mdp.lower_link_velocity_l1, weight=-0.01, params=_JOINT_DATA_PARAMS)
    upright = RewTerm(
        func=mdp.upright,
        weight=1.0,
        params={**_JOINT_DATA_PARAMS, "success_upright_angle": math.pi / 12},
    )
    action = RewTerm(func=mdp.cart_action_l2, weight=-0.01)


@configclass
class SharedTerminationsCfg:
    """The shared direct-task termination conditions for both agents."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(
        func=mdp.out_of_bounds,
        params={"cart_cfg": _CART_CFG, "pole_cfg": _POLE_CFG, "max_cart_pos": 3.0},
    )


@configclass
class CartTerminationsCfg(SharedTerminationsCfg):
    """Cart termination configuration, including the shared success tracker."""

    success_tracker = DoneTerm(
        func=mdp.ConsecutiveUprightSuccess,
        params={
            "pole_cfg": _POLE_CFG,
            "pendulum_cfg": _PENDULUM_CFG,
            "success_upright_angle": math.pi / 12,
            "success_duration_s": 1.0,
        },
    )


@configclass
class PendulumTerminationsCfg(SharedTerminationsCfg):
    """Lower-pendulum termination configuration without duplicate success updates."""


@configclass
class PendulumMARLManagerEnvCfg(ManagerBasedMARLEnvCfg):
    """Manager-based multi-agent cart-double-pendulum balancing environment."""

    scene: PendulumMARLSceneCfg = PendulumMARLSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2, physics=PendulumPhysicsCfg())
    events: ResetEventCfg = ResetEventCfg()

    decimation = 2
    episode_length_s = 5.0
    agents = {
        "cart": ManagerBasedMARLEnvCfg.AgentCfg(
            actions=CartActionsCfg(),
            observations=CartObservationsCfg(),
            rewards=TeamRewardsCfg(),
            terminations=CartTerminationsCfg(),
        ),
        "pendulum": ManagerBasedMARLEnvCfg.AgentCfg(
            actions=PendulumActionsCfg(),
            observations=PendulumObservationsCfg(),
            rewards=TeamRewardsCfg(),
            terminations=PendulumTerminationsCfg(),
        ),
    }
    state: StateObservationsCfg = StateObservationsCfg()
