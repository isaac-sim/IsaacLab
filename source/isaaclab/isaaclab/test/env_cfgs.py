# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared core-only environment configurations for Isaac Lab tests.

This module imports simulator-dependent configuration types and must therefore be imported
only after :class:`isaaclab.app.AppLauncher` starts the simulator.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

_DEFAULT_DECIMATION = 4
_DEFAULT_EPISODE_LENGTH_S = 5.0
_DEFAULT_SIM_DT = 0.005

_CARTPOLE_TEST_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


def make_empty_manager_based_env_cfg(
    device: str = "cuda:0",
    num_envs: int = 1,
    env_spacing: float = 1.0,
) -> ManagerBasedEnvCfg:
    """Create an empty manager-based environment configuration.

    Args:
        device: Device on which to run the simulation.
        num_envs: Number of environment instances.
        env_spacing: Distance between environment origins [m].

    Returns:
        A configuration with empty action and observation managers.
    """
    return ManagerBasedEnvCfg(
        decimation=_DEFAULT_DECIMATION,
        sim=sim_utils.SimulationCfg(
            device=device,
            dt=_DEFAULT_SIM_DT,
            render_interval=_DEFAULT_DECIMATION,
        ),
        scene=EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing),
        actions=EmptyManagerCfg(),
        observations=EmptyManagerCfg(),
    )


def make_empty_manager_based_rl_env_cfg(
    device: str = "cuda:0",
    num_envs: int = 1,
    env_spacing: float = 1.0,
) -> ManagerBasedRLEnvCfg:
    """Create an empty manager-based reinforcement-learning environment configuration.

    Args:
        device: Device on which to run the simulation.
        num_envs: Number of environment instances.
        env_spacing: Distance between environment origins [m].

    Returns:
        A configuration with empty action, observation, reward, and termination managers.
    """
    return ManagerBasedRLEnvCfg(
        decimation=_DEFAULT_DECIMATION,
        episode_length_s=_DEFAULT_EPISODE_LENGTH_S,
        sim=sim_utils.SimulationCfg(
            device=device,
            dt=_DEFAULT_SIM_DT,
            render_interval=_DEFAULT_DECIMATION,
        ),
        scene=EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing),
        actions=EmptyManagerCfg(),
        observations=EmptyManagerCfg(),
        rewards=EmptyManagerCfg(),
        terminations=EmptyManagerCfg(),
    )


def make_empty_direct_marl_env_cfg(
    device: str = "cuda:0",
    num_envs: int = 1,
    env_spacing: float = 1.0,
) -> DirectMARLEnvCfg:
    """Create an empty direct multi-agent reinforcement-learning environment configuration.

    The dummy agents and spaces satisfy the structural requirements of
    :class:`isaaclab.envs.DirectMARLEnvCfg`; tests may replace them with their own definitions.

    Args:
        device: Device on which to run the simulation.
        num_envs: Number of environment instances.
        env_spacing: Distance between environment origins [m].

    Returns:
        A configuration with an empty scene and two dummy agents.
    """
    return DirectMARLEnvCfg(
        decimation=1,
        episode_length_s=100.0,
        sim=sim_utils.SimulationCfg(device=device, dt=_DEFAULT_SIM_DT, render_interval=1),
        scene=EmptySceneCfg(num_envs=num_envs, env_spacing=env_spacing),
        possible_agents=["agent_0", "agent_1"],
        action_spaces={"agent_0": 1, "agent_1": 2},
        observation_spaces={"agent_0": 3, "agent_1": 4},
        state_space=-1,
    )


@configclass
class EmptyManagerCfg:
    """Empty manager term configuration."""

    pass


@configclass
class EmptySceneCfg(InteractiveSceneCfg):
    """Configuration for a scene without entities."""

    pass


@configclass
class CartpoleTestSceneCfg(InteractiveSceneCfg):
    """Configuration for a minimal cart-pole articulation scene."""

    robot: ArticulationCfg = _CARTPOLE_TEST_CFG.copy()


@configclass
class ArticulationRigidObjectSceneCfg(CartpoleTestSceneCfg):
    """Configuration for a minimal scene with articulation and rigid-object state."""

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )
