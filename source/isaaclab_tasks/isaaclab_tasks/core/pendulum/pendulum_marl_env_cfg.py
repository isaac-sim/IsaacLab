# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import config_field, replace_config

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG


@dataclass
class PendulumPhysicsCfg(PresetCfg):
    """Physics presets for the multi-agent pendulum environment."""

    isaacsim_physx: PhysxCfg = config_field(PhysxCfg())
    ovphysx: OvPhysxCfg = config_field(OvPhysxCfg())
    physx: PhysxAutoCfg = config_field(PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx))
    newton_mjwarp: NewtonCfg = config_field(
        NewtonCfg(
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
    default: NewtonCfg = config_field(newton_mjwarp)
    newton_kamino: NewtonCfg = config_field(
        NewtonCfg(
            solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
            debug_mode=False,
            use_cuda_graph=True,
        )
    )


@dataclass
class PendulumMARLEnvCfg(DirectMARLEnvCfg):
    """Configuration for the multi-agent cart-double-pendulum balancing environment."""

    # env
    decimation: Any = config_field(2)
    episode_length_s: Any = config_field(5.0)
    possible_agents: Any = config_field(["cart", "pendulum"])
    action_spaces: Any = config_field({"cart": 1, "pendulum": 1})
    observation_spaces: Any = config_field({"cart": 4, "pendulum": 3})
    state_space: Any = config_field(-1)

    # simulation
    sim: SimulationCfg = config_field(
        SimulationCfg(dt=1 / 120, render_interval=decimation, physics=PendulumPhysicsCfg())
    )

    # robot
    robot_cfg: ArticulationCfg = config_field(
        replace_config(CART_DOUBLE_PENDULUM_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    )
    robot_cfg.actuators["pendulum_actuator"].armature = 0.05
    cart_dof_name: Any = config_field("slider_to_cart")
    pole_dof_name: Any = config_field("cart_to_pole")
    pendulum_dof_name: Any = config_field("pole_to_pendulum")

    # scene
    scene: InteractiveSceneCfg = config_field(
        InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    )

    # reset
    max_cart_pos: Any = config_field(3.0)  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range: Any = config_field(
        [-0.25, 0.25]
    )  # the range in which the pole angle is sampled from on reset [rad]
    initial_pendulum_angle_range: Any = config_field(
        [-0.25, 0.25]
    )  # the range in which the pendulum angle is sampled from on reset [rad]

    # success metric
    success_upright_angle: Any = config_field(math.pi / 12)  # both physical links must remain within this angle [rad]
    success_duration_s: Any = config_field(1.0)  # required consecutive upright duration [s]

    # action scales
    cart_action_scale: Any = config_field(100.0)  # [N]
    pendulum_action_scale: Any = config_field(50.0)  # [Nm]

    # reward scales
    rew_scale_alive: Any = config_field(1.0)
    rew_scale_terminated: Any = config_field(-2.0)
    rew_scale_cart_vel: Any = config_field(-0.01)
    rew_scale_pole_pos: Any = config_field(1.0)
    rew_scale_pole_vel: Any = config_field(-0.01)
    rew_scale_pendulum_pos: Any = config_field(1.0)
    rew_scale_pendulum_vel: Any = config_field(-0.01)
    rew_scale_upright: Any = config_field(1.0)
    rew_scale_action: Any = config_field(-0.01)
