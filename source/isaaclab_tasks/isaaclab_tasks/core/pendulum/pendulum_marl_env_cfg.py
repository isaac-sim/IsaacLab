# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import replace_config

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG


@dataclass
class PendulumPhysicsCfg(PresetCfg):
    """Physics presets for the multi-agent pendulum environment."""

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


@dataclass
class PendulumMARLEnvCfg(DirectMARLEnvCfg):
    """Configuration for the multi-agent cart-double-pendulum balancing environment."""

    # env
    decimation: Any = 2
    episode_length_s: Any = 5.0
    possible_agents: Any = field(default_factory=lambda: ["cart", "pendulum"])
    action_spaces: Any = field(default_factory=lambda: {"cart": 1, "pendulum": 1})
    observation_spaces: Any = field(default_factory=lambda: {"cart": 4, "pendulum": 3})
    state_space: Any = -1

    # simulation
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(dt=1 / 120, render_interval=2, physics=PendulumPhysicsCfg())
    )

    # robot
    robot_cfg: ArticulationCfg = field(
        default_factory=lambda: replace_config(CART_DOUBLE_PENDULUM_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    )
    robot_cfg.actuators["pendulum_actuator"].armature = 0.05
    cart_dof_name: Any = "slider_to_cart"
    pole_dof_name: Any = "cart_to_pole"
    pendulum_dof_name: Any = "pole_to_pendulum"

    # scene
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    )

    # reset
    max_cart_pos: Any = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range: Any = field(
        default_factory=lambda: [-0.25, 0.25]
    )  # the range in which the pole angle is sampled from on reset [rad]
    initial_pendulum_angle_range: Any = field(
        default_factory=lambda: [-0.25, 0.25]
    )  # the range in which the pendulum angle is sampled from on reset [rad]

    # success metric
    success_upright_angle: Any = field(
        default_factory=lambda: math.pi / 12
    )  # both physical links must remain within this angle [rad]
    success_duration_s: Any = 1.0  # required consecutive upright duration [s]

    # action scales
    cart_action_scale: Any = 100.0  # [N]
    pendulum_action_scale: Any = 50.0  # [Nm]

    # reward scales
    rew_scale_alive: Any = 1.0
    rew_scale_terminated: Any = -2.0
    rew_scale_cart_vel: Any = -0.01
    rew_scale_pole_pos: Any = 1.0
    rew_scale_pole_vel: Any = -0.01
    rew_scale_pendulum_pos: Any = 1.0
    rew_scale_pendulum_vel: Any = -0.01
    rew_scale_upright: Any = 1.0
    rew_scale_action: Any = -0.01
