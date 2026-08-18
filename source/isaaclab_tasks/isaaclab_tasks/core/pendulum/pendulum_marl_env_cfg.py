# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG


@configclass
class PendulumPhysicsCfg(PresetCfg):
    """Physics presets for the multi-agent pendulum environment."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
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
    default: NewtonCfg = newton_mjwarp
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )


@configclass
class PendulumMARLEnvCfg(DirectMARLEnvCfg):
    """Configuration for the multi-agent cart-double-pendulum balancing environment."""

    # env
    decimation = 2
    episode_length_s = 5.0
    possible_agents = ["cart", "pendulum"]
    action_spaces = {"cart": 1, "pendulum": 1}
    observation_spaces = {"cart": 4, "pendulum": 3}
    state_space = -1

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=PendulumPhysicsCfg())

    # robot
    robot_cfg: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.actuators["pendulum_actuator"].armature = 0.05
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    pendulum_dof_name = "pole_to_pendulum"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]
    initial_pendulum_angle_range = [-0.25, 0.25]  # the range in which the pendulum angle is sampled from on reset [rad]

    # success metric
    success_upright_angle = math.pi / 12  # both physical links must remain within this angle [rad]
    success_duration_s = 1.0  # required consecutive upright duration [s]

    # action scales
    cart_action_scale = 100.0  # [N]
    pendulum_action_scale = 50.0  # [Nm]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_pos = 1.0
    rew_scale_pole_vel = -0.01
    rew_scale_pendulum_pos = 1.0
    rew_scale_pendulum_vel = -0.01
    rew_scale_upright = 1.0
    rew_scale_action = -0.01
