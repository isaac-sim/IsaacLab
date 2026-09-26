# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the direct-workflow cartpole environment."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from .cartpole_common import LIGHT_ORIENTATION, CartpolePhysicsCfg


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Cartpole assets constructed and cloned as one scene."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=2000.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=LIGHT_ORIENTATION),
    )


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    """Configuration for the direct-workflow cartpole balancing environment."""

    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=CartpolePhysicsCfg())

    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    scene: CartpoleSceneCfg = CartpoleSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_cart_position_range = (-1.0, 1.0)  # [m]
    initial_cart_velocity_range = (-0.5, 0.5)  # [m/s]
    initial_pole_angle_range = (-0.25 * math.pi, 0.25 * math.pi)  # [rad]
    initial_pole_velocity_range = (-0.25 * math.pi, 0.25 * math.pi)  # [rad/s]
    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))
