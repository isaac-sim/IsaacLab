# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import replace_config
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


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
    newton_kamino: NewtonCfg = field(
        default_factory=lambda: NewtonCfg(
            solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
            debug_mode=False,
            use_cuda_graph=True,
        )
    )
    default: Any = field(
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


@dataclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # env
    decimation: Any = 2
    episode_length_s: Any = 5.0
    action_scale: Any = 100.0  # [N]
    action_space: Any = 1
    observation_space: Any = 4
    state_space: Any = 0

    # simulation
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(dt=1 / 120, render_interval=2, physics=CartpolePhysicsCfg())
    )

    # robot
    robot_cfg: ArticulationCfg = field(
        default_factory=lambda: replace_config(CARTPOLE_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    )
    cart_dof_name: Any = "slider_to_cart"
    pole_dof_name: Any = "cart_to_pole"

    # scene
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(
            num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
        )
    )

    # reset
    max_cart_pos: Any = 3.0  # the cart is reset if it exceeds that position [m]
    initial_cart_position_range: Any = (-1.0, 1.0)  # [m]
    initial_cart_velocity_range: Any = (-0.5, 0.5)  # [m/s]
    initial_pole_angle_range: Any = (-0.25 * math.pi, 0.25 * math.pi)  # [rad]
    initial_pole_velocity_range: Any = (-0.25 * math.pi, 0.25 * math.pi)  # [rad/s]
    # reward scales
    rew_scale_alive: Any = 1.0
    rew_scale_terminated: Any = -2.0
    rew_scale_pole_pos: Any = -1.0
    rew_scale_cart_vel: Any = -0.01
    rew_scale_pole_vel: Any = -0.005

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))
