# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole teacher configurations for learned-dynamics data collection and deployment."""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.utils import configclass

from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg

CARTPOLE_REMOVE_ROOT_TRANSLATION = False
"""Root translation removed from Cartpole model inputs. The fixed-base cart has no free root."""


@configclass
class CartpoleTeacherEnvCfg(CartpoleEnvCfg):
    """Direct Cartpole stepped once per environment step by MuJoCo Warp at 120 Hz.

    One environment step equals one physics frame, as required for recording and for deploying a
    dynamics artifact with :func:`~isaaclab_physicsnemo.apply_learned_physics`.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.decimation = 1
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 1
        self.sim.create_stage_in_memory = True
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=5, nconmax=3, cone="pyramidal", impratio=1.0, integrator="implicitfast"),
            num_substeps=1,
            use_cuda_graph=False,
            deterministic_mode="not_guaranteed",
        )


@configclass
class CartpoleExplorationEnvCfg(CartpoleTeacherEnvCfg):
    """Cartpole teacher that starts episodes from task-independent states for dynamics data collection.

    Following the task-independent data of NeRD and FlashNeRD, episodes start anywhere on the cart track,
    at every pole angle, and with large velocities, and they last long enough for 200-frame rollout
    evaluation. The Newton model equals that of :class:`CartpoleTeacherEnvCfg`, so artifacts trained on
    this data deploy to every Cartpole configuration built on it.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 2.0
        self.initial_cart_position_range = (-2.0, 2.0)
        self.initial_cart_velocity_range = (-2.0, 2.0)
        self.initial_pole_angle_range = (-math.pi, math.pi)
        self.initial_pole_velocity_range = (-2.0 * math.pi, 2.0 * math.pi)
