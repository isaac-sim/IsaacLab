# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Cartpole tasks for offline dynamics collection and learned-physics policy training."""

import math

from isaaclab_newton.envs.mdp.recorders import NeRDRecorderCfg, NeRDRecorderManagerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.nerd_manager_cfg import NeRDSolverCfg

from isaaclab.utils import configclass

from .cartpole_manager_env_cfg import CartpoleEnvCfg


@configclass
class CartpoleNeRDEnvCfg(CartpoleEnvCfg):
    """MuJoCo Warp teacher at 120 Hz, with optional native transition capture."""

    recorders: NeRDRecorderManagerCfg = NeRDRecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = self.sim.render_interval = 1
        self.sim.dt = 1.0 / 120.0
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=5, nconmax=3, cone="pyramidal", impratio=1.0, integrator="implicitfast"),
            num_substeps=1,
            use_cuda_graph=True,
        )


@configclass
class CartpoleNeRDExplorationEnvCfg(CartpoleNeRDEnvCfg):
    """Task-independent initial states for a dynamics dataset, including full pole rotations."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 2.0
        self.events.reset_cart_position.params.update(position_range=(-2.0, 2.0), velocity_range=(-2.0, 2.0))
        self.events.reset_pole_position.params.update(
            position_range=(-math.pi, math.pi), velocity_range=(-2 * math.pi, 2 * math.pi)
        )
        self.recorders.transition = NeRDRecorderCfg()


@configclass
class CartpoleNeRDLearnedEnvCfg(CartpoleNeRDEnvCfg):
    """The same control task advanced by a NeRD checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = NewtonCfg(solver_cfg=NeRDSolverCfg(), num_substeps=1, use_cuda_graph=True)
