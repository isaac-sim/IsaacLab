# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg


def test_g1_rough_newton_has_sufficient_constraint_capacity():
    env_cfg = G1RoughEnvCfg()

    solver_cfg = env_cfg.sim.physics.newton_mjwarp.solver_cfg

    assert isinstance(solver_cfg, MJWarpSolverCfg)
    assert solver_cfg.njmax == 300
