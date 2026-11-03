# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


@pytest.mark.parametrize(
    "task,asset,obs_groups,edge_reward",
    [
        ("Isaac-Velocity-Rough-G1", "g1_minimal.usd", ["policy"], False),
        ("Isaac-Velocity-Rough-G1-29dof", "g1_29dof.usd", ["proprio", "height_scan"], True),
    ],
)
def test_g1_rough_task_configs(task, asset, obs_groups, edge_reward):
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")

    assert env_cfg.scene.robot.spawn.usd_path.endswith(f"/{asset}")
    assert bool(getattr(env_cfg.rewards, "feet_support_edge", None)) == edge_reward
    assert agent_cfg.obs_groups == {"actor": obs_groups, "critic": obs_groups}
    for group in obs_groups:
        assert getattr(env_cfg.observations, group) is not None
    solver_cfg = env_cfg.sim.physics.newton_mjwarp.solver_cfg
    assert isinstance(solver_cfg, MJWarpSolverCfg)
    assert solver_cfg.njmax == 300
