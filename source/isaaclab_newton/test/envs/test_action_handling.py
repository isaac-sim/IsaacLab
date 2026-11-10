# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton running a manager-based environment's actions inside its step."""

from pathlib import Path

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.mdp import JointEffortActionCfg, joint_pos
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    effort = JointEffortActionCfg(asset_name="robot", joint_names=[".*"])


@pytest.mark.parametrize("opted_out", [False, True], ids=["graphable", "opted-out"])
def test_actions_wrapped_after_construction_stay_in_the_environment_loop(opted_out: bool):
    """Physics takes the actions on the first step, unless the action manager was marked not graphable before it."""
    scene_cfg = InteractiveSceneCfg(num_envs=2, env_spacing=5.0)
    scene_cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parents[1] / "assets/data/articulation_ordering_branching.usda")
        ),
        actuators={},
    )
    observations = ObservationGroupCfg()
    observations.joints = ObservationTermCfg(func=joint_pos)
    cfg = ManagerBasedEnvCfg(
        sim=SimulationCfg(dt=0.01, render_interval=2, device="cuda:0", physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
        scene=scene_cfg,
        decimation=2,
        observations={"policy": observations},
        actions=ActionsCfg(),
        seed=7,
    )
    env = ManagerBasedEnv(cfg)
    try:
        env.reset()
        applied = []
        if opted_out:
            # a Python wrapper installed after construction, as export does
            apply_action = env.action_manager.apply_action
            env.action_manager.apply_action = lambda: (applied.append(None), apply_action())
            env.action_manager.is_graphable = False
        for _ in range(3):
            env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
        assert env.sim.backend_manager.backend.handles_actions is not opted_out
        assert len(applied) == (3 * cfg.decimation if opted_out else 0)
    finally:
        env.close()
