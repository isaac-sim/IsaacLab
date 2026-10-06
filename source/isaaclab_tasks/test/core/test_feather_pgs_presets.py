# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded behavior checks of the FeatherPGS task presets."""

from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=FeatherPGSSolverCfg())))

import gymnasium as gym
import pytest
import torch
from isaaclab_newton.physics import NewtonManager
from newton.solvers import SolverFeatherPGS

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

_TASKS = ["Isaac-Cartpole", "Isaac-Cartpole-Direct", "Isaac-Ant", "Isaac-Velocity-Flat-UnitreeGo2"]


@pytest.mark.parametrize("task_name", _TASKS)
def test_feather_pgs_preset_resolves(task_name):
    """The ``feather_pgs`` physics preset selects the FeatherPGS solver."""
    env_cfg = parse_env_cfg(task_name, overrides=["physics=feather_pgs"])

    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert isinstance(env_cfg.sim.physics.solver_cfg, FeatherPGSSolverCfg)


@pytest.mark.parametrize("task_name", _TASKS)
def test_feather_pgs_preset_steps_and_resets_finitely(task_name):
    """Random actions with a partial reset keep the state finite, raising if the step drops constraint rows."""
    if not torch.cuda.is_available():
        pytest.skip("The FeatherPGS presets use the CUDA-only matrix-free solve.")
    sim_utils.create_new_stage()
    env_cfg = parse_env_cfg(task_name, device="cuda:0", num_envs=8, overrides=["physics=feather_pgs"])
    env_cfg.sim.physics.solver_cfg.raise_on_constraint_overflow = True
    env_cfg.seed = 7
    env = gym.make(task_name, cfg=env_cfg)
    try:
        env.reset()
        assert isinstance(NewtonManager._solver, SolverFeatherPGS)
        with torch.inference_mode():
            for step in range(120):
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                obs, _, terminated, truncated, _ = env.step(actions)
                if step == 60:
                    env.unwrapped._reset_idx(torch.arange(0, 8, 2, device=env.unwrapped.device))
                for value in obs.values():
                    assert torch.isfinite(value).all()
        for articulation in env.unwrapped.scene.articulations.values():
            assert torch.isfinite(articulation.data.joint_pos.torch).all()
    finally:
        env.close()
