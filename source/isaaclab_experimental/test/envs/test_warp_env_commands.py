# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Commands of Warp environments built one after another in one process."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

import isaaclab_experimental.envs.mdp as warp_mdp
import torch
import warp as wp
from isaaclab_experimental.envs.frontend import WarpFrontend

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_TASK = "Isaac-Velocity-Flat-UnitreeGo2"
_NUM_ENVS = 8


def test_each_velocity_env_reads_its_own_commands():
    """A second Warp velocity environment builds, and its command-reading terms read its own commands.

    The terms read the command through the environment's command manager on every call, so nothing on the term
    functions refers to an earlier environment.
    """
    for seed in (7, 8):
        env_cfg, _ = resolve_task_config(_TASK, "", overrides=("physics=newton_mjwarp",))
        env_cfg.seed = seed
        env_cfg.scene.num_envs = _NUM_ENVS
        sim_utils.create_new_stage()
        env = WarpFrontend.build_env(env_cfg, _TASK).unwrapped
        try:
            env.reset()
            env.step(torch.zeros((_NUM_ENVS, env.action_space.shape[-1]), device=env.device))
            out = wp.zeros((_NUM_ENVS, 3), dtype=wp.float32, device=env.device)

            warp_mdp.generated_commands(env, out, command_name="base_velocity")

            assert torch.equal(wp.to_torch(out), env.command_manager.get_command("base_velocity"))
        finally:
            env.close()
