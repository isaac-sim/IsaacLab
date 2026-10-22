# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab_tasks.core.velocity.config.g1 import rough_env_cfg
from isaaclab_tasks.core.velocity.config.g1.agents import rsl_rl_ppo_cfg
from isaaclab_tasks.utils import load_cfg_from_registry

from isaaclab_assets import G1_MINIMAL_CFG


def test_g1_rough_teacher_registration_and_newton_capacity(monkeypatch):
    """Rough G1 resolves to one teacher task and retains sufficient constraint capacity."""
    monkeypatch.setattr(
        rough_env_cfg, "get_action_parameters", lambda cfg: (1.0, (0.2, 0.06, 0.02), (0.03, 0.0, -0.02))
    )
    env_cfg = load_cfg_from_registry("Isaac-Velocity-Rough-G1", "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry("Isaac-Velocity-Rough-G1", "rsl_rl_cfg_entry_point")

    assert isinstance(env_cfg, rough_env_cfg.G1RoughTeacherEnvCfg)
    assert isinstance(runner_cfg, rsl_rl_ppo_cfg.G1RoughTeacherPPORunnerCfg)
    solver_cfg = env_cfg.sim.physics.newton_mjwarp.solver_cfg
    assert isinstance(solver_cfg, MJWarpSolverCfg)
    assert solver_cfg.njmax == 300
    assert runner_cfg.algorithm.symmetry_cfg.use_data_augmentation
    assert "Isaac-Velocity-Rough-G1-Teacher" not in gym.registry
    assert not hasattr(rough_env_cfg, "G1RoughEnvCfg")
    assert not hasattr(rsl_rl_ppo_cfg, "G1RoughPPORunnerCfg")


def test_g1_flat_keeps_its_robot_and_ppo_settings():
    """The flat task keeps its original asset and PPO settings when rough G1 becomes the teacher."""
    env_cfg = load_cfg_from_registry("Isaac-Velocity-Flat-G1", "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point")

    assert env_cfg.scene.robot.spawn.usd_path == G1_MINIMAL_CFG.spawn.usd_path
    assert env_cfg.scene.terrain.terrain_type == "plane"
    assert env_cfg.scene.height_scanner is None
    assert runner_cfg.algorithm.symmetry_cfg is None
    assert runner_cfg.actor.hidden_dims == runner_cfg.critic.hidden_dims == [256, 128, 128]
    assert runner_cfg.max_iterations == 1500
    assert runner_cfg.save_interval == 50
