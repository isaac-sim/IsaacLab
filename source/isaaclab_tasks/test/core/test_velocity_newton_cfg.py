# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import ast
from importlib import import_module
from inspect import getsource

import gymnasium as gym
from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab.envs import mdp as core_mdp

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
    assert isinstance(env_cfg.scene, rough_env_cfg.G1SceneCfg)
    assert isinstance(env_cfg.rewards, rough_env_cfg.G1Rewards)
    assert isinstance(env_cfg.events, rough_env_cfg.G1EventsCfg)
    assert isinstance(env_cfg.curriculum, rough_env_cfg.G1CurriculumCfg)
    assert isinstance(env_cfg.terminations, rough_env_cfg.G1TerminationsCfg)
    assert env_cfg.sim.use_newton_actuators
    solver_cfg = env_cfg.sim.physics.newton_mjwarp.solver_cfg
    assert isinstance(solver_cfg, MJWarpSolverCfg)
    assert solver_cfg.njmax == 300
    assert runner_cfg.algorithm.symmetry_cfg.use_data_augmentation
    assert "Isaac-Velocity-Rough-G1-Teacher" not in gym.registry
    assert not hasattr(rough_env_cfg, "G1RoughEnvCfg")
    assert not hasattr(rough_env_cfg, "TerrainGeneratorCfg")
    assert not hasattr(rsl_rl_ppo_cfg, "G1RoughPPORunnerCfg")
    assert not hasattr(rsl_rl_ppo_cfg, "G1_SYMMETRY_CFG")
    student_env_cfg = load_cfg_from_registry("Isaac-Velocity-Rough-G1-Depth-Student", "env_cfg_entry_point")
    student_runner_cfg = load_cfg_from_registry("Isaac-Velocity-Rough-G1-Depth-Student", "rsl_rl_cfg_entry_point")
    assert student_runner_cfg.obs_groups["teacher"] == runner_cfg.obs_groups["actor"]
    for cfg, runner in ((env_cfg, runner_cfg), (student_env_cfg, student_runner_cfg)):
        assert not {"policy", "teacher"} & vars(cfg.observations).keys()
        for groups in runner.obs_groups.values():
            assert all(isinstance(getattr(cfg.observations, group), rough_env_cfg.ObsGroup) for group in groups)

    # Generic manager terms belong to core MDP; task modules must not duplicate or alias them.
    for func, module_name in (
        (env_cfg.rewards.joint_deviation_hip.func, "rewards"),
        (env_cfg.rewards.pelvis_height.func, "rewards"),
        (env_cfg.rewards.torso_ang_vel.func, "rewards"),
        (env_cfg.rewards.root_downward_speed.func, "rewards"),
        (env_cfg.terminations.base_height.func, "terminations"),
        (env_cfg.curriculum.base_height.params["modify_fn"], "curriculums"),
    ):
        task_module = import_module(f"isaaclab_tasks.core.velocity.mdp.{module_name}")
        assert func is getattr(core_mdp, func.__name__)
        assert func.__module__ == f"isaaclab.envs.mdp.{module_name}"
        assert func.__name__ not in vars(task_module)

    # Scene and manager configs own static settings; initialization only supplies asset-derived values.
    for node in ast.walk(ast.parse(getsource(rough_env_cfg.G1RoughTeacherEnvCfg))):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assert not ast.unparse(target).startswith(
                    (
                        "self.scene.",
                        "self.rewards.",
                        "self.events.",
                        "self.curriculum.",
                        "self.terminations.",
                        "self.commands.",
                        "self.observations.",
                        "self.sim.",
                    )
                )
    student_class = ast.parse(getsource(rough_env_cfg.G1RoughDepthStudentEnvCfg)).body[0]
    assert not any(isinstance(node, ast.FunctionDef) and node.name == "__post_init__" for node in student_class.body)
    assert not any(
        isinstance(cfg, type)
        and cfg.__module__ == rough_env_cfg.__name__
        and issubclass(cfg, (rough_env_cfg.ObsGroup, rough_env_cfg.CameraCfg))
        for cfg in vars(rough_env_cfg).values()
    )


def test_g1_flat_keeps_its_robot_and_ppo_settings():
    """The flat task keeps its original asset and PPO settings when rough G1 becomes the teacher."""
    env_cfg = load_cfg_from_registry("Isaac-Velocity-Flat-G1", "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry("Isaac-Velocity-Flat-G1", "rsl_rl_cfg_entry_point")

    assert env_cfg.scene.robot.spawn.usd_path == G1_MINIMAL_CFG.spawn.usd_path
    assert env_cfg.scene.terrain.terrain_type == "plane"
    assert env_cfg.scene.height_scanner is None
    assert not hasattr(env_cfg.observations, "policy")
    assert env_cfg.observations.height_scan is None
    assert runner_cfg.obs_groups == {"actor": ["proprio"], "critic": ["proprio"]}
    assert "skrl_cfg_entry_point" not in gym.spec("Isaac-Velocity-Flat-G1").kwargs
    assert runner_cfg.algorithm.symmetry_cfg is None
    assert runner_cfg.actor.hidden_dims == runner_cfg.critic.hidden_dims == [256, 128, 128]
    assert runner_cfg.max_iterations == 1500
    assert runner_cfg.save_interval == 50
