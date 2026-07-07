# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration checks for the Franka rigid cube-lift physics presets."""

from __future__ import annotations

import sys

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.lift.lift_env_cfg import LiftPhysicsCfg
from isaaclab_tasks.core.lift.mdp import CurriculumDifferentialInverseKinematicsActionCfg
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-Lift-Cube-Franka"
_CURRICULUM_TASK = "Isaac-Lift-Cube-Franka-Newton-Curriculum"
_AGENT = "rsl_rl_cfg_entry_point"


def test_lift_cube_preserves_physx_default() -> None:
    """The existing task behavior should remain on its tuned PhysX configuration."""
    cfg = load_cfg_from_registry(_TASK, "env_cfg_entry_point")

    assert isinstance(cfg.sim.physics, LiftPhysicsCfg)
    assert isinstance(cfg.sim.physics.default, PhysxCfg)
    assert cfg.sim.physics.default.bounce_threshold_velocity == 0.01
    assert cfg.sim.physics.default.friction_correlation_distance == 0.00625
    agent_cfg = load_cfg_from_registry(_TASK, _AGENT)
    assert agent_cfg.clip_actions.default is None
    assert cfg.rewards.object_dropping.weight.default == 0.0


def test_lift_cube_resolves_newton_mjwarp_physics_selector() -> None:
    """The typed selector should replace the physics preset with Newton MJWarp."""
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], "physics=newton_mjwarp"]
        cfg, agent_cfg = resolve_task_config(_TASK, _AGENT)
    finally:
        sys.argv = original_argv

    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
    assert cfg.sim.physics.solver_cfg.integrator == "implicitfast"
    assert cfg.sim.physics.solver_cfg.cone == "elliptic"
    assert cfg.sim.physics.solver_cfg.impratio == 10
    assert cfg.sim.physics.num_substeps == 2
    assert cfg.scene.robot.actuators["panda_shoulder"].armature == 0.1
    assert cfg.scene.robot.actuators["panda_forearm"].armature == 0.1
    assert cfg.scene.robot.actuators["panda_shoulder"].damping == 10.0
    assert cfg.scene.robot.actuators["panda_forearm"].damping == 10.0
    assert cfg.actions.arm_action.scale == 0.5
    assert agent_cfg.actor.distribution_cfg.init_std == 1.0
    assert agent_cfg.actor.distribution_cfg.std_range == (1.0, 1.0)
    assert agent_cfg.algorithm.entropy_coef == 0.001
    assert agent_cfg.algorithm.schedule == "adaptive"
    assert agent_cfg.clip_actions == 1.0
    assert cfg.rewards.object_dropping.weight == -200.0
    assert cfg.rewards.object_dropping.params["term_keys"] == "object_dropping"
    assert cfg.curriculum.action_rate.params["num_steps"] == 10000
    assert cfg.curriculum.joint_vel.params["num_steps"] == 10000


def test_newton_curriculum_task_is_newton_only_and_uses_final_task_curriculum() -> None:
    """The curriculum task should pin Newton and remove the abrupt penalty schedule."""
    cfg = load_cfg_from_registry(_CURRICULUM_TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(_CURRICULUM_TASK, _AGENT)

    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.actions.arm_action, CurriculumDifferentialInverseKinematicsActionCfg)
    assert cfg.actions.arm_action.scale == (0.1, 0.1, 0.1, 0.25, 0.25, 0.25)
    assert cfg.actions.arm_action.controller.use_relative_mode
    assert cfg.actions.arm_action.full_control_difficulty == 0.30
    assert cfg.actions.gripper_action.close_command_expr == {"panda_finger_.*": 0.016}
    assert cfg.actions.gripper_action.force_close_below_difficulty == 0.45
    assert cfg.scene.robot.spawn.rigid_props.disable_gravity
    assert cfg.scene.robot.actuators["panda_shoulder"].stiffness == 20000.0
    assert cfg.scene.robot.actuators["panda_forearm"].damping == 2000.0
    assert cfg.scene.robot.actuators["panda_forearm"].effort_limit_sim == 100.0
    assert cfg.events.reset_curriculum.func is not None
    assert cfg.events.reset_curriculum.params["closed_finger_position"] == 0.016
    assert cfg.curriculum.lift_difficulty.params["max_difficulty"] == 40
    assert cfg.curriculum.lift_difficulty.params["initial_difficulty"] == 0
    assert cfg.curriculum.lift_difficulty.params["successes_to_promote"] == 1
    assert cfg.curriculum.lift_difficulty.params["success_termination_name"] == "success"
    assert cfg.rewards.action_rate.weight == -1e-3
    assert cfg.rewards.joint_vel.weight == -1e-4
    assert cfg.rewards.object_goal_tracking.params["std"] == 0.12
    assert cfg.rewards.object_goal_orientation.params["std"] == 1.0
    assert cfg.rewards.object_goal_orientation.weight == 4.0
    assert cfg.rewards.object_goal_fine_tracking.params["std"] == 0.025
    assert cfg.rewards.object_goal_fine_orientation.params["std"] == 0.10
    assert cfg.rewards.object_goal_pose_accuracy.weight == 10.0
    assert cfg.rewards.action_magnitude.weight == -0.05
    assert cfg.rewards.object_angular_velocity.weight == -0.02
    assert cfg.rewards.object_dropping.weight == -50.0
    assert cfg.rewards.success_bonus.weight == 5000.0
    assert cfg.terminations.success.params["position_threshold"] == 0.02
    assert cfg.terminations.success.params["orientation_threshold"] == 0.15
    assert cfg.terminations.success.params["hold_time"] == 1.0
    assert cfg.terminations.object_dropping.func is not None
    assert cfg.terminations.object_dropping.params["height_margin"] == 0.10
    assert cfg.observations.policy.ee_to_object.func is not None
    assert cfg.observations.policy.object_to_goal.func is not None
    assert cfg.observations.policy.object_orientation_to_goal.func is not None
    assert agent_cfg.clip_actions == 1.0
    assert agent_cfg.actor.distribution_cfg.init_std == 0.10
    assert agent_cfg.actor.distribution_cfg.std_range == (0.01, 1.0)
