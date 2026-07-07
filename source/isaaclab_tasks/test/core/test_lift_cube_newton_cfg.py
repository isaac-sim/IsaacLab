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
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-Lift-Cube-Franka"
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
