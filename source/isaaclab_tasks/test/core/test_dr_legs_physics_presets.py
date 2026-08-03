# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DR Legs physics presets."""

from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from isaaclab.physics import PhysxAutoCfg

from isaaclab_tasks.contrib.dr_legs.hold_pose_env_cfg import DrLegsHoldPoseEnvCfg, DrLegsPhysicsCfg
from isaaclab_tasks.contrib.dr_legs.walk_env_cfg import DrLegsWalkEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

from isaaclab_assets.robots.dr_legs import DR_LEGS_IMPLICIT_PD_CFG


def test_dr_legs_physx_uses_assembled_joint_reset_and_bounded_joint_velocity() -> None:
    """PhysX must assemble the joints and apply the asset's solver velocity limit."""
    env_cfg = resolve_presets(DrLegsWalkEnvCfg(), selected=("physx",))

    assert isinstance(env_cfg.sim.physics, PhysxAutoCfg)
    assert isinstance(env_cfg.sim.physics.isaacsim_physx, PhysxCfg)
    assert isinstance(env_cfg.scene.contact_forces, PhysXContactSensorCfg)
    assert env_cfg.actions.joint_pos.scale == 0.1
    assert env_cfg.scene.robot == DR_LEGS_IMPLICIT_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    assert env_cfg.scene.robot.actuators["driven_joints"].velocity_limit_sim == 100.0
    assert env_cfg.scene.robot.actuators["passive_joints"].velocity_limit_sim == 100.0
    assert env_cfg.events.reset_robot_joints.params["position_range"] == (0.0, 0.0)
    assert env_cfg.events.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)
    assert env_cfg.events.reset_base.params["pose_range"]["x"] == (-0.05, 0.05)
    assert env_cfg.events.reset_base.params["velocity_range"]["x"] == (-0.2, 0.2)
    assert env_cfg.events.randomize_joint_params is not None


def test_dr_legs_kamino_presets_remain_unchanged() -> None:
    """PhysX support must not change the existing Kamino task distribution."""
    physics = DrLegsPhysicsCfg()
    env_cfg = resolve_presets(DrLegsHoldPoseEnvCfg(), selected=("newton_kamino",))

    assert isinstance(physics.default, NewtonCfg)
    assert isinstance(physics.newton_kamino.solver_cfg, KaminoSolverCfg)
    assert env_cfg.events.reset_base.params["pose_range"]["x"] == (-0.05, 0.05)
    assert env_cfg.events.reset_robot_joints.params["position_range"] == (-0.1, 0.1)
    assert env_cfg.events.reset_robot_joints.params["velocity_range"] == (-0.05, 0.05)
    assert env_cfg.actions.joint_pos.scale == 0.3
