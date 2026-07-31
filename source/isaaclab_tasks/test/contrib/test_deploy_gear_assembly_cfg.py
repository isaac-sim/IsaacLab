# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Deploy GearAssembly environment configuration defaults."""

import pytest
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg

from pxr import Usd

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.deploy.gear_assembly.config.rizon_4s.ik_newton_env_cfg import (
    Rizon4sGearAssemblyIKNewtonEnvCfg,
)
from isaaclab_tasks.contrib.deploy.gear_assembly.config.rizon_4s.joint_pos_env_cfg import (
    Rizon4sGearAssemblyEnvCfg,
)
from isaaclab_tasks.contrib.deploy.gear_assembly.gear_assembly_env_cfg import (
    _NEWTON_GEAR_OFFSETS,
    _PHYSX_GEAR_OFFSETS,
    NEWTON_GEAR_ASSETS_DIR,
)
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.parametrize(
    "task_name",
    [
        "IsaacContrib-Deploy-GearAssembly-UR10e-2F140",
        "IsaacContrib-Deploy-GearAssembly-UR10e-2F85",
    ],
)
def test_ur10e_gear_assembly_default_num_envs(task_name: str):
    """UR10e GearAssembly training configs should fit on 16 GB GPUs by default."""
    env_cfg = parse_env_cfg(task_name)

    assert env_cfg.scene.num_envs == 2048


def test_rizon_gear_newton_collision_presets_use_local_sdf_assets():
    """Newton point and hydro presets should preserve concave package-local gear collision."""
    default_cfg = resolve_presets(Rizon4sGearAssemblyEnvCfg(), {"default"})
    mujoco_cfg = resolve_presets(Rizon4sGearAssemblyEnvCfg(), {"newton_mjwarp"})
    point_cfg = resolve_presets(Rizon4sGearAssemblyEnvCfg(), {"newton_sdf"})
    hydro_cfg = resolve_presets(Rizon4sGearAssemblyEnvCfg(), {"newton_hydroelastic"})

    assert type(default_cfg.sim.physics).__name__ == "PhysxCfg"
    assert default_cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert default_cfg.decimation == 4
    assert default_cfg.scene.replicate_physics is False
    assert default_cfg.scene.robot.spawn.joint_drive_props is None
    assert default_cfg.scene.robot.spawn.rigid_props.disable_gravity is True
    assert default_cfg.scene.factory_gear_base.init_state.pos[2] == 0.071
    assert "/assets/newton/" not in default_cfg.scene.factory_gear_base.spawn.usd_path

    assert mujoco_cfg.sim.physics.solver_cfg.use_mujoco_contacts is True
    assert "/assets/newton/" in mujoco_cfg.scene.factory_gear_base.spawn.usd_path

    for env_cfg in (mujoco_cfg, point_cfg, hydro_cfg):
        assert env_cfg.sim.dt == 0.01
        assert env_cfg.decimation == 3
        assert env_cfg.scene.replicate_physics is True
        assert env_cfg.sim.physics.num_substeps == 20
        assert env_cfg.sim.physics.collision_decimation == 0
        assert env_cfg.sim.physics.solver_cfg.update_data_interval == 10
        assert env_cfg.scene.robot.spawn.joint_drive_props.actuatorgravcomp is True
        assert env_cfg.scene.robot.spawn.rigid_props.disable_gravity is False

    for env_cfg in (point_cfg, hydro_cfg):
        assert env_cfg.sim.physics.solver_cfg.use_mujoco_contacts is False
        assert env_cfg.sim.physics.collision_cfg.max_triangle_pairs == 4_194_304
        assert env_cfg.sim.physics.default_shape_cfg.gap == 0.005
        for asset_name in (
            "factory_gear_base",
            "factory_gear_small",
            "factory_gear_medium",
            "factory_gear_large",
        ):
            usd_path = getattr(env_cfg.scene, asset_name).spawn.usd_path
            assert "/assets/newton/" in usd_path
            assert usd_path.endswith(".usda")

    assert point_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config is None
    assert hydro_cfg.sim.physics.collision_cfg.sdf_hydroelastic_config is not None
    assert point_cfg.scene.factory_gear_small.spawn.usd_path.endswith("factory_gear_small.usda")
    assert hydro_cfg.scene.factory_gear_small.spawn.usd_path.endswith("factory_gear_small_hydroelastic.usda")


def test_rizon_gear_uses_shaft_targets_relative_actions_and_physical_gripper():
    """Newton should use shaft targets and a physical grasp while preserving PhysX defaults."""
    default_cfg = resolve_presets(Rizon4sGearAssemblyEnvCfg(), {"default"})
    hydro_cfg = resolve_presets(Rizon4sGearAssemblyEnvCfg(), {"newton_hydroelastic"})

    for env_cfg in (default_cfg, hydro_cfg):
        assert type(env_cfg.actions.arm_action).__name__ == "RelativeJointPositionActionCfg"
        assert env_cfg.actions.gripper_action is None

    assert default_cfg.gear_offsets == _PHYSX_GEAR_OFFSETS
    assert default_cfg.rewards.end_effector_gear_keypoint_tracking.params["gear_offsets"] is None
    assert default_cfg.events.pin_unselected_gears_to_shafts is None
    assert default_cfg.grasp_center_body_names is None
    assert default_cfg.scene.robot.actuators["gripper_passive"].stiffness == 0.0

    assert hydro_cfg.scene.factory_gear_base.init_state.pos[2] == -0.005
    assert hydro_cfg.gear_offsets == _NEWTON_GEAR_OFFSETS
    assert hydro_cfg.observations.policy.gear_shaft_pos.params["gear_offsets"] == _NEWTON_GEAR_OFFSETS
    assert hydro_cfg.rewards.end_effector_gear_keypoint_tracking.params["gear_offsets"] == _NEWTON_GEAR_OFFSETS
    assert hydro_cfg.rewards.end_effector_gear_keypoint_tracking_exp.params["gear_offsets"] == (_NEWTON_GEAR_OFFSETS)
    assert hydro_cfg.events.pin_unselected_gears_to_shafts is not None
    assert hydro_cfg.grasp_center_body_names == ("left_finger_tip", "right_finger_tip")
    assert all(offset == [0.0, 0.0, -0.026] for offset in hydro_cfg.gear_offsets_grasp.values())

    passive_gripper = hydro_cfg.scene.robot.actuators["gripper_passive"]
    assert passive_gripper.stiffness == 2_000.0
    assert passive_gripper.damping == 10.0
    assert passive_gripper.effort_limit_sim == 20.0
    assert hydro_cfg.scene.robot.actuators["shoulder"].stiffness == 6_000.0


def test_rizon_gear_newton_ik_commands_physical_flange():
    """Newton IK should expose a relative flange-pose action with normal safety limits."""
    env_cfg = resolve_presets(Rizon4sGearAssemblyIKNewtonEnvCfg(), {"newton_hydroelastic"})
    pose_objective = env_cfg.actions.arm_action.objectives[0]

    assert isinstance(env_cfg.actions.arm_action, NewtonInverseKinematicsActionCfg)
    assert pose_objective.body_name == "flange"
    assert pose_objective.use_relative_mode is True
    assert pose_objective.scale == 0.025
    assert env_cfg.terminations.gear_orientation_exceeded.params["roll_threshold_deg"] == 15.0
    assert env_cfg.terminations.gear_orientation_exceeded.params["pitch_threshold_deg"] == 15.0


def test_rizon_gear_assets_author_newton_sdf_per_collider():
    """Every package-local gear mesh should preserve concavity with authored Newton SDF metadata."""
    for asset_name in (
        "factory_gear_base",
        "factory_gear_small",
        "factory_gear_medium",
        "factory_gear_large",
    ):
        for suffix, hydroelastic_enabled in (("", False), ("_hydroelastic", True)):
            usd_path = f"{NEWTON_GEAR_ASSETS_DIR}/{asset_name}/{asset_name}{suffix}.usda"
            stage = Usd.Stage.Open(usd_path)
            sdf_prims = [prim for prim in stage.Traverse() if prim.HasAttribute("newton:sdfMaxResolution")]

            assert len(sdf_prims) == 1
            assert sdf_prims[0].GetAttribute("newton:sdfMaxResolution").Get() == 128
            assert sdf_prims[0].GetAttribute("newton:hydroelasticEnabled").Get() is hydroelastic_enabled
            assert sdf_prims[0].GetAttribute("physics:approximation").Get() == "sdf"
