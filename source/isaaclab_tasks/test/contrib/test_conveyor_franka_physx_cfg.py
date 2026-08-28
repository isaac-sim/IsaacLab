# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-light checks for the checkpoint-compatible PhysX conveyor configuration."""

import gymnasium as gym
import pytest
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import ConveyorFrankaEnvCfg
from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_physx_env_cfg import (
    ConveyorFrankaPhysxEnvCfg,
    physx_belt_section_specs,
)
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import BELT_TURN_RADIUS, MeshSpec
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def test_physx_task_is_registered_with_a_dedicated_config() -> None:
    """The native backend is opt-in and cannot alter the Newton task registration."""
    physx_spec = gym.spec("IsaacContrib-Conveyor-Franka-PhysX-CPU-v0")
    newton_spec = gym.spec("IsaacContrib-Conveyor-Franka-Newton-v0")

    assert physx_spec.entry_point == newton_spec.entry_point
    assert physx_spec.kwargs["env_cfg_entry_point"].endswith(":ConveyorFrankaPhysxEnvCfg")
    assert newton_spec.kwargs["env_cfg_entry_point"].endswith(":ConveyorFrankaEnvCfg")


def test_physx_config_preserves_policy_and_timing_contracts() -> None:
    """A Newton checkpoint sees the same ordered 8-D action and 60 Hz policy interface."""
    newton_cfg = ConveyorFrankaEnvCfg()
    physx_cfg = ConveyorFrankaPhysxEnvCfg()

    newton_cfg.validate()
    physx_cfg.validate()
    assert isinstance(physx_cfg.sim.physics, PhysxCfg)
    assert physx_cfg.sim.device == "cpu"
    assert physx_cfg.scene.num_envs == 1
    assert physx_cfg.sim.dt == newton_cfg.sim.dt == 1.0 / 120.0
    assert physx_cfg.decimation == newton_cfg.decimation == 2
    assert physx_cfg.actions.arm_action.joint_names == newton_cfg.actions.arm_action.joint_names
    assert physx_cfg.actions.gripper_action.joint_names == newton_cfg.actions.gripper_action.joint_names
    assert physx_cfg.conveyor_force.speed == newton_cfg.conveyor_force.speed == 0.35
    assert physx_cfg.scene.robot.spawn.joint_drive_props is None
    assert physx_cfg.scene.robot.spawn.rigid_props.disable_gravity is True


def test_physx_task_default_device_survives_an_unset_parser_override() -> None:
    """Default-backend callers can preserve the task's declared CPU device."""
    cfg = parse_env_cfg("IsaacContrib-Conveyor-Franka-PhysX-CPU-v0", device=None, num_envs=2)

    assert cfg.sim.device == "cpu"
    assert cfg.scene.num_envs == 2


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_physx_config_rejects_broken_gpu_surface_velocity_contacts(device: str) -> None:
    """The pinned Isaac Sim GPU path must not silently let cubes tunnel through belts."""
    cfg = ConveyorFrankaPhysxEnvCfg()
    cfg.sim.device = device

    with pytest.raises(ValueError, match="CPU-only.*--device cpu"):
        cfg.validate()


def test_curved_physx_sections_are_pivot_local_watertight_sdfs() -> None:
    """Turn roots sit at their pivots so native angular velocity has the intended center."""
    cfg = ConveyorFrankaPhysxEnvCfg()
    sections = physx_belt_section_specs("Left", velocity=0.35)

    assert len(sections) == 4
    for section, root_position in sections[2:]:
        assert isinstance(section.geometry, MeshSpec)
        assert section.belt.curved
        assert section.belt.radius == BELT_TURN_RADIUS
        assert section.belt.pivot_point == (0.0, 0.0, 0.0)
        assert root_position[2] == 0.0

    turn_asset = cfg.scene.conveyor_left_right_turn_collision
    assert turn_asset.spawn.collision_approximation == "sdf"
    assert isinstance(turn_asset.spawn.physics_material, PhysxRigidBodyMaterialCfg)
    assert turn_asset.spawn.physics_material.dynamic_friction == 0.5
    assert turn_asset.spawn.rigid_props.kinematic_enabled is True


def test_runtime_specs_and_material_override_have_one_source_of_truth() -> None:
    """Final overrides reach both the runtime view and native PhysX material authoring."""
    cfg = ConveyorFrankaPhysxEnvCfg()
    cfg.scene.configure_conveyor(friction_coefficient=0.42)
    specs = cfg.scene.build_conveyor_belt_specs(
        velocity=0.21,
        friction_coefficient=0.42,
        contact_threshold=0.95,
    )

    assert len(specs) == 8
    assert {spec.velocity for spec in specs} == {0.21}
    assert {spec.friction_coefficient for spec in specs} == {0.42}
    assert {spec.contact_threshold for spec in specs} == {0.95}
    for side in ("left", "right"):
        for section_key in ("top_straight", "bottom_straight", "right_turn", "left_turn"):
            asset = getattr(cfg.scene, f"conveyor_{side}_{section_key}_collision")
            assert asset.spawn.physics_material.dynamic_friction == 0.42
