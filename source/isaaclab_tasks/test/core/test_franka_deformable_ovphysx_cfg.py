# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the Franka deformable OvPhysX presets."""

from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import (
    PhysxDeformableBodyMaterialCfg,
    PhysxSurfaceDeformableBodyMaterialCfg,
)

from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import (
    FrankaClothCameraEnvCfg,
    FrankaClothCameraScenePresetCfg,
    FrankaClothEnvCfg,
)
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import (
    FrankaSoftCameraEnvCfg,
    FrankaSoftCameraSceneCfg,
    FrankaSoftEnvCfg,
)
from isaaclab_tasks.utils.hydra import resolve_presets


def test_soft_task_ovphysx_preset():
    cfg = resolve_presets(FrankaSoftEnvCfg(), ("ovphysx",))

    assert isinstance(cfg.sim.physics, OvPhysxCfg)
    assert cfg.scene.replicate_physics is True
    assert isinstance(cfg.scene.deformable.spawn.deformable_props, PhysxDeformableBodyPropertiesCfg)
    assert isinstance(cfg.scene.deformable.spawn.physics_material, PhysxDeformableBodyMaterialCfg)
    assert cfg.events.variable_gravity is None
    assert cfg.curriculum.gravity is None


def test_cloth_task_ovphysx_preset():
    cfg = resolve_presets(FrankaClothEnvCfg(), ("ovphysx",))

    assert isinstance(cfg.sim.physics, OvPhysxCfg)
    assert cfg.scene.replicate_physics is True
    assert isinstance(cfg.scene.deformable.spawn.deformable_props, PhysxDeformableBodyPropertiesCfg)
    assert isinstance(cfg.scene.deformable.spawn.physics_material, PhysxSurfaceDeformableBodyMaterialCfg)
    assert cfg.events.variable_gravity is None
    assert cfg.curriculum.gravity is None


def test_camera_tasks_have_explicit_ovphysx_presets():
    assert "ovphysx" in FrankaSoftCameraSceneCfg.__dataclass_fields__
    assert "ovphysx" in FrankaClothCameraScenePresetCfg.__dataclass_fields__

    soft_cfg = resolve_presets(FrankaSoftCameraEnvCfg(), ("ovphysx",))
    cloth_cfg = resolve_presets(FrankaClothCameraEnvCfg(), ("ovphysx",))

    assert isinstance(soft_cfg.sim.physics, OvPhysxCfg)
    assert soft_cfg.scene.replicate_physics is True
    assert isinstance(cloth_cfg.sim.physics, OvPhysxCfg)
    assert cloth_cfg.scene.replicate_physics is True


def test_isaacsim_physx_preserves_gravity_curriculum():
    soft_cfg = resolve_presets(FrankaSoftEnvCfg(), ("isaacsim_physx",))
    cloth_cfg = resolve_presets(FrankaClothEnvCfg(), ("isaacsim_physx",))

    assert soft_cfg.events.variable_gravity is not None
    assert soft_cfg.curriculum.gravity is not None
    assert cloth_cfg.events.variable_gravity is not None
    assert cloth_cfg.curriculum.gravity is not None
