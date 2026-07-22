# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free regression tests for the Franka deformable OvPhysX presets."""

from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import (
    PhysxDeformableBodyMaterialCfg,
    PhysxSurfaceDeformableBodyMaterialCfg,
)

from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def test_soft_task_ovphysx_preset_selects_complete_authored_scene():
    """Test that the soft-task OvPhysX preset selects a fully authored scene."""
    cfg = resolve_presets(FrankaSoftEnvCfg(), ("ovphysx",))

    assert isinstance(cfg.sim.physics, OvPhysxCfg)
    assert cfg.scene.replicate_physics is False
    assert isinstance(cfg.scene.deformable.spawn.deformable_props, PhysxDeformableBodyPropertiesCfg)
    assert isinstance(cfg.scene.deformable.spawn.physics_material, PhysxDeformableBodyMaterialCfg)


def test_cloth_task_ovphysx_preset_selects_complete_authored_scene():
    """Test that the cloth-task OvPhysX preset selects a fully authored scene."""
    cfg = resolve_presets(FrankaClothEnvCfg(), ("ovphysx",))

    assert isinstance(cfg.sim.physics, OvPhysxCfg)
    assert cfg.scene.replicate_physics is False
    assert isinstance(cfg.scene.deformable.spawn.deformable_props, PhysxDeformableBodyPropertiesCfg)
    assert isinstance(cfg.scene.deformable.spawn.physics_material, PhysxSurfaceDeformableBodyMaterialCfg)
