# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free regression tests for the Franka deformable OvPhysX presets."""

import pytest
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import (
    PhysxDeformableBodyMaterialCfg,
    PhysxSurfaceDeformableBodyMaterialCfg,
)

from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import (
    FrankaClothCameraEnvCfg,
    FrankaClothEnvCfg,
)
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftCameraEnvCfg, FrankaSoftEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def test_soft_task_ovphysx_preset_selects_complete_authored_scene():
    """Test that the soft-task OvPhysX preset selects a fully authored scene."""
    cfg = resolve_presets(FrankaSoftEnvCfg(), ("ovphysx",))

    assert isinstance(cfg.sim.physics, OvPhysxCfg)
    assert cfg.scene.replicate_physics is True
    assert isinstance(cfg.scene.deformable.spawn.deformable_props, PhysxDeformableBodyPropertiesCfg)
    assert isinstance(cfg.scene.deformable.spawn.physics_material, PhysxDeformableBodyMaterialCfg)


def test_cloth_task_ovphysx_preset_selects_complete_authored_scene():
    """Test that the cloth-task OvPhysX preset selects a fully authored scene."""
    cfg = resolve_presets(FrankaClothEnvCfg(), ("ovphysx",))

    assert isinstance(cfg.sim.physics, OvPhysxCfg)
    assert cfg.scene.replicate_physics is True
    assert isinstance(cfg.scene.deformable.spawn.deformable_props, PhysxDeformableBodyPropertiesCfg)
    assert isinstance(cfg.scene.deformable.spawn.physics_material, PhysxSurfaceDeformableBodyMaterialCfg)
    assert cfg.events.robot_physics_material.params["asset_cfg"].body_names is None


@pytest.mark.parametrize(
    ("env_cfg_type", "material_type"),
    [
        (FrankaSoftCameraEnvCfg, PhysxDeformableBodyMaterialCfg),
        (FrankaClothCameraEnvCfg, PhysxSurfaceDeformableBodyMaterialCfg),
    ],
)
def test_camera_task_ovphysx_preset_selects_complete_authored_scene(env_cfg_type, material_type):
    """Test that each camera-task OvPhysX preset selects a fully authored scene."""
    cfg = resolve_presets(env_cfg_type(), ("ovphysx",))

    assert isinstance(cfg.sim.physics, OvPhysxCfg)
    assert cfg.scene.replicate_physics is True
    assert isinstance(cfg.scene.deformable.spawn.deformable_props, PhysxDeformableBodyPropertiesCfg)
    assert isinstance(cfg.scene.deformable.spawn.physics_material, material_type)


def test_cloth_rendering_variant_applies_deterministic_overrides():
    """Test that the rendering test applies overrides after preset resolution."""
    from rendering_test_utils import _configure_franka_camera_test_env_cfg

    expected_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }

    cfg = resolve_presets(FrankaClothCameraEnvCfg(), ("ovphysx",))
    _configure_franka_camera_test_env_cfg(cfg, "rgb")

    assert cfg.scene.num_envs == 4
    assert cfg.scene.replicate_physics is True
    assert cfg.scene.base_camera.data_types == ["rgb"]
    assert cfg.events.reset_deformable.params["position_range"] == expected_range
