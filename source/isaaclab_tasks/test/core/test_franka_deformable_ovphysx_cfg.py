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

from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothEnvCfg
from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
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


def test_cloth_rendering_variant_updates_all_event_presets():
    """Test that the rendering variant can override reset ranges before preset resolution."""
    from rendering_test_utils import _make_franka_cloth_camera_env_cfg

    expected_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }

    for preset_name, replicate_physics in (("newton_mjwarp_vbd", True), ("ovphysx", True)):
        cfg = resolve_presets(_make_franka_cloth_camera_env_cfg("rgb"), (preset_name,))

        assert cfg.scene.replicate_physics is replicate_physics
        assert cfg.events.reset_deformable.params["position_range"] == expected_range


@pytest.mark.parametrize(
    ("rendering_test_name", "factory_name"),
    (
        ("rendering_test_franka_soft", "_make_franka_soft_camera_env_cfg"),
        ("rendering_test_franka_cloth", "_make_franka_cloth_camera_env_cfg"),
    ),
)
def test_ovphysx_deformable_rendering_waits_for_reviewed_goldens(
    monkeypatch: pytest.MonkeyPatch,
    rendering_test_name: str,
    factory_name: str,
):
    """Test that OvPhysX deformable rendering skips before constructing an environment."""
    import rendering_test_utils

    monkeypatch.setattr(
        rendering_test_utils,
        factory_name,
        lambda _data_type: pytest.fail("environment construction must not run without reviewed golden images"),
    )

    rendering_test = getattr(rendering_test_utils, rendering_test_name)
    with pytest.raises(pytest.skip.Exception, match="reviewed OvPhysX golden images"):
        rendering_test("ovphysx", "ovrtx_renderer", "rgb", [])
