# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``Isaac-Reach-Franka-Kitchen-Camera`` task's registration, presets, and kitchen spawner.

No Kit/GPU required: the config is loaded and resolved through the registry, and the spawner runs on
a small stand-in for the Kitchen Set.
"""

from dataclasses import MISSING

import gymnasium as gym
import pytest

from pxr import Gf, Usd, UsdGeom, UsdLux

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-Reach-Franka-Kitchen-Camera"
_DATA_TYPES = (
    "rgb",
    "depth",
    "albedo",
    "simple_shading_constant_diffuse",
    "simple_shading_diffuse_mdl",
    "simple_shading_full_mdl",
    "semantic_segmentation",
)
_RESOLUTIONS = (64, 128, 256)


def _load_cfg():
    return load_cfg_from_registry(_TASK, "env_cfg_entry_point")


def test_default_scene():
    cfg = _load_cfg()

    assert cfg.scene.num_envs == 256
    assert cfg.scene.env_spacing == pytest.approx(10.0)
    assert cfg.scene.kitchen.prim_path == "{ENV_REGEX_NS}/Kitchen"
    assert type(cfg.scene.kitchen.spawn.usd_path) is type(MISSING)
    assert cfg.scene.kitchen.spawn.scale == (0.01, 0.01, 0.01)
    assert cfg.scene.kitchen.init_state.pos == (-0.71, 0.28, -1.05)
    assert gym.spec(_TASK).disable_env_checker is True


@pytest.fixture(scope="module")
def kitchen_presets():
    """Collect every preset once, before any preset selection mutates the config in place."""
    return collect_presets(_load_cfg())


def test_camera_presets_resolve_to_expected_data_types(kitchen_presets):
    """Each camera preset requests the data type and resolution it is named after (``default`` is ``rgb64``)."""
    camera_presets = kitchen_presets["scene.camera"]
    expected = {
        f"{data_type}{resolution}": (data_type, resolution) for data_type in _DATA_TYPES for resolution in _RESOLUTIONS
    }
    assert set(camera_presets) == {"default", *expected}
    expected["default"] = ("rgb", 64)
    for preset_name, resolved in camera_presets.items():
        data_type, resolution = expected[preset_name]
        assert resolved.data_types == [data_type]
        assert (resolved.width, resolved.height) == (resolution, resolution)


def test_one_preset_selects_camera_and_image_term():
    cfg = resolve_presets(_load_cfg(), selected=("simple_shading_full_mdl64",))

    assert cfg.scene.camera.data_types == ["simple_shading_full_mdl"]
    assert (cfg.scene.camera.width, cfg.scene.camera.height) == (64, 64)
    assert cfg.observations.camera_image.image.params["data_type"] == "simple_shading_full_mdl"
    assert cfg.observations.camera_image.image.params["sensor_cfg"].name == "camera"


@pytest.fixture
def stage():
    """A new in-memory stage, current only for the test so the previous stage is restored afterwards."""
    stage = Usd.Stage.CreateInMemory()
    with sim_utils.use_stage(stage):
        yield stage


def test_kitchen_spawner_deactivates_set_lights(tmp_path, stage):
    """Every environment's copy keeps its geometry and loses the set's lights."""
    asset_path = tmp_path / "Kitchen_set.usda"
    asset = Usd.Stage.CreateNew(str(asset_path))
    asset.SetDefaultPrim(UsdGeom.Xform.Define(asset, "/Kitchen_set").GetPrim())
    UsdGeom.Mesh.Define(asset, "/Kitchen_set/Props/Mesh")
    UsdLux.SphereLight.Define(asset, "/Kitchen_set/SphereLight")
    UsdLux.RectLight.Define(asset, "/Kitchen_set/Props/RectLight")
    asset.Save()

    for env_index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_index}")
    cfg = _load_cfg().scene.kitchen.spawn
    cfg.usd_path = str(asset_path)
    cfg.func("/World/envs/env_.*/Kitchen", cfg)

    for env_index in range(2):
        kitchen = f"/World/envs/env_{env_index}/Kitchen"
        assert stage.GetPrimAtPath(f"{kitchen}/Props/Mesh").IsActive()
        assert not stage.GetPrimAtPath(f"{kitchen}/SphereLight").IsActive()
        assert not stage.GetPrimAtPath(f"{kitchen}/Props/RectLight").IsActive()
        transform = Gf.Transform(UsdGeom.Xformable(stage.GetPrimAtPath(kitchen)).GetLocalTransformation())
        assert tuple(transform.GetScale()) == pytest.approx((0.01, 0.01, 0.01))
