# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``Isaac-RizonSharpa-Coffee-Kitchen-Camera-Benchmark`` task's config, recording, and kitchen spawner.

No Kit/GPU required: the config is loaded and resolved through the registry, and the spawner runs on
a small stand-in for the coffee kitchen.
"""

import json
from pathlib import Path

import gymnasium as gym
import pytest

from pxr import Usd, UsdGeom, UsdLux

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.benchmark.coffee_kitchen_camera.coffee_kitchen_camera_env_cfg import (
    KITCHEN_CAMERA_COUNT,
    KITCHEN_LIGHT_COUNT,
)
from isaaclab_tasks.contrib.rizon_sharpa_coffee.robot import ARM_JOINT_NAMES, COFFEE_READY_POSITIONS
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-RizonSharpa-Coffee-Kitchen-Camera-Benchmark"
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
    assert cfg.scene.env_spacing == pytest.approx(8.0)
    assert cfg.scene.kitchen.prim_path == "{ENV_REGEX_NS}/Kitchen"
    assert cfg.scene.kitchen.spawn.usd_path.endswith("/kitchen.usda")
    assert cfg.scene.coffee is None and cfg.scene.water is None
    assert gym.spec(_TASK).disable_env_checker is True


def test_recording_matches_robot_and_step_rate():
    """The shipped recording drives the demo's arm joints from its ready pose at the environment's step rate."""
    cfg = _load_cfg()
    motion = json.loads(Path(cfg.actions.arm_motion.motion_file).read_text(encoding="utf-8"))

    assert motion["joint_names"] == list(ARM_JOINT_NAMES)
    assert motion["rate_hz"] * cfg.sim.dt * cfg.decimation == pytest.approx(1.0)
    assert all(len(row) == len(ARM_JOINT_NAMES) for row in motion["positions"])
    assert motion["positions"][0] == pytest.approx(COFFEE_READY_POSITIONS, abs=1e-5)


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


def _write_kitchen(path: Path, light_count: int) -> None:
    """A stand-in for the coffee kitchen: geometry, a ceiling, and the expected lights and cameras."""
    asset = Usd.Stage.CreateNew(str(path))
    asset.SetDefaultPrim(UsdGeom.Xform.Define(asset, "/Kitchen").GetPrim())
    UsdGeom.Mesh.Define(asset, "/Kitchen/KitchenRoom/Floor")
    UsdGeom.Mesh.Define(asset, "/Kitchen/KitchenRoom/Ceiling")
    for index in range(light_count):
        UsdLux.SphereLight.Define(asset, f"/Kitchen/Lights/Light{index}")
    for index in range(KITCHEN_CAMERA_COUNT):
        UsdGeom.Camera.Define(asset, f"/Kitchen/Cameras/Camera{index}")
    asset.Save()


def _spawn_kitchen(stage: Usd.Stage, path: Path) -> None:
    """Spawn the kitchen into two environments, each with a table whose studio floor the spawner hides."""
    for env_index in range(2):
        UsdGeom.Mesh.Define(stage, f"/World/envs/env_{env_index}/Table/StudioFloor")
    cfg = _load_cfg().scene.kitchen.spawn
    cfg.usd_path = str(path)
    cfg.func("/World/envs/env_.*/Kitchen", cfg)


def test_kitchen_spawner_removes_ceiling_lights_and_cameras(tmp_path, stage):
    """Every environment's copy keeps its geometry, loses the ceiling, lights and cameras, and hides the floor."""
    kitchen_path = tmp_path / "kitchen.usda"
    _write_kitchen(kitchen_path, KITCHEN_LIGHT_COUNT)
    _spawn_kitchen(stage, kitchen_path)

    for env_index in range(2):
        env = f"/World/envs/env_{env_index}"
        assert stage.GetPrimAtPath(f"{env}/Kitchen/KitchenRoom/Floor").IsActive()
        assert not stage.GetPrimAtPath(f"{env}/Kitchen/KitchenRoom/Ceiling").IsActive()
        assert not stage.GetPrimAtPath(f"{env}/Kitchen/Lights/Light0").IsActive()
        assert not stage.GetPrimAtPath(f"{env}/Kitchen/Cameras/Camera0").IsActive()
        floor = UsdGeom.Imageable(stage.GetPrimAtPath(f"{env}/Table/StudioFloor"))
        assert floor.ComputeVisibility() == UsdGeom.Tokens.invisible


def test_kitchen_spawner_rejects_changed_light_count(tmp_path, stage):
    """A kitchen bundle with an unexpected light count fails instead of adding lights to every environment."""
    kitchen_path = tmp_path / "kitchen.usda"
    _write_kitchen(kitchen_path, KITCHEN_LIGHT_COUNT + 1)

    with pytest.raises(RuntimeError, match=f"{KITCHEN_LIGHT_COUNT + 1} lights"):
        _spawn_kitchen(stage, kitchen_path)
