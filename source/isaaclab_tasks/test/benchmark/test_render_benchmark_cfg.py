# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``Isaac-RenderBenchmark-Franka-Cabinet`` task's registration and presets.

No Kit/GPU required: these only load and resolve the config through the registry, the same way
:mod:`scripts.benchmarks.benchmark_renderer` selects a preset before launching a run.
"""

import gymnasium as gym
import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.physics import PhysxAutoCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-RenderBenchmark-Franka-Cabinet"


def _load_cfg():
    return load_cfg_from_registry(_TASK, "env_cfg_entry_point")


def test_task_registered_with_direct_entry_point():
    spec = gym.spec(_TASK)

    assert spec.kwargs["env_cfg_entry_point"] == (
        "isaaclab_tasks.benchmark.render_benchmark.render_benchmark_env_cfg:RenderBenchmarkFrankaCabinetEnvCfg"
    )
    assert spec.disable_env_checker is True


def test_default_scene_and_articulations():
    cfg = _load_cfg()

    assert cfg.decimation == 2
    assert cfg.episode_length_s == pytest.approx(60.0)
    assert cfg.scene.num_envs == 4
    assert cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert cfg.scene.cabinet.prim_path == "{ENV_REGEX_NS}/Cabinet"
    assert cfg.joint_animation_amplitude == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("presets", "expected_type"),
    [
        ((), NewtonCfg),
        (("newton_mjwarp",), NewtonCfg),
        (("physx",), PhysxAutoCfg),
        (("isaacsim_physx",), PhysxCfg),
    ],
)
def test_physics_presets_resolve_to_expected_backend(presets, expected_type):
    cfg = resolve_presets(_load_cfg(), selected=presets)

    assert isinstance(cfg.sim.physics, expected_type)


_CAMERA_DATA_TYPE_PRESETS = [
    ("default", ["rgb"]),
    ("rgb", ["rgb"]),
    ("albedo", ["albedo"]),
    ("depth", ["depth"]),
    ("simple_shading_constant_diffuse", ["simple_shading_constant_diffuse"]),
    ("simple_shading_diffuse_mdl", ["simple_shading_diffuse_mdl"]),
    ("simple_shading_full_mdl", ["simple_shading_full_mdl"]),
]


@pytest.fixture(scope="module")
def render_benchmark_presets():
    """Collect every preset once, before any preset selection mutates the config in place."""
    return collect_presets(_load_cfg())


@pytest.mark.parametrize("preset_name,expected_data_types", _CAMERA_DATA_TYPE_PRESETS)
def test_camera_presets_resolve_to_expected_data_types(render_benchmark_presets, preset_name, expected_data_types):
    camera_presets = render_benchmark_presets["scene.tiled_camera"]
    assert preset_name in camera_presets, f"Preset '{preset_name}' not found in tiled_camera presets"
    resolved = camera_presets[preset_name]

    assert resolved.data_types == expected_data_types
    assert resolved.width > 0
    assert resolved.height > 0
    assert isinstance(resolved.renderer_cfg.newton_renderer, NewtonWarpRendererCfg)


def test_camera_resolution_defaults_to_256():
    """``BENCHMARK_RENDER_RESOLUTION`` is read once at import, so its effect is process-scoped;
    this documents the default a fresh process gets when the env var is unset."""
    cfg = resolve_presets(_load_cfg(), selected=())

    assert cfg.scene.tiled_camera.width == 256
    assert cfg.scene.tiled_camera.height == 256


def test_ground_tile_matches_default_env_spacing():
    cfg = _load_cfg()

    assert cfg.scene.ground.spawn.size[:2] == (cfg.scene.env_spacing, cfg.scene.env_spacing)
