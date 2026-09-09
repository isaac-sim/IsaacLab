# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``Isaac-RenderBenchmark-Franka-Cabinet`` task's registration and presets.

No Kit/GPU required: these only load and resolve the config through the registry, the same way
:mod:`scripts.benchmarks.benchmark_renderer` selects a preset before launching a run.
"""

from types import SimpleNamespace

import gymnasium as gym
import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.render_benchmark.render_benchmark_env import RenderBenchmarkEnv
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-RenderBenchmark-Franka-Cabinet"


def _load_cfg():
    return load_cfg_from_registry(_TASK, "env_cfg_entry_point")


def test_task_registered_with_direct_entry_point():
    spec = gym.spec(_TASK)

    assert spec.kwargs["env_cfg_entry_point"] == (
        "isaaclab_tasks.core.render_benchmark.render_benchmark_env_cfg:RenderBenchmarkFrankaCabinetEnvCfg"
    )
    assert spec.disable_env_checker is True


def test_default_scene_and_articulations():
    cfg = _load_cfg()

    assert cfg.decimation == 2
    assert cfg.episode_length_s == pytest.approx(60.0)
    assert cfg.scene.num_envs == 4
    assert set(cfg.articulations) == {"robot", "cabinet"}
    assert cfg.joint_animation_amplitude == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("presets", "expected_type"),
    [((), NewtonCfg), (("newton_mjwarp",), NewtonCfg), (("physx",), PhysxCfg)],
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
    camera_presets = render_benchmark_presets["tiled_camera"]
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

    assert cfg.tiled_camera.width == 256
    assert cfg.tiled_camera.height == 256


def _fake_env_for_ground_cfg(ground_size: tuple[float, float], env_spacing: float) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(
            ground_size=ground_size, ground_thickness=0.1, ground_color=(0.5, 0.5, 0.5), ground_top_z=0.0
        ),
        scene=SimpleNamespace(cfg=SimpleNamespace(env_spacing=env_spacing)),
    )


def test_ground_cfg_clamps_size_to_env_spacing():
    """An oversized ground tile must be clamped, or cloned environments' grounds overlap."""
    fake_env = _fake_env_for_ground_cfg(ground_size=(50.0, 50.0), env_spacing=3.0)

    ground_cfg = RenderBenchmarkEnv._ground_cfg(fake_env)

    assert ground_cfg.spawn.size[:2] == (3.0, 3.0)


def test_ground_cfg_keeps_size_smaller_than_env_spacing():
    """A tile already smaller than the env spacing must not be inflated."""
    fake_env = _fake_env_for_ground_cfg(ground_size=(2.0, 1.5), env_spacing=3.0)

    ground_cfg = RenderBenchmarkEnv._ground_cfg(fake_env)

    assert ground_cfg.spawn.size[:2] == (2.0, 1.5)
