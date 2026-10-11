# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for visualizer config construction and base visualizer behavior."""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab.utils import validate
from isaaclab.utils.string import ResolvableString
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg, SceneCameraCfg, VisualizerCfg, WindowCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

#
# Config construction
#


@pytest.mark.parametrize(
    "module_name,cfg_name,implementation",
    [
        ("isaaclab_visualizers.kit", "KitVisualizerCfg", "KitVisualizer"),
        ("isaaclab_visualizers.newton", "NewtonGLVisualizerCfg", "NewtonGLVisualizer"),
        ("isaaclab_visualizers.newton", "NewtonRTXVisualizerCfg", "NewtonRTXVisualizer"),
        ("isaaclab_visualizers.rerun", "RerunVisualizerCfg", "RerunVisualizer"),
        ("isaaclab_visualizers.viser", "ViserVisualizerCfg", "ViserVisualizer"),
    ],
)
def test_visualizer_cfg_names_its_implementation(module_name, cfg_name, implementation):
    cfg_type = getattr(pytest.importorskip(module_name), cfg_name)
    cfg = cfg_type()
    class_type = cfg.class_type
    assert isinstance(class_type, ResolvableString)
    assert class_type.__name__ == implementation
    assert cfg.streaming_sensor_prim_path is None
    assert "cameras" in vars(cfg) and "camera" not in vars(cfg)
    assert not any(name.startswith(("streaming_cam_", "tiled_cam_")) for name in vars(cfg))


def test_visualizer_cfg_camera_sources():
    cfg = VisualizerCfg()
    assert cfg.focal_length == 12.0
    assert cfg.background_color is None
    assert cfg.streaming_view is False
    assert cfg.streaming_envs == 32
    assert cfg.streaming_sensor_prim_path is None
    cfg = VisualizerCfg(cameras=[SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Camera")])
    assert cfg.streaming_view
    # Resolve the source at initialization instead of copying its path between config fields.
    assert cfg.streaming_sensor_prim_path is None
    cfg = VisualizerCfg(cameras=[PerspectiveCameraCfg(eye=(1.0, 2.0, 3.0), focal_length=24.0)])
    assert (cfg.eye, cfg.focal_length, cfg.streaming_view) == ((1.0, 2.0, 3.0), 24.0, False)
    with pytest.raises(ValueError, match="at least one display source"):
        VisualizerCfg(cameras=[])


def test_visualizer_cfg_validates_presentation():
    assert VisualizerCfg(background_color=None).background_color is None
    assert VisualizerCfg(background_color=[0, 0.5, 1]).background_color == (0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="three normalized RGB values"):
        VisualizerCfg(background_color=(0.0, 0.5, 1.1))
    for size in ((0, 720), (1280, -1), (1280,), (1.5, 720)):
        with pytest.raises(ValueError, match="WindowCfg.size"):
            validate(WindowCfg(size=size))
    for fps in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="WindowCfg.fps"):
            validate(WindowCfg(fps=fps))


#
# Base visualizer (env filtering, camera pose)
#


class _DummyVisualizer(BaseVisualizer):
    def initialize(self, sim, *, cameras) -> None:
        super().initialize(sim, cameras=cameras)
        self._is_initialized = True

    def step(self, dt: float) -> None:
        return

    def close(self) -> None:
        self._is_closed = True

    def is_running(self) -> bool:
        return True


@pytest.mark.parametrize(
    "env_ids, cap, num_envs, expected",
    [
        (None, None, 10, None),
        (None, 3, 10, [0, 1, 2]),
        (None, 0, 10, []),
        (None, -1, 10, []),
        (None, 20, 3, [0, 1, 2]),
        (None, 5, 0, None),
        ([], 2, 10, []),
        ([5, -1, 3, 3, 99, 1], 3, 10, [5, 3, 1]),
        ([1, 3, 5], 2, 10, [1, 3]),
        ([1, 3], None, 10, [1, 3]),
    ],
)
def test_visualizer_initialization_resolves_visible_envs(env_ids, cap, num_envs, expected):
    cfg = VisualizerCfg(visible_env_indices=env_ids, max_visible_envs=cap, randomly_sample_visible_envs=False)
    viz = _DummyVisualizer(cfg)
    sim = Mock(stage=None)
    sim.get_scene_data_provider.return_value = SimpleNamespace(num_envs=num_envs)
    viz.initialize(sim, cameras=[])
    assert viz.get_visualized_env_ids() == expected


def test_visualizer_initialization_samples_visible_envs_once(monkeypatch):
    sample = Mock(wraps=random.sample)
    monkeypatch.setattr(random, "sample", sample)
    cfg = VisualizerCfg(max_visible_envs=3, randomly_sample_visible_envs=True)
    viz = _DummyVisualizer(cfg)
    sim = Mock(stage=None)
    sim.get_scene_data_provider.return_value = SimpleNamespace(num_envs=10)
    viz.initialize(sim, cameras=[])
    sampled = viz.get_visualized_env_ids()
    assert sampled is not None and len(sampled) == 3
    assert sampled == sorted(sampled)
    assert len(set(sampled)) == 3
    assert all(0 <= i < 10 for i in sampled)

    viz.reset(soft=True)
    assert viz.get_visualized_env_ids() == sampled

    cfg.visible_env_indices, cfg.max_visible_envs = [1, 5], 1
    viz.initialize(sim, cameras=[])
    assert viz.get_visualized_env_ids() == [1]
    assert sample.call_count == 1


def test_physics_backend_returns_none_without_simulation_context():
    """physics_backend is None when no SimulationContext is active."""
    viz = _DummyVisualizer(VisualizerCfg())
    assert viz.physics_backend is None
