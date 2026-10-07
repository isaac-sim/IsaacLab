# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for visualizer config construction and base visualizer behavior."""

from __future__ import annotations

import importlib.util

import pytest

from isaaclab.utils.string import ResolvableString
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg, SceneCameraCfg, VisualizerCfg

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
    assert cfg.background_color == (0.30, 0.55, 0.82)
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


def test_visualizer_cfg_validates_background_color():
    assert VisualizerCfg(background_color=None).background_color is None
    assert VisualizerCfg(background_color=[0, 0.5, 1]).background_color == (0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="three normalized RGB values"):
        VisualizerCfg(background_color=(0.0, 0.5, 1.1))


#
# Base visualizer (env filtering, camera pose)
#


class _DummyVisualizer(BaseVisualizer):
    def initialize(self, scene_data_provider, *, cameras, stage=None) -> None:
        self._scene_data_provider = scene_data_provider
        self._is_initialized = True

    def step(self, dt: float) -> None:
        return

    def close(self) -> None:
        self._is_closed = True

    def is_running(self) -> bool:
        return True


def _make_cfg(**kwargs):
    cfg = {
        "max_visible_envs": None,
        "visible_env_indices": None,
        # Default off in tests: contiguous cap-only path matches historical assertions.
        "randomly_sample_visible_envs": False,
    }
    cfg.update(kwargs)
    return VisualizerCfg(**cfg)


_HAS_ISAACLAB_VIZ = importlib.util.find_spec("isaaclab_visualizers") is not None


class _FakeProvider:
    def __init__(self, num_envs: int = 0):
        self._num_envs = num_envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def get_metadata(self) -> dict:
        return {"num_envs": self._num_envs}


def test_compute_visualized_env_ids_cap_only_returns_none():
    """Cap-only path: :meth:`_compute_visualized_env_ids` is ``None``.

    The cap is applied later by ``resolve_visible_env_indices``.
    """
    viz = _DummyVisualizer(_make_cfg(max_visible_envs=3, visible_env_indices=None))
    viz._scene_data_provider = _FakeProvider(num_envs=10)
    assert viz._compute_visualized_env_ids() is None


def test_compute_visualized_env_ids_from_visible_indices_filters_out_of_range():
    viz = _DummyVisualizer(_make_cfg(visible_env_indices=[-1, 0, 3, 99]))
    viz._scene_data_provider = _FakeProvider(num_envs=4)
    assert viz._compute_visualized_env_ids() == [0, 3]


@pytest.mark.skipif(not _HAS_ISAACLAB_VIZ, reason="isaaclab_visualizers not installed")
def test_compute_visualized_env_ids_random_cap_only_sorted_once():
    """Cap-only random mode returns a sorted sample; explicit indices ignore the flag."""
    cfg = _make_cfg(max_visible_envs=3, visible_env_indices=None, randomly_sample_visible_envs=True)
    viz = _DummyVisualizer(cfg)
    viz._scene_data_provider = _FakeProvider(num_envs=10)
    sampled = viz._compute_visualized_env_ids()
    assert sampled is not None and len(sampled) == 3
    assert sampled == sorted(sampled)
    assert len(set(sampled)) == 3
    assert all(0 <= i < 10 for i in sampled)

    cfg_explicit = _make_cfg(
        visible_env_indices=[1, 5],
        max_visible_envs=1,
        randomly_sample_visible_envs=True,
    )
    viz2 = _DummyVisualizer(cfg_explicit)
    viz2._scene_data_provider = _FakeProvider(num_envs=10)
    assert viz2._compute_visualized_env_ids() == [1, 5]


def test_physics_backend_returns_none_without_simulation_context():
    """physics_backend is None when no SimulationContext is active."""
    viz = _DummyVisualizer(_make_cfg())
    assert viz.physics_backend is None
