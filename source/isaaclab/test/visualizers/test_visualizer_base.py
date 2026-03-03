"""Unit tests for visualizer base behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isaaclab.visualizers.visualizer import Visualizer


class _DummyVisualizer(Visualizer):
    def initialize(self, scene_data_provider) -> None:
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
        "env_filter_mode": "none",
        "env_filter_ids": [0, 2, 4],
        "env_filter_random_n": 2,
        "env_filter_seed": 7,
    }
    cfg.update(kwargs)
    return SimpleNamespace(**cfg)


class _FakeProvider:
    def __init__(self, num_envs: int = 0, transforms: dict | None = None):
        self._num_envs = num_envs
        self._transforms = transforms

    def get_metadata(self) -> dict:
        return {"num_envs": self._num_envs}

    def get_camera_transforms(self):
        return self._transforms


def test_compute_visualized_env_ids_none_mode():
    viz = _DummyVisualizer(_make_cfg(env_filter_mode="none"))
    viz._scene_data_provider = _FakeProvider(num_envs=8)
    assert viz._compute_visualized_env_ids() is None


def test_compute_visualized_env_ids_from_ids_filters_out_of_range():
    viz = _DummyVisualizer(_make_cfg(env_filter_mode="env_ids", env_filter_ids=[-1, 0, 3, 99]))
    viz._scene_data_provider = _FakeProvider(num_envs=4)
    assert viz._compute_visualized_env_ids() == [0, 3]


def test_compute_visualized_env_ids_random_n_is_deterministic():
    cfg = _make_cfg(env_filter_mode="random_n", env_filter_random_n=3, env_filter_seed=123)
    viz_a = _DummyVisualizer(cfg)
    viz_b = _DummyVisualizer(cfg)
    viz_a._scene_data_provider = _FakeProvider(num_envs=10)
    viz_b._scene_data_provider = _FakeProvider(num_envs=10)
    assert viz_a._compute_visualized_env_ids() == viz_b._compute_visualized_env_ids()


def test_resolve_camera_pose_from_usd_path_uses_provider_transforms():
    transforms = {
        "order": ["/World/envs/env_%d/Camera"],
        "positions": [[[1.0, 2.0, 3.0]]],
        "orientations": [[[0.0, 0.0, 0.0, 1.0]]],
    }
    viz = _DummyVisualizer(_make_cfg())
    viz._scene_data_provider = _FakeProvider(num_envs=1, transforms=transforms)
    pos, target = viz._resolve_camera_pose_from_usd_path("/World/envs/env_0/Camera")
    assert pos == (1.0, 2.0, 3.0)
    assert target == pytest.approx((1.0, 2.0, 2.0))
