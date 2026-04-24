# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for visualizer config factory and base visualizer behavior."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer import Visualizer
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

#
# Config factory
#


def test_create_visualizer_raises_for_base_cfg():
    cfg = VisualizerCfg()
    with pytest.raises(ValueError, match="Cannot create visualizer from base VisualizerCfg class"):
        cfg.create_visualizer()


def test_create_visualizer_raises_for_unknown_type():
    cfg = VisualizerCfg(visualizer_type="unknown-backend")
    with pytest.raises(ValueError, match="not registered"):
        cfg.create_visualizer()


def test_create_visualizer_raises_import_error_when_backend_unavailable(monkeypatch):
    monkeypatch.setattr(Visualizer, "_get_module_name", classmethod(lambda cls, backend: "does.not.exist"))
    cfg = VisualizerCfg(visualizer_type="newton")
    with pytest.raises(ImportError, match="isaaclab_visualizers"):
        cfg.create_visualizer()


#
# Base visualizer (env filtering, camera pose)
#


class _DummyVisualizer(BaseVisualizer):
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
        "max_visible_envs": None,
        "visible_env_indices": None,
        # Default off in tests: contiguous cap-only path matches historical assertions.
        "randomly_sample_visible_envs": False,
    }
    cfg.update(kwargs)
    return SimpleNamespace(**cfg)


_HAS_ISAACLAB_VIZ = importlib.util.find_spec("isaaclab_visualizers") is not None


class _FakeProvider:
    def __init__(self, num_envs: int = 0, transforms: dict | None = None):
        self._num_envs = num_envs
        self._transforms = transforms

    def get_metadata(self) -> dict:
        return {"num_envs": self._num_envs}

    def get_camera_transforms(self):
        return self._transforms


def test_compute_visualized_env_ids_cap_only_returns_none():
    """Cap-only path: :meth:`_compute_visualized_env_ids` is ``None``.

    The cap is applied later by ``resolve_visible_env_indices``.
    """
    viz = _DummyVisualizer(_make_cfg(visible_env_indices=None))
    viz._scene_data_provider = _FakeProvider(num_envs=8)
    assert viz._compute_visualized_env_ids() is None


def test_compute_visualized_env_ids_from_visible_indices_filters_out_of_range():
    viz = _DummyVisualizer(_make_cfg(visible_env_indices=[-1, 0, 3, 99]))
    viz._scene_data_provider = _FakeProvider(num_envs=4)
    assert viz._compute_visualized_env_ids() == [0, 3]


@pytest.mark.skipif(not _HAS_ISAACLAB_VIZ, reason="isaaclab_visualizers not installed")
def test_partial_visualization_cap_only_uses_resolver():
    """With ``visible_env_indices`` unset, :func:`resolve_visible_env_indices` applies ``max_visible_envs``."""
    from isaaclab_visualizers.newton_adapter import resolve_visible_env_indices

    cfg = _make_cfg(max_visible_envs=3, visible_env_indices=None)
    viz = _DummyVisualizer(cfg)
    viz._scene_data_provider = _FakeProvider(num_envs=10)
    assert viz._compute_visualized_env_ids() is None
    assert resolve_visible_env_indices(None, cfg.max_visible_envs, 10) == [0, 1, 2]
    assert resolve_visible_env_indices(None, 3, 10) == [0, 1, 2]


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


@pytest.mark.skipif(not _HAS_ISAACLAB_VIZ, reason="isaaclab_visualizers not installed")
def test_explicit_visible_env_indices_truncated_by_max_visible_envs():
    """Explicit indices from :meth:`_compute_visualized_env_ids`; ``max_visible_envs`` truncates from the end."""
    from isaaclab_visualizers.newton_adapter import resolve_visible_env_indices

    cfg = _make_cfg(visible_env_indices=[0, 2, 4], max_visible_envs=1)
    viz = _DummyVisualizer(cfg)
    viz._scene_data_provider = _FakeProvider(num_envs=10)
    ids = viz._compute_visualized_env_ids()
    assert ids == [0, 2, 4]
    assert resolve_visible_env_indices(ids, cfg.max_visible_envs, 10) == [0]


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
