# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for visualizer config construction and base visualizer behavior."""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import warp as wp

from isaaclab.utils import validate
from isaaclab.utils.string import ResolvableString
from isaaclab.utils.warp import ProxyArray
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.image_view import ImageView
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

    def set_camera_view(self, eye, target) -> None:
        self.camera_pose = (tuple(eye), tuple(target))
        self.cfg.eye, self.cfg.lookat = tuple(eye), tuple(target)

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


@pytest.fixture
def tracking_scene():
    """Two distinct environment and asset positions expose wrong-instance tracking."""
    asset = SimpleNamespace(
        is_initialized=True,
        data=SimpleNamespace(
            root_pos_w=ProxyArray(wp.array([[0.0, 0.0, 1.0], [12.0, 24.0, 2.0]], dtype=wp.vec3, device="cpu")),
            body_pos_w=ProxyArray(
                wp.array(
                    [[[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]], [[12.0, 24.0, 2.0], [13.0, 25.0, 4.0]]],
                    dtype=wp.vec3,
                    device="cpu",
                )
            ),
        ),
        find_bodies=Mock(return_value=([1], ["torso_link"])),
    )
    scene = MagicMock(num_envs=2, env_origins=np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 0.0]]))
    scene.__getitem__.side_effect = {"robot": asset}.__getitem__
    sim = Mock(stage=None)
    sim.get_scene_data_provider.return_value = SimpleNamespace(num_envs=2, get_interactive_scene=lambda: scene)
    sim.get_physics_step_count.return_value = 1
    sim.get_or_create_backend.side_effect = lambda cfg, **kwargs: ImageView(cfg, **kwargs)
    return sim, asset


@pytest.mark.parametrize(
    "origin_type,path,expected_eye,expected_target",
    [
        ("env", None, (12.0, 22.0, 2.0), (10.0, 20.0, 0.0)),
        ("asset", "robot", (14.0, 26.0, 4.0), (12.0, 24.0, 2.0)),
        ("asset", "robot/torso_link", (15.0, 27.0, 6.0), (13.0, 25.0, 4.0)),
    ],
)
def test_camera_tracking_selects_origin_and_preserves_offsets(
    tracking_scene, origin_type, path, expected_eye, expected_target
):
    """Camera offsets stay relative when tracking an environment, root, or body."""
    sim, asset = tracking_scene
    viz = _DummyVisualizer(
        VisualizerCfg(
            eye=(2.0, 2.0, 2.0),
            lookat=(0.0, 0.0, 0.0),
            origin_type=origin_type,
            origin_track_path=path,
            origin_env_index=1,
        )
    )
    viz.initialize(sim, cameras=[])
    viz.update_camera_tracking()
    assert viz.camera_pose == (expected_eye, expected_target)
    asset.data.root_pos_w.warp.numpy()[1, 0] += 5.0
    asset.data.body_pos_w.warp.numpy()[1, 1, 0] += 5.0
    sim.get_physics_step_count.return_value = 2
    viz.update_camera_tracking()
    displacement = 0.0 if origin_type == "env" else 5.0
    assert viz.camera_pose == (
        (expected_eye[0] + displacement, *expected_eye[1:]),
        (expected_target[0] + displacement, *expected_target[1:]),
    )
    assert viz.cfg.eye == (2.0, 2.0, 2.0)
    assert viz.cfg.lookat == (0.0, 0.0, 0.0)


def test_perspective_capture_tracks_before_render_and_reset(tracking_scene):
    """Recording without a display step follows the asset and refreshes after reset."""
    sim, asset = tracking_scene
    viz = _DummyVisualizer(
        VisualizerCfg(
            visualizer_type="newton_gl",
            eye=(2.0, 2.0, 2.0),
            origin_type="asset",
            origin_track_path="robot",
            origin_env_index=1,
        )
    )
    viz.initialize(sim, cameras=[PerspectiveCameraCfg()])
    view = viz._image_views[0]
    poses = []

    def capture(_):
        poses.append(viz.camera_pose)
        return wp.zeros((1, 1, 4), dtype=wp.uint8, device="cpu")

    view.render = capture
    view.read(1)
    assert poses == [((14.0, 26.0, 4.0), (12.0, 24.0, 2.0))]
    view.read(1)
    assert len(poses) == 1
    asset.data.root_pos_w.warp.numpy()[1] = [1.0, 2.0, 3.0]
    viz.reset(soft=True)
    view.invalidate()
    view.read(1)
    assert poses[-1] == ((3.0, 4.0, 5.0), (1.0, 2.0, 3.0))


def test_tracking_does_not_move_scene_camera_views(tracking_scene):
    """Selecting a sensor view must not turn tracking into sensor navigation."""
    sim, _ = tracking_scene
    viz = _DummyVisualizer(VisualizerCfg(origin_type="asset", origin_track_path="robot"))
    viz.initialize(sim, cameras=[])
    viz.current_image_view = SimpleNamespace(camera=object())
    viz.update_camera_tracking()
    assert not hasattr(viz, "camera_pose")


@pytest.mark.parametrize("env_index,path", [(2, "robot"), (0, None)])
def test_camera_tracking_rejects_invalid_origin(tracking_scene, env_index, path):
    sim, _ = tracking_scene
    viz = _DummyVisualizer(
        VisualizerCfg(
            origin_type="asset",
            origin_track_path=path,
            origin_env_index=env_index,
        )
    )
    with pytest.raises(ValueError, match="origin_env_index|origin_track_path"):
        viz.initialize(sim, cameras=[])
