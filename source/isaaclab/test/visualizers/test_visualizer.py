# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for visualizer config construction and base visualizer behavior."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

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
        super().initialize(scene_data_provider, cameras=cameras, stage=stage)
        self._is_initialized = True

    def step(self, dt: float) -> None:
        self._update_camera_tracking(dt)

    def _apply_camera_pose(self, pose) -> None:
        self.camera_pose = pose

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
    viz.initialize(SimpleNamespace(num_envs=num_envs), cameras=[])
    assert viz.get_visualized_env_ids() == expected


def test_visualizer_initialization_samples_visible_envs_once(monkeypatch):
    sample = Mock(wraps=random.sample)
    monkeypatch.setattr(random, "sample", sample)
    cfg = VisualizerCfg(max_visible_envs=3, randomly_sample_visible_envs=True)
    viz = _DummyVisualizer(cfg)
    viz.initialize(SimpleNamespace(num_envs=10), cameras=[])
    sampled = viz.get_visualized_env_ids()
    assert sampled is not None and len(sampled) == 3
    assert sampled == sorted(sampled)
    assert len(set(sampled)) == 3
    assert all(0 <= i < 10 for i in sampled)

    viz.reset(soft=True)
    assert viz.get_visualized_env_ids() == sampled

    cfg.visible_env_indices, cfg.max_visible_envs = [1, 5], 1
    viz.initialize(SimpleNamespace(num_envs=10), cameras=[])
    assert viz.get_visualized_env_ids() == [1]
    assert sample.call_count == 1


def test_physics_backend_returns_none_without_simulation_context():
    """physics_backend is None when no SimulationContext is active."""
    viz = _DummyVisualizer(VisualizerCfg())
    assert viz.physics_backend is None


class _CameraScene(dict):
    def __init__(self, origins):
        super().__init__()
        self.env_origins = torch.as_tensor(origins, dtype=torch.float32)
        self.num_envs = len(origins)


def _camera_asset(**state):
    return SimpleNamespace(
        is_initialized=True,
        data=SimpleNamespace(**{name: SimpleNamespace(torch=value) for name, value in state.items()}),
    )


def _camera_visualizer(scene, **kwargs):
    cfg = VisualizerCfg(eye=(2.0, 0.0, 1.0), lookat=(1.0, 0.0, 0.0), **kwargs)
    viz = _DummyVisualizer(cfg)
    viz.initialize(SimpleNamespace(num_envs=scene.num_envs, get_interactive_scene=lambda: scene), cameras=[])
    return viz


@pytest.mark.parametrize(
    ("origins", "visible", "index", "expected"),
    [
        # Four environments share the center of the 8x8 grid; select the lowest index.
        ("grid", None, "center", 27),
        # Only visible environments are candidates, and height is ignored.
        ([[-4, 0, 0], [0, 0, 100], [1, 0, 0], [4, 0, 0]], [3, 1], "center", 1),
        # A hidden explicit index falls back to the nearest visible environment.
        ([[-4, 0, 0], [0, 0, 100], [1, 0, 0], [4, 0, 0]], [3, 0], 2, 3),
    ],
)
def test_camera_selects_visible_env_once(origins, visible, index, expected):
    if origins == "grid":
        from isaaclab.cloner.clone_plan import grid_transforms

        origins, _ = grid_transforms(64)
    scene = _CameraScene(origins)
    viz = _camera_visualizer(scene, origin_type="env", origin_env_index=index)
    viz._env_ids = visible
    viz.step(0.1)

    origin = scene.env_origins[expected]
    assert viz.camera_env_index == expected
    assert viz.camera_pose[0] == pytest.approx((origin + torch.tensor(viz.cfg.eye)).tolist())
    assert viz.camera_pose[1] == pytest.approx((origin + torch.tensor(viz.cfg.lookat)).tolist())

    # Fixed cameras must not move when a terrain curriculum relocates environment origins.
    first_pose = viz.camera_pose
    scene.env_origins.add_(100.0)
    viz.step(0.1)
    assert viz.camera_pose == first_pose


@pytest.mark.parametrize("follow_heading", [False, True])
@pytest.mark.parametrize("track_path", ["robot", "robot/base"])
def test_camera_tracks_position_and_optional_yaw_without_offset_drift(follow_heading, track_path):
    scene = _CameraScene([[0, 0, 0], [20, 0, 0], [10, 0, 0]])
    positions = torch.tensor([[100.0, 0, 0], [200.0, 0, 0], [10.0, 2, 3]])
    # XYZW quaternions: 90-degree roll, with +90-degree root yaw and -90-degree body yaw.
    orientations = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).expand(3, -1)
    body_positions = torch.stack((positions + 100, positions + torch.tensor([0, 0, 0.5])), dim=1)
    body_orientation = torch.tensor([[0.5, -0.5, -0.5, 0.5]]).expand(3, -1)
    body_orientations = torch.stack((orientations, body_orientation), dim=1)
    scene["robot"] = _camera_asset(
        root_pos_w=positions, root_quat_w=orientations, body_pos_w=body_positions, body_quat_w=body_orientations
    )
    scene["robot"].find_bodies = lambda name: ([1], ["base"]) if name == "base" else ([], [])
    viz = _camera_visualizer(
        scene,
        origin_type="asset",
        origin_env_index="center",
        origin_track_path=track_path,
        origin_follow_heading=follow_heading,
    )
    viz.step(0.1)
    for origin in ([10.0, 2, 3], [20.0, -5, 2], [-10.0, 8, 1]):
        positions[2] = torch.tensor(origin)
        body_positions[2, 1] = positions[2] + torch.tensor([0, 0, 0.5])
        scene.env_origins[2] += 100  # Reset/terrain movement must not change the selected robot.
        viz.step(0.1)
        direction = -1 if track_path == "robot/base" else 1
        height = 0.5 if track_path == "robot/base" else 0
        eye_offset = [0, direction * 2, 1 + height] if follow_heading else [2, 0, 1 + height]
        target_offset = [0, direction, height] if follow_heading else [1, 0, height]
        assert viz.camera_pose[0] == pytest.approx([a + b for a, b in zip(origin, eye_offset)], abs=1e-5)
        assert viz.camera_pose[1] == pytest.approx([a + b for a, b in zip(origin, target_offset)], abs=1e-5)
        assert viz.cfg.eye == (2.0, 0.0, 1.0)
        assert viz.cfg.lookat == (1.0, 0.0, 0.0)


def test_camera_waits_for_asset_state_and_accepts_new_environment_selection():
    scene = _CameraScene([[10, 20, 0], [30, 20, 0]])
    asset = SimpleNamespace(is_initialized=False)
    scene["robot"] = asset
    viz = _camera_visualizer(scene, origin_type="asset", origin_track_path="robot")
    viz.step(0.1)
    assert not hasattr(viz, "camera_pose")
    viz.set_camera_view((7.0, 12.0, 3.0), (4.0, 9.0, 1.0))
    asset.is_initialized = True
    asset.data = SimpleNamespace(root_pos_w=SimpleNamespace(torch=scene.env_origins))
    viz.step(0.1)
    assert viz.camera_pose == ((7.0, 12.0, 3.0), (4.0, 9.0, 1.0))
    viz.cfg.origin_env_index = 1
    viz.step(0.1)
    assert viz.camera_pose == ((27.0, 12.0, 3.0), (24.0, 9.0, 1.0))


@pytest.mark.parametrize("substeps", [1, 20])
def test_camera_heading_filter_uses_elapsed_time_and_keeps_position_tracking(substeps):
    scene = _CameraScene([[0, 0, 0]])
    positions = torch.zeros((1, 3))
    orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    scene["robot"] = _camera_asset(root_pos_w=positions, root_quat_w=orientations)
    viz = _camera_visualizer(
        scene,
        origin_type="asset",
        origin_track_path="robot",
        origin_follow_heading=True,
        origin_heading_smoothing_time_constant=0.2,
    )
    viz.step(0.0)
    orientations[0] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
    positions[0] = torch.tensor([10.0, 20.0, 3.0])
    # One half-life must produce 45 degrees, regardless of the render cadence.
    for _ in range(substeps):
        viz.step(0.2 * math.log(2) / substeps)
    assert viz.camera_pose[0] == pytest.approx((10 + 2**0.5, 20 + 2**0.5, 4), abs=1e-5)
    assert viz.camera_pose[1] == pytest.approx((10 + 2**-0.5, 20 + 2**-0.5, 3), abs=1e-5)
    pose = viz.camera_pose
    viz.step(0.0)
    assert viz.camera_pose == pose
    # UI edits are converted through the filtered heading, without a jump while paused.
    viz.set_camera_view((7.0, 12.0, 3.0), (4.0, 9.0, 1.0))
    viz.step(0.0)
    assert viz.camera_pose[0] == pytest.approx((7.0, 12.0, 3.0), abs=1e-5)
    assert viz.camera_pose[1] == pytest.approx((4.0, 9.0, 1.0), abs=1e-5)


def test_camera_heading_filter_takes_short_path_and_reinitializes_on_target_change():
    scene = _CameraScene([[0, 0, 0], [0, 0, 0]])
    orientations = torch.tensor(
        [
            [0.0, 0.0, math.sin(math.radians(179) / 2), math.cos(math.radians(179) / 2)],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scene["robot"] = _camera_asset(root_pos_w=scene.env_origins, root_quat_w=orientations)
    viz = _camera_visualizer(
        scene,
        origin_type="asset",
        origin_track_path="robot",
        origin_follow_heading=True,
        origin_heading_smoothing_time_constant=0.2,
    )
    viz.step(0.0)
    assert viz.camera_pose[0] == pytest.approx(
        (2 * math.cos(math.radians(179)), 2 * math.sin(math.radians(179)), 1), abs=1e-5
    )
    orientations[0, 2] *= -1
    viz.step(0.2 * math.log(2))
    assert viz.camera_pose[0] == pytest.approx((-2.0, 0.0, 1.0), abs=1e-5)
    viz.cfg.origin_env_index = 1
    viz.step(0.0)
    assert viz.camera_pose[0] == pytest.approx((2.0, 0.0, 1.0), abs=1e-5)
