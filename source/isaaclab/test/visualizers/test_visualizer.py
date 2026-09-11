# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for visualizer config construction and base visualizer behavior."""

from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace

import pytest
import torch

import isaaclab.visualizers as visualizers
from isaaclab.envs.utils.camera_view import apply_camera_view_from_origins, prim_world_positions
from isaaclab.utils.string import ResolvableString
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

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
    class_type = cfg_type().class_type
    assert isinstance(class_type, ResolvableString)
    assert class_type.__name__ == implementation


def test_visualizer_construction_has_no_factory_api():
    assert not hasattr(visualizers, "Visualizer")
    assert not any(
        hasattr(VisualizerCfg, name)
        for name in ("build", "build_visualizer", "clone_context", "create_visualizer", "get_visualizer_type")
    )


def test_visualizer_cfg_streaming_view_is_opt_in():
    cfg = VisualizerCfg()
    assert cfg.focal_length == 12.0
    assert cfg.background_color == (0.3, 0.55, 0.82)
    assert cfg.streaming_view is False
    assert cfg.streaming_envs == 32


def test_visualizer_cfg_validates_background_color():
    assert VisualizerCfg(background_color=None).background_color is None
    assert VisualizerCfg(background_color=[0, 0.5, 1]).background_color == (0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="three normalized RGB values"):
        VisualizerCfg(background_color=(0.0, 0.5, 1.1))


def test_streaming_cfg_fields_on_visualizer_cfg():
    """streaming_view is opt-in (False) and streaming_cam_renderer defaults to None."""
    cfg = VisualizerCfg()
    assert cfg.streaming_view is False
    assert cfg.streaming_cam_renderer is None


#
# Base visualizer (env filtering, camera pose)
#


class _DummyVisualizer(BaseVisualizer):
    def initialize(self, scene_data_provider) -> None:
        self._scene_data_provider = scene_data_provider
        self._is_initialized = True

    def step(self, dt: float) -> None:
        self._update_camera_tracking(dt)

    def set_camera_view(self, eye, target) -> None:
        self.camera_pose = (eye, target)

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

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def get_metadata(self) -> dict:
        return {"num_envs": self._num_envs}

    def get_camera_transforms(self):
        return self._transforms


class _FakeCamera:
    device = "cpu"

    def __init__(self):
        self.set_world_poses_from_view_calls = []
        self.update_poses_calls = []

    def set_world_poses_from_view(self, eyes, targets, env_ids=None):
        self.set_world_poses_from_view_calls.append((eyes.clone(), targets.clone(), env_ids))

    def _update_poses(self, dt):
        self.update_poses_calls.append(dt)


def test_apply_camera_view_from_origins_forwards_env_ids():
    camera = _FakeCamera()
    origins = torch.tensor([[1.0, 2.0, 3.0]])

    apply_camera_view_from_origins(camera, origins, eye=(0.5, 0.0, 1.0), lookat=(0.0, 0.0, 0.0), env_ids=[2])

    eyes, targets, env_ids = camera.set_world_poses_from_view_calls[0]
    assert eyes.tolist() == [[1.5, 2.0, 4.0]]
    assert targets.tolist() == [[1.0, 2.0, 3.0]]
    assert env_ids == [2]
    assert camera.update_poses_calls == [None]


def test_prim_world_positions_prefers_scene_articulation_state():
    body_pos_w = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
            [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
        ]
    )
    articulation = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/envs/env_[^/]+/Robot"),
        body_names=["base", "foot"],
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.zeros((2, 3))),
            body_pos_w=SimpleNamespace(torch=body_pos_w),
        ),
        find_bodies=lambda name, **_: ([0], [name]),
    )
    scene = SimpleNamespace(articulations={"robot": articulation})

    positions = prim_world_positions(None, "/World/envs/*/Robot/base", [1, 0], scene=scene)

    assert torch.equal(positions, torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]))


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


def test_physics_backend_returns_none_without_simulation_context():
    """physics_backend is None when no SimulationContext is active."""
    viz = _DummyVisualizer(_make_cfg())
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


def _camera_visualizer(monkeypatch, scene, **kwargs):
    from isaaclab.sim import SimulationContext

    monkeypatch.setattr(SimulationContext, "instance", lambda: SimpleNamespace(_interactive_scene=scene))
    cfg = VisualizerCfg(eye=(2.0, 0.0, 1.0), lookat=(1.0, 0.0, 0.0), **kwargs)
    viz = _DummyVisualizer(cfg)
    viz.initialize(_FakeProvider(scene.num_envs))
    return viz


@pytest.mark.parametrize("num_envs", [1, 5, 9, 64])
def test_camera_selects_spatial_center_once(monkeypatch, num_envs):
    from isaaclab.cloner.clone_plan import grid_transforms

    origins, _ = grid_transforms(num_envs)
    scene = _CameraScene(origins)
    viz = _camera_visualizer(monkeypatch, scene, origin_type="env", origin_env_index="center")
    viz.step(0.1)

    # Independently compare horizontal distances to the layout's bounding-box center.
    points = scene.env_origins.tolist()
    center = [(min(p[i] for p in points) + max(p[i] for p in points)) / 2 for i in (0, 1)]
    index = min(range(num_envs), key=lambda j: sum((points[j][i] - center[i]) ** 2 for i in (0, 1)))
    origin = scene.env_origins[index]
    assert viz.camera_env_index == index
    assert viz.camera_pose[0] == pytest.approx((origin + torch.tensor(viz.cfg.eye)).tolist())
    assert viz.camera_pose[1] == pytest.approx((origin + torch.tensor(viz.cfg.lookat)).tolist())
    if num_envs == 64:
        assert index == 27

    # Fixed cameras must not move when a terrain curriculum relocates environment origins.
    first_pose = viz.camera_pose
    scene.env_origins.add_(100.0)
    viz.step(0.1)
    assert viz.camera_pose == first_pose


def test_camera_center_uses_visible_envs_and_ignores_height(monkeypatch):
    scene = _CameraScene([[-4, 0, 0], [0, 0, 100], [1, 0, 0], [4, 0, 0]])
    viz = _camera_visualizer(monkeypatch, scene, origin_type="env", origin_env_index="center")
    viz._resolved_visible_env_ids = [3, 1]
    viz.step(0.1)
    assert viz.camera_pose == ((2.0, 0.0, 101.0), (1.0, 0.0, 100.0))


@pytest.mark.parametrize("follow_heading", [False, True])
@pytest.mark.parametrize("track_path", ["robot", "robot/base"])
def test_camera_tracks_position_and_optional_yaw_without_offset_drift(monkeypatch, follow_heading, track_path):
    from isaaclab.utils.math import quat_from_euler_xyz

    scene = _CameraScene([[0, 0, 0], [20, 0, 0], [10, 0, 0]])
    positions = torch.tensor([[100.0, 0, 0], [200.0, 0, 0], [10.0, 2, 3]])
    # Rz(90 degrees) * Ry(30 degrees) * Rx(60 degrees), xyzw quaternion.
    orientations = quat_from_euler_xyz(
        torch.full((3,), math.pi / 3), torch.full((3,), math.pi / 6), torch.full((3,), math.pi / 2)
    )
    body_positions = torch.stack((positions + 100, positions + torch.tensor([0, 0, 0.5])), dim=1)
    body_orientation = quat_from_euler_xyz(
        torch.full((3,), math.pi / 3), torch.full((3,), math.pi / 6), torch.full((3,), -math.pi / 2)
    )
    body_orientations = torch.stack((orientations, body_orientation), dim=1)
    scene["robot"] = _camera_asset(
        root_pos_w=positions, root_quat_w=orientations, body_pos_w=body_positions, body_quat_w=body_orientations
    )
    scene["robot"].find_bodies = lambda name: ([1], ["base"]) if name == "base" else ([], [])
    viz = _camera_visualizer(
        monkeypatch,
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


@pytest.mark.parametrize("env_index,visible", [(3, None), (-1, None), (1, [0]), ("center", [])])
def test_camera_rejects_unavailable_environment(monkeypatch, env_index, visible):
    viz = _camera_visualizer(
        monkeypatch, _CameraScene([[0, 0, 0], [1, 0, 0]]), origin_type="env", origin_env_index=env_index
    )
    viz._resolved_visible_env_ids = visible
    with pytest.raises(ValueError, match="environment|origin_env_index"):
        viz.step(0.1)


@pytest.mark.parametrize("path", [None, "missing", "robot/missing", "robot/.*"])
def test_camera_rejects_missing_or_ambiguous_asset_target(monkeypatch, path):
    scene = _CameraScene([[0, 0, 0]])
    scene["robot"] = SimpleNamespace(is_initialized=True, find_bodies=lambda name: ([], []))
    if path == "robot/.*":
        scene["robot"].find_bodies = lambda name: ([0, 1], ["base", "foot"])
    viz = _camera_visualizer(monkeypatch, scene, origin_type="asset", origin_track_path=path)
    with pytest.raises(ValueError, match="origin_track_path"):
        viz.step(0.1)


def test_camera_waits_for_asset_state_and_accepts_new_environment_selection(monkeypatch):
    scene = _CameraScene([[10, 20, 0], [30, 20, 0]])
    asset = SimpleNamespace(is_initialized=False)
    scene["robot"] = asset
    viz = _camera_visualizer(monkeypatch, scene, origin_type="asset", origin_track_path="robot")
    viz.step(0.1)
    assert not hasattr(viz, "camera_pose")
    viz._set_camera_pose_cfg((7.0, 12.0, 3.0), (4.0, 9.0, 1.0))
    asset.is_initialized = True
    asset.data = SimpleNamespace(root_pos_w=SimpleNamespace(torch=scene.env_origins))
    viz.step(0.1)
    assert viz.camera_pose == ((7.0, 12.0, 3.0), (4.0, 9.0, 1.0))
    viz.cfg.origin_env_index = 1
    viz.step(0.1)
    assert viz.camera_pose == ((27.0, 12.0, 3.0), (24.0, 9.0, 1.0))


@pytest.mark.parametrize("substeps", [1, 5, 20])
def test_camera_heading_filter_uses_elapsed_time_and_keeps_position_tracking(monkeypatch, substeps):
    scene = _CameraScene([[0, 0, 0]])
    positions = torch.zeros((1, 3))
    orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    scene["robot"] = _camera_asset(root_pos_w=positions, root_quat_w=orientations)
    viz = _camera_visualizer(
        monkeypatch,
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
    viz._set_camera_pose_cfg((7.0, 12.0, 3.0), (4.0, 9.0, 1.0))
    viz.step(0.0)
    assert viz.camera_pose[0] == pytest.approx((7.0, 12.0, 3.0), abs=1e-5)
    assert viz.camera_pose[1] == pytest.approx((4.0, 9.0, 1.0), abs=1e-5)


def test_camera_heading_filter_takes_short_path_and_reinitializes_on_target_change(monkeypatch):
    scene = _CameraScene([[0, 0, 0], [0, 0, 0]])
    orientations = torch.tensor(
        [
            [0.0, 0.0, math.sin(math.radians(179) / 2), math.cos(math.radians(179) / 2)],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scene["robot"] = _camera_asset(root_pos_w=scene.env_origins, root_quat_w=orientations)
    viz = _camera_visualizer(
        monkeypatch,
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


@pytest.mark.parametrize("time_constant", [-0.1, math.inf, math.nan])
def test_camera_rejects_invalid_heading_smoothing_time_constant(time_constant):
    with pytest.raises(ValueError, match="origin_heading_smoothing_time_constant"):
        VisualizerCfg(origin_heading_smoothing_time_constant=time_constant)
