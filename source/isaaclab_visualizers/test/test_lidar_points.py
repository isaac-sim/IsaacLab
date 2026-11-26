# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-frame lidar hits drawn by the Newton GL and Viser visualizers."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_visualizers.lidar_points import (
    collect_lidar_hits,
    frame_environments,
    frame_visualizer_closeup,
    lidar_world_points,
    log_lidar_hits,
    sync_lidar_hits,
)

from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = [pytest.mark.kitless, pytest.mark.ov]


def _sensor(points: torch.Tensor, valid: torch.Tensor, frame: str, position=None, rotation=None, *, is_global=False):
    """Build the lidar attributes ``lidar_world_points`` reads."""
    return SimpleNamespace(
        data=SimpleNamespace(points=SimpleNamespace(torch=points), valid=SimpleNamespace(torch=valid)),
        cfg=SimpleNamespace(output_frame_of_reference=frame),
        _pos_w=None if position is None else SimpleNamespace(torch=position),
        _quat_w=None if rotation is None else SimpleNamespace(torch=rotation),
        _is_global=is_global,
    )


def test_lidar_world_points_keeps_finite_hits_for_the_requested_envs():
    """Misses drop out, a sensor-frame hit moves by its pose, and other environments stay out."""
    world = torch.tensor(
        [
            [[5.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            [[20.0, 1.0, 2.0], [float("inf"), float("inf"), float("inf")], [30.0, 0.0, 0.0]],
        ]
    )
    world_valid = torch.tensor([[True, False, True], [True, True, False]])
    selected = lidar_world_points(_sensor(world, world_valid, "WORLD"), [1])
    np.testing.assert_allclose(selected, [[20.0, 1.0, 2.0]])

    half = math.sqrt(0.5)
    # ISO 8855: +X forward, +Y left, +Z up. The prim looks along -Z, with +X right and +Y up.
    sensor_points = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [float("inf"), 0.0, 0.0]]])
    sensor_valid = torch.tensor([[True, True, True, True]])
    position = torch.tensor([[2.0, 3.0, 4.0]])
    rotation = torch.tensor([[0.0, 0.0, half, half]])
    transformed = lidar_world_points(
        _sensor(sensor_points, sensor_valid, "SENSOR", position, rotation),
        None,
    )
    # A yaw about Z leaves the vertical axis alone. Forward is prim -Z, left is prim -X, up is prim +Y.
    np.testing.assert_allclose(transformed, [[2.0, 3.0, 3.0], [2.0, 2.0, 4.0], [1.0, 3.0, 4.0]], atol=1e-6)


def _world_lidar(points: torch.Tensor, valid: torch.Tensor):
    """Lidar stand-in whose ``data`` buffers are already in the world frame."""
    from isaaclab_ov.renderers.ovrtx_lidar_sensor import OVRTXLidar

    class _Lidar(OVRTXLidar):
        def __init__(self):
            self._data = SimpleNamespace(points=SimpleNamespace(torch=points), valid=SimpleNamespace(torch=valid))
            self.cfg = SimpleNamespace(output_frame_of_reference="WORLD")
            self._pos_w = None
            self._quat_w = None
            self._is_global = False

        @property
        def data(self):
            return self._data

        def __del__(self):
            return

    return _Lidar()


def _lidar_sim():
    """Simulation stand-in with one hit lidar, one empty lidar, and a non-lidar sensor."""
    scene = SimpleNamespace(
        sensors={
            "range": _world_lidar(torch.tensor([[[1.0, 2.0, 3.0]]]), torch.tensor([[True]])),
            "empty": _world_lidar(
                torch.tensor([[[float("inf"), float("inf"), float("inf")]]]), torch.tensor([[False]])
            ),
            "camera": object(),
        }
    )
    return SimpleNamespace(get_scene_data_provider=lambda: SimpleNamespace(get_interactive_scene=lambda: scene))


def test_collect_lidar_hits_returns_lidar_clouds():
    """Only lidar sensors are returned, including an empty cloud so a viewer can hide the last batch."""
    hits = collect_lidar_hits(_lidar_sim(), [0])
    assert [name for name, _ in hits] == ["range", "empty"]
    np.testing.assert_allclose(hits[0][1], [[1.0, 2.0, 3.0]])
    assert hits[1][1].shape == (0, 3)


def test_collect_lidar_hits_draws_every_environment_without_a_subset():
    """No visualized subset means every environment's finite hits are drawn."""
    sensor = _world_lidar(
        torch.tensor([[[70.0, -70.0, 1.0]], [[10.0, -10.0, 2.0]]]),
        torch.tensor([[True], [True]]),
    )
    scene = SimpleNamespace(sensors={"lidar": sensor})
    sim = SimpleNamespace(get_scene_data_provider=lambda: SimpleNamespace(get_interactive_scene=lambda: scene))
    hits = collect_lidar_hits(sim, None)
    np.testing.assert_allclose(hits[0][1], [[70.0, -70.0, 1.0], [10.0, -10.0, 2.0]])


def test_frame_environments_fits_every_origin_and_keeps_a_single_cart_closeup():
    """Several environments pull the camera back. One environment at the look target stays close."""
    origins = np.array([[70.0, -70.0, 0.0], [10.0, -10.0, 0.0]], dtype=np.float32)
    eye, lookat = frame_environments((3.0, -3.0, 2.5), (0.0, 0.0, 1.0), origins)
    assert lookat == (40.0, -40.0, 1.0)
    distance = np.linalg.norm(np.subtract(eye, lookat))
    radius = np.linalg.norm(origins - np.array(lookat), axis=1).max()
    assert radius <= distance * math.tan(math.radians(30.0)) + 1e-3
    single = frame_environments((3.0, -3.0, 2.5), (0.0, 0.0, 1.0), [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(single[0], (3.0, -3.0, 2.5))
    np.testing.assert_allclose(single[1], (0.0, 0.0, 1.0))


def test_frame_visualizer_closeup_keeps_an_explicit_camera_angle():
    """The default view pulls back. A camera aimed elsewhere keeps that eye and look target."""
    from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg

    default = PerspectiveCameraCfg(eye=(3.0, -3.0, 2.5), lookat=(0.0, 0.0, 1.0))
    side = PerspectiveCameraCfg(eye=(8.0, 0.0, 2.0), lookat=(0.0, 1.0, 0.5))
    origins = torch.tensor([[70.0, -70.0, 0.0], [10.0, -10.0, 0.0]])
    scene = SimpleNamespace(env_origins=origins)
    cfg = SimpleNamespace(eye=default.eye, lookat=default.lookat)
    visualizer = SimpleNamespace(
        _env_ids=None,
        cfg=cfg,
        _image_views=[SimpleNamespace(cfg=SimpleNamespace(source=side))],
        _sim=SimpleNamespace(get_scene_data_provider=lambda: SimpleNamespace(get_interactive_scene=lambda: scene)),
    )
    frame_visualizer_closeup(visualizer, [default, side])
    assert cfg.lookat == (40.0, -40.0, 1.0)
    assert default.eye == cfg.eye and default.lookat == cfg.lookat
    assert side.eye == (8.0, 0.0, 2.0) and side.lookat == (0.0, 1.0, 0.5)


class _RecordingViewer:
    """Captures the batch ``log_lidar_hits`` hands to ``log_points``."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.batches: list[tuple] = []

    def log_points(self, name, points, radii=None, colors=None) -> None:
        self.batches.append((name, points, radii, colors))


class _ViewerGL(_RecordingViewer):
    """Name matches the Newton GL viewers, whose point kernel runs on ``viewer.device``."""


class _ViewerViser(_RecordingViewer):
    """Name matches the Viser viewer, which copies the batch on the CPU."""


def test_sync_lidar_hits_hides_previous_clouds_until_enabled():
    """A disabled overlay hides the last batch without reading the scene. Enabling uploads lidar hits."""
    viewer = _ViewerViser("cpu")
    assert sync_lidar_hits(None, [0], viewer, False, []) == []
    assert viewer.batches == []

    assert sync_lidar_hits(None, [0], viewer, False, ["range"]) == []
    name, cloud, _, _ = viewer.batches[0]
    assert name == "lidar/range"
    assert cloud is None

    names = sync_lidar_hits(_lidar_sim(), [0], viewer, True, [])
    assert names == ["range", "empty"]
    uploaded = [batch[0] for batch in viewer.batches[1:]]
    assert uploaded == ["lidar/range", "lidar/empty"]
    np.testing.assert_allclose(viewer.batches[1][1].numpy(), [[1.0, 2.0, 3.0]])
    assert viewer.batches[2][1] is None


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_log_lidar_hits_allocates_gl_points_on_the_viewer_device(device: str) -> None:
    """A GL viewer must receive positions and colors on its own device."""
    viewer = _ViewerGL(device)
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    log_lidar_hits(viewer, [("range_image", points)])
    _, cloud, _, colors = viewer.batches[0]
    assert cloud.device == wp.get_device(device)
    assert colors.device == wp.get_device(device)
    np.testing.assert_allclose(cloud.numpy(), points)
    np.testing.assert_allclose(colors.numpy(), [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_log_lidar_hits_keeps_viser_points_on_the_cpu() -> None:
    """A Viser viewer receives a CPU cloud and the shared color tuple."""
    viewer = _ViewerViser("cuda:0")
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    log_lidar_hits(viewer, [("range_image", points)])
    _, cloud, _, colors = viewer.batches[0]
    assert cloud.device.is_cpu
    assert colors == (1.0, 0.0, 0.0)
    np.testing.assert_allclose(cloud.numpy(), points)
