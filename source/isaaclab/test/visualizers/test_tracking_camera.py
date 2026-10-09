# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tracking cameras: their poses and when the launcher adds them to the scene."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.visualizers import TrackingCameraCfg, VisualizerCfg
from isaaclab.visualizers.tracking_camera import TrackingCameraUpdater, add_tracking_cameras


def _quat_z(yaw: float) -> list[float]:
    """Quaternion (x, y, z, w) of a rotation of *yaw* about +Z."""
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


class _Scene:
    """Holds one asset, as the scene does."""

    def __init__(self, asset, lazy: bool):
        self.cfg = SimpleNamespace(lazy_sensor_update=lazy)
        self._asset = asset

    def __getitem__(self, name):
        return self._asset


class _Camera:
    """Records the poses it is given and whether it was asked for fresh pixels."""

    is_initialized, device, refreshed = True, "cpu", False

    def update(self, dt, force_recompute=False):
        self.refreshed = force_recompute

    def set_world_poses_from_view(self, eyes, targets, env_ids=None):
        self.eyes, self.targets, self.env_ids = eyes, targets, env_ids


def _update(cfg: TrackingCameraCfg, yaws: list[float], dt: float = 0.1, lazy: bool = True) -> _Camera:
    """Run one update per yaw of a robot at (10, 0, 0.5) in env 1 of 2 and return the camera."""
    camera = _Camera()
    data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.5]])),
        root_quat_w=SimpleNamespace(),
    )
    asset = SimpleNamespace(is_initialized=True, num_instances=2, data=data)
    updater = TrackingCameraUpdater(cfg, camera, _Scene(asset, lazy))
    for yaw in yaws:
        data.root_quat_w.torch = torch.tensor([[0.0, 0.0, 0.0, 1.0], _quat_z(yaw)])
        updater.update([1], dt)
    return camera


def test_heading_rotates_the_offsets_and_smoothing_lags_the_turn():
    cfg = TrackingCameraCfg(eye=(-3.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.0), track_path="robot", follow_heading=True)
    # turned a quarter turn, the robot's back is at -y
    camera = _update(cfg, [math.pi / 2])
    torch.testing.assert_close(camera.eyes, torch.tensor([[10.0, -3.0, 1.5]]), atol=1e-5, rtol=0)
    torch.testing.assert_close(camera.targets, torch.tensor([[10.0, 0.0, 0.5]]), atol=1e-5, rtol=0)
    assert camera.env_ids.tolist() == [1]

    # the first heading applies at once; a later turn is filtered
    cfg.heading_smoothing_time_constant = 0.5
    camera = _update(cfg, [0.0, math.pi / 2])
    yaw = math.atan2(-camera.eyes[0, 1].item(), -(camera.eyes[0, 0].item() - 10.0))
    assert 0.0 < yaw < math.pi / 2


def _env_cfg(**sim_kwargs):
    default = VisualizerCfg(cameras=[TrackingCameraCfg(prim_path="{ENV_REGEX_NS}/Chase")])
    return ManagerBasedRLEnvCfg(
        sim=SimulationCfg(default_visualizer_cfg=default, **sim_kwargs), scene=InteractiveSceneCfg()
    )


def _add(env_cfg) -> bool:
    return add_tracking_cameras(env_cfg, env_cfg.sim, None)


def test_an_eager_scene_gets_fresh_pixels_after_the_camera_moves():
    cfg = TrackingCameraCfg(track_path="robot")
    assert _update(cfg, [0.0], lazy=False).refreshed
    assert not _update(cfg, [0.0], lazy=True).refreshed


def test_cameras_are_added_only_when_a_visualizer_uses_them():
    idle = _env_cfg()
    viewed = _env_cfg(visualizer_cfgs=[VisualizerCfg(visualizer_type="newton_gl", background_color=(0.1, 0.2, 0.3))])
    assert not _add(idle) and not hasattr(idle.scene, "Chase")
    assert _add(viewed)
    assert viewed.scene.Chase.prim_path == "{ENV_REGEX_NS}/Chase"
    assert viewed.scene.Chase.background_color == (0.1, 0.2, 0.3)


def test_newton_physics_renders_with_the_newton_warp_renderer():
    from isaaclab_newton.physics import NewtonCfg

    cfg = _env_cfg(visualizer_cfgs=VisualizerCfg(visualizer_type="newton_gl"))
    assert add_tracking_cameras(cfg, cfg.sim, NewtonCfg())
    assert cfg.scene.Chase.renderer_cfg.renderer_type == "newton_warp"


def test_a_visualizers_own_cameras_win_and_conflicts_are_rejected():
    own = VisualizerCfg(cameras=[TrackingCameraCfg(prim_path="{ENV_REGEX_NS}/Chase", resolution=(320, 240))])
    cfg = _env_cfg(visualizer_cfgs=[own])
    assert _add(cfg)
    assert (cfg.scene.Chase.width, cfg.scene.Chase.height) == (320, 240)

    cfg = _env_cfg(visualizer_cfgs=[own, VisualizerCfg()])
    with pytest.raises(ValueError, match="different tracking cameras"):
        _add(cfg)

    cfg = _env_cfg(visualizer_cfgs=[VisualizerCfg()])
    assert _add(cfg)
    assert not _add(cfg)  # a reused config keeps the camera it already has
    cfg.scene.Chase = object()
    with pytest.raises(ValueError, match="already has an entry"):
        _add(cfg)


def test_negative_smoothing_time_constant_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        TrackingCameraCfg(heading_smoothing_time_constant=-1.0)
