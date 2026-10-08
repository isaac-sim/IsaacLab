# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tracked cameras: their poses and when the launcher adds them to the scene."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.visualizers import TrackedCameraCfg, VisualizerCfg
from isaaclab.visualizers.tracked_camera import TrackedCameraUpdater, add_tracked_cameras


def _quat_z(yaw: float) -> list[float]:
    """Quaternion (x, y, z, w) of a rotation of *yaw* about +Z."""
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


class _Camera:
    """Records the poses it is given."""

    is_initialized, device = True, "cpu"

    def set_world_poses_from_view(self, eyes, targets, env_ids=None):
        self.eyes, self.targets, self.env_ids = eyes, targets, env_ids


def _update(cfg: TrackedCameraCfg, yaws: list[float], dt: float = 0.1) -> _Camera:
    """Run one update per yaw of a robot at (10, 0, 0.5) in env 1 of 2 and return the camera."""
    camera = _Camera()
    data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.5]])),
        root_quat_w=SimpleNamespace(),
    )
    asset = SimpleNamespace(is_initialized=True, num_instances=2, data=data)
    updater = TrackedCameraUpdater(cfg, camera, {"robot": asset})
    for yaw in yaws:
        data.root_quat_w.torch = torch.tensor([[0.0, 0.0, 0.0, 1.0], _quat_z(yaw)])
        updater.update([1], dt)
    return camera


def test_heading_rotates_the_offsets_and_smoothing_lags_the_turn():
    cfg = TrackedCameraCfg(eye=(-3.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.0), track_path="robot", follow_heading=True)
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


def test_cameras_are_added_only_when_a_visualizer_uses_them():
    def env_cfg():
        visualizer = VisualizerCfg(cameras=[TrackedCameraCfg(prim_path="{ENV_REGEX_NS}/Chase")])
        return ManagerBasedRLEnvCfg(sim=SimulationCfg(default_visualizer_cfg=visualizer), scene=InteractiveSceneCfg())

    idle, viewed = env_cfg(), env_cfg()
    add_tracked_cameras(idle, {"visualizer": []})
    add_tracked_cameras(viewed, {"visualizer": ["newton_gl"], "physics": "newton_mjwarp"})
    assert not hasattr(idle.scene, "Chase")
    assert viewed.scene.Chase.prim_path == "{ENV_REGEX_NS}/Chase"


def test_negative_smoothing_time_constant_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        TrackedCameraCfg(heading_smoothing_time_constant=-1.0)
