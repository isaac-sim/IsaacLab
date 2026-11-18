# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The foveated camera class selects its implementation from the renderer."""

from dataclasses import dataclass

import pytest
from isaaclab_fovi.sensors import FoveatedCamera, FoveatedCameraCfg
from isaaclab_fovi.sensors.foveated_newton_camera import FoveatedNewtonCamera
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera.camera import Camera
from isaaclab.sim import PinholeCameraCfg

pytestmark = pytest.mark.kitless


@dataclass
class RendererSelection:
    renderer_type: str


@dataclass
class CameraSelection:
    renderer_cfg: RendererSelection


def dispatch(cfg) -> type:
    """The implementation class the camera selects for ``cfg``, without initialising it."""
    return type(FoveatedCamera.__new__(FoveatedCamera, cfg))


def test_default_renderer_resolves_to_isaac_rtx() -> None:
    cfg = FoveatedCameraCfg(prim_path="/World/Camera")
    assert type(cfg.renderer_cfg) is IsaacRtxRendererCfg
    assert cfg.renderer_cfg.renderer_type == "isaac_rtx"
    assert dispatch(cfg) is FoveatedRTXCamera


@pytest.mark.parametrize(
    ("renderer_type", "expected"),
    [("ovrtx", FoveatedRTXCamera), ("newton_warp", FoveatedNewtonCamera), ("newton_optix", FoveatedNewtonCamera)],
)
def test_renderer_selects_its_implementation(renderer_type: str, expected: type) -> None:
    assert dispatch(CameraSelection(RendererSelection(renderer_type))) is expected


def test_unknown_renderer_fails_explicitly() -> None:
    cfg = CameraSelection(RendererSelection("unknown"))
    with pytest.raises(ValueError, match="Unsupported foveated renderer"):
        FoveatedCamera(cfg)


@pytest.mark.parametrize(
    ("renderer_cfg", "turns_pose_by_gaze"),
    [(NewtonWarpRendererCfg(), True), (OVRTXRendererCfg(), True), (IsaacRtxRendererCfg(), False)],
    ids=["newton_warp", "ovrtx", "isaac_rtx"],
)
def test_cameras_that_turn_their_pose_by_the_gaze_need_per_frame_pose_updates(
    monkeypatch, renderer_cfg, turns_pose_by_gaze: bool
) -> None:
    """Newton and OVRTX turn the rendered pose by the gaze in each pose update, so without pose updates the renderer
    would never see the gaze; Isaac RTX writes the gaze into its camera prims instead."""
    monkeypatch.setattr(Camera, "__init__", lambda self, cfg: setattr(self, "stage", None))
    monkeypatch.setattr(Camera, "__del__", lambda self: None)
    monkeypatch.setattr(FoveatedRTXCamera, "_prepare_projection", lambda self: None)
    monkeypatch.setattr(sim_utils, "find_matching_prims", lambda *args: [])
    cfg = FoveatedCameraCfg(
        prim_path="/World/Camera", spawn=PinholeCameraCfg(), renderer_cfg=renderer_cfg, update_latest_camera_pose=False
    )
    if turns_pose_by_gaze:
        with pytest.raises(ValueError, match="update_latest_camera_pose=True"):
            FoveatedCamera(cfg)
    else:
        assert not FoveatedCamera(cfg).turns_pose_by_gaze
