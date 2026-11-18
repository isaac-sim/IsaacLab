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
from isaaclab_physx.renderers import IsaacRtxRendererCfg

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
