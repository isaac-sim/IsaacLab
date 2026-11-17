# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The foveated camera class selects its implementation from the renderer."""

from dataclasses import dataclass

import pytest
from isaaclab_fovi.sensors import FoveatedCamera, FoveatedCameraCfg, FoveatedNewtonCamera, FoveatedRTXCamera


@dataclass
class RendererSelection:
    renderer_type: str


@dataclass
class CameraSelection:
    renderer_cfg: RendererSelection


def test_default_renderer_is_resolved_to_a_concrete_one() -> None:
    cfg = FoveatedCameraCfg(prim_path="/World/Camera")
    assert cfg.renderer_cfg.renderer_type != "default"


def test_unknown_renderer_fails_explicitly() -> None:
    cfg = CameraSelection(RendererSelection("unknown"))
    with pytest.raises(ValueError, match="Unsupported foveated renderer"):
        FoveatedCamera(cfg)


@pytest.mark.parametrize(
    ("renderer_type", "implementation"),
    [("newton_warp", FoveatedNewtonCamera), ("isaac_rtx", FoveatedRTXCamera), ("ovrtx", FoveatedRTXCamera)],
)
def test_renderer_selects_implementation(renderer_type: str, implementation: type) -> None:
    camera = FoveatedCamera.__new__(FoveatedCamera, CameraSelection(RendererSelection(renderer_type)))
    assert type(camera) is implementation
    assert isinstance(camera, FoveatedCamera)
