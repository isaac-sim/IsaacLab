# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer output contract."""

import pytest
import torch

pytest.importorskip("isaaclab_ov")
pytest.importorskip("ovrtx")

from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderData, OVRTXRenderer

from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg

pytestmark = pytest.mark.isaacsim_ci

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def _make_ovrtx_render_data() -> OVRTXRenderData:
    rd = OVRTXRenderData.__new__(OVRTXRenderData)
    rd.warp_buffers = {}
    return rd


def test_ovrtx_supported_output_types_key_set():
    """OVRTX publishes the documented key set and per-output spec."""
    renderer = OVRTXRenderer(OVRTXRendererCfg())
    specs = renderer.supported_output_types()

    assert set(specs.keys()) == {
        RenderBufferKind.RGB,
        RenderBufferKind.RGBA,
        RenderBufferKind.ALBEDO,
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.DEPTH,
        RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
        RenderBufferKind.DISTANCE_TO_CAMERA,
    }
    assert specs[RenderBufferKind.RGBA] == RenderBufferSpec(4, torch.uint8)
    assert specs[RenderBufferKind.DEPTH] == RenderBufferSpec(1, torch.float32)


def test_ovrtx_set_outputs_wraps_caller_torch_zero_copy():
    """OVRTXRenderer.set_outputs publishes warp views over the caller's torch storage."""
    import warp as wp

    renderer = OVRTXRenderer(OVRTXRendererCfg())

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb", "rgba", "depth"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    renderer.set_outputs(render_data, data.output)

    assert set(render_data.warp_buffers.keys()) >= {"rgba", "depth"}
    assert render_data.warp_buffers["rgba"].ptr == wp.from_torch(data.output["rgba"]).ptr
    assert render_data.warp_buffers["depth"].ptr == wp.from_torch(data.output["depth"]).ptr
    assert "rgb" not in render_data.warp_buffers


def test_ovrtx_read_output_is_a_no_op_after_consolidation():
    """OVRTXRenderer.read_output is a no-op once set_outputs wires up zero-copy."""
    renderer = OVRTXRenderer(OVRTXRendererCfg())
    render_data = _make_ovrtx_render_data()
    camera_data = CameraData()
    camera_data.info = {}
    camera_data.output = {}

    result = renderer.read_output(render_data, camera_data)
    assert result is None
    assert render_data.warp_buffers == {}
    assert camera_data.info == {}
    assert camera_data.output == {}
