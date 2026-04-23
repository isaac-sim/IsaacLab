# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the renderer-owned output buffers contract and CameraCfg deprecation forwarding."""

import warnings

import pytest
import torch

pytest.importorskip("isaaclab_physx")

from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.sensors.camera import CameraCfg, TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


# -----------------------------------------------------------------------------
# CameraCfg deprecation forwarding
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name,deprecated_value",
    [
        ("colorize_semantic_segmentation", False),
        ("colorize_instance_segmentation", False),
        ("colorize_instance_id_segmentation", False),
        ("semantic_filter", ["class"]),
        ("semantic_segmentation_mapping", {"class:cube": (1, 2, 3, 4)}),
        ("depth_clipping_behavior", "max"),
    ],
)
def test_camera_cfg_forwards_deprecated_fields_to_renderer_cfg(field_name, deprecated_value):
    """Deprecated RTX-flavored field set on CameraCfg lands on renderer_cfg and warns."""
    kwargs = {
        "height": 64,
        "width": 64,
        "prim_path": "/World/Camera",
        "spawn": _SPAWN,
        "data_types": ["rgb"],
        field_name: deprecated_value,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = CameraCfg(**kwargs)

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any(f"CameraCfg.{field_name}" in str(w.message) for w in deprecation_warnings)
    assert getattr(cfg.renderer_cfg, field_name) == deprecated_value


def test_camera_cfg_default_does_not_warn_or_forward():
    """Default-valued deprecated fields stay silent."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = CameraCfg(
            height=64,
            width=64,
            prim_path="/World/Camera",
            spawn=_SPAWN,
            data_types=["rgb"],
        )

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning) and "CameraCfg." in str(w.message)
    ]
    assert deprecation_warnings == []
    assert cfg.renderer_cfg.colorize_semantic_segmentation is True


def test_camera_cfg_post_construction_mutation_is_silent_no_op():
    """Mutating a deprecated field after construction does not propagate to renderer_cfg."""
    cfg = CameraCfg(
        height=64,
        width=64,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=["rgb"],
    )
    assert cfg.renderer_cfg.colorize_semantic_segmentation is True
    cfg.colorize_semantic_segmentation = False
    assert cfg.renderer_cfg.colorize_semantic_segmentation is True


def test_tiled_camera_cfg_does_not_forward_deprecated_fields():
    """TiledCameraCfg skips CameraCfg's per-field forwarder by design."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = TiledCameraCfg(
            height=64,
            width=64,
            prim_path="/World/Camera",
            spawn=_SPAWN,
            data_types=["rgb"],
            colorize_semantic_segmentation=False,
        )

    tiled_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning) and "TiledCameraCfg" in str(w.message)
    ]
    assert tiled_warnings

    field_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning) and "CameraCfg.colorize_" in str(w.message)
    ]
    assert field_warnings == []

    assert cfg.renderer_cfg.colorize_semantic_segmentation is True


# -----------------------------------------------------------------------------
# IsaacRtxRenderer.create_output_buffers contract
# -----------------------------------------------------------------------------


def test_isaac_rtx_create_output_buffers_omits_unsupported_and_aliases_rgb():
    """RTX backend drops names it cannot produce; rgb aliases rgba storage."""
    pytest.importorskip("pxr")
    pytest.importorskip("isaacsim.core")
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    renderer.cfg = IsaacRtxRendererCfg()

    requested = ["rgb", "rgba", "depth", "definitely_not_supported"]
    buffers = renderer.create_output_buffers(data_types=requested, height=8, width=16, num_views=2, device="cpu")

    assert "definitely_not_supported" not in buffers
    assert {"rgb", "rgba", "depth"} <= set(buffers.keys())
    assert buffers["rgba"].shape == (2, 8, 16, 4)
    assert buffers["rgba"].dtype == torch.uint8
    assert buffers["depth"].shape == (2, 8, 16, 1)
    assert buffers["depth"].dtype == torch.float32
    assert buffers["rgb"].data_ptr() == buffers["rgba"].data_ptr()


# -----------------------------------------------------------------------------
# OVRTX zero-copy consolidation
# -----------------------------------------------------------------------------


def _make_ovrtx_render_data():
    """Construct a minimal OVRTXRenderData without invoking its sensor-based __init__."""
    pytest.importorskip("isaaclab_ov")
    pytest.importorskip("ovrtx")
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderData

    rd = OVRTXRenderData.__new__(OVRTXRenderData)
    rd.warp_buffers = {}
    return rd


def test_ovrtx_set_outputs_wraps_caller_torch_zero_copy():
    """OVRTXRenderer.set_outputs publishes warp views over the caller's torch storage."""
    pytest.importorskip("isaaclab_ov")
    pytest.importorskip("ovrtx")
    import warp as wp
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer

    renderer = OVRTXRenderer(OVRTXRendererCfg())

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    buffers = renderer.create_output_buffers(
        data_types=["rgb", "rgba", "depth"], height=8, width=16, num_views=2, device=device
    )
    render_data = _make_ovrtx_render_data()
    renderer.set_outputs(render_data, buffers)

    assert set(render_data.warp_buffers.keys()) >= {"rgba", "depth"}
    assert render_data.warp_buffers["rgba"].ptr == wp.from_torch(buffers["rgba"]).ptr
    assert render_data.warp_buffers["depth"].ptr == wp.from_torch(buffers["depth"]).ptr
    assert "rgb" not in render_data.warp_buffers


def test_ovrtx_read_output_is_a_no_op_after_consolidation():
    """OVRTXRenderer.read_output is a no-op once set_outputs wires up zero-copy."""
    pytest.importorskip("isaaclab_ov")
    pytest.importorskip("ovrtx")
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer

    from isaaclab.sensors.camera.camera_data import CameraData

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
