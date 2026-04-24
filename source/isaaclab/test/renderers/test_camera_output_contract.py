# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the renderer→camera output contract.

The renderer publishes the per-output layout via
:meth:`isaaclab.renderers.BaseRenderer.supported_output_types`; ``CameraData``
allocates storage for the supported subset of the requested types and aliases
``rgb`` into ``rgba``. These tests cover both halves of that contract plus
:class:`CameraCfg` deprecation forwarding (which feeds the renderer cfg the
contract is published from).
"""

import warnings
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("isaaclab_physx")

from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.sensors.camera import CameraCfg, TiledCameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, CameraDataType, OutputSpec
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
# Renderer.supported_output_types contract
# -----------------------------------------------------------------------------


def _make_isaac_rtx_renderer(cfg: IsaacRtxRendererCfg | None = None):
    """Construct an ``IsaacRtxRenderer`` instance without invoking its sim-coupled ``__init__``."""
    pytest.importorskip("pxr")
    pytest.importorskip("isaacsim.core")
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    renderer.cfg = cfg if cfg is not None else IsaacRtxRendererCfg()
    return renderer


def _fake_sim_version(major: int):
    """Return a stand-in for ``get_isaac_sim_version()`` exposing only ``.major``."""

    class _V:
        pass

    v = _V()
    v.major = major
    return v


def test_isaac_rtx_supported_output_types_sim6_includes_albedo_and_simple_shading():
    renderer = _make_isaac_rtx_renderer()

    with patch(
        "isaaclab_physx.renderers.isaac_rtx_renderer.get_isaac_sim_version",
        return_value=_fake_sim_version(6),
    ):
        specs = renderer.supported_output_types()

    assert CameraDataType.ALBEDO in specs
    for shading in (
        CameraDataType.SIMPLE_SHADING_CONSTANT_DIFFUSE,
        CameraDataType.SIMPLE_SHADING_DIFFUSE_MDL,
        CameraDataType.SIMPLE_SHADING_FULL_MDL,
    ):
        assert shading in specs
        assert specs[shading] == OutputSpec(3, torch.uint8)
    assert specs[CameraDataType.ALBEDO] == OutputSpec(4, torch.uint8)


def test_isaac_rtx_supported_output_types_pre_sim6_omits_albedo_and_simple_shading():
    renderer = _make_isaac_rtx_renderer()

    with patch(
        "isaaclab_physx.renderers.isaac_rtx_renderer.get_isaac_sim_version",
        return_value=_fake_sim_version(5),
    ):
        specs = renderer.supported_output_types()

    assert CameraDataType.ALBEDO not in specs
    for shading in (
        CameraDataType.SIMPLE_SHADING_CONSTANT_DIFFUSE,
        CameraDataType.SIMPLE_SHADING_DIFFUSE_MDL,
        CameraDataType.SIMPLE_SHADING_FULL_MDL,
    ):
        assert shading not in specs
    # Color/depth/etc. still ship pre-sim 6.
    assert specs[CameraDataType.RGBA] == OutputSpec(4, torch.uint8)
    assert specs[CameraDataType.DEPTH] == OutputSpec(1, torch.float32)


@pytest.mark.parametrize(
    "data_type,flag_attr",
    [
        (CameraDataType.SEMANTIC_SEGMENTATION, "colorize_semantic_segmentation"),
        (CameraDataType.INSTANCE_SEGMENTATION_FAST, "colorize_instance_segmentation"),
        (CameraDataType.INSTANCE_ID_SEGMENTATION_FAST, "colorize_instance_id_segmentation"),
    ],
)
def test_isaac_rtx_segmentation_specs_follow_colorize_flags(data_type, flag_attr):
    """Each segmentation entry's spec follows the corresponding ``colorize_*`` flag."""
    cfg_colorized = IsaacRtxRendererCfg()
    setattr(cfg_colorized, flag_attr, True)
    cfg_raw = IsaacRtxRendererCfg()
    setattr(cfg_raw, flag_attr, False)

    with patch(
        "isaaclab_physx.renderers.isaac_rtx_renderer.get_isaac_sim_version",
        return_value=_fake_sim_version(6),
    ):
        specs_colorized = _make_isaac_rtx_renderer(cfg_colorized).supported_output_types()
        specs_raw = _make_isaac_rtx_renderer(cfg_raw).supported_output_types()

    assert specs_colorized[data_type] == OutputSpec(4, torch.uint8)
    assert specs_raw[data_type] == OutputSpec(1, torch.int32)


def test_newton_warp_supported_output_types_key_set():
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    renderer.cfg = NewtonWarpRendererCfg()
    specs = renderer.supported_output_types()

    assert set(specs.keys()) == {
        CameraDataType.RGB,
        CameraDataType.RGBA,
        CameraDataType.ALBEDO,
        CameraDataType.DEPTH,
        CameraDataType.NORMALS,
        CameraDataType.INSTANCE_SEGMENTATION_FAST,
    }


def test_ovrtx_supported_output_types_key_set():
    pytest.importorskip("isaaclab_ov")
    pytest.importorskip("ovrtx")
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer

    renderer = OVRTXRenderer(OVRTXRendererCfg())
    specs = renderer.supported_output_types()

    assert set(specs.keys()) == {
        CameraDataType.RGB,
        CameraDataType.RGBA,
        CameraDataType.ALBEDO,
        CameraDataType.SEMANTIC_SEGMENTATION,
        CameraDataType.DEPTH,
        CameraDataType.DISTANCE_TO_IMAGE_PLANE,
        CameraDataType.DISTANCE_TO_CAMERA,
    }


# -----------------------------------------------------------------------------
# CameraData allocation
# -----------------------------------------------------------------------------


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def test_camera_data_allocates_supported_subset_and_aliases_rgb():
    """``CameraData`` allocates the intersection of requested + supported and aliases rgb into rgba."""
    cfg = _make_camera_cfg(["rgb", "rgba", "depth"])
    specs = {
        CameraDataType.RGBA: OutputSpec(4, torch.uint8),
        CameraDataType.RGB: OutputSpec(3, torch.uint8),
        CameraDataType.DEPTH: OutputSpec(1, torch.float32),
        CameraDataType.NORMALS: OutputSpec(3, torch.float32),
    }
    data = CameraData.allocate(
        data_types=cfg.data_types, height=8, width=16, num_views=2, device="cpu", supported_specs=specs
    )

    assert set(data.output.keys()) == {"rgba", "rgb", "depth"}
    assert data.output["rgba"].shape == (2, 8, 16, 4)
    assert data.output["rgba"].dtype == torch.uint8
    assert data.output["depth"].shape == (2, 8, 16, 1)
    assert data.output["depth"].dtype == torch.float32
    assert data.output["rgb"].data_ptr() == data.output["rgba"].data_ptr()
    assert data.image_shape == (8, 16)
    assert data.info == {"rgba": None, "rgb": None, "depth": None}


def test_camera_data_drops_requested_types_not_in_supported_specs():
    """Requested types absent from ``supported_specs`` are absent from ``data.output``."""
    cfg = _make_camera_cfg(["rgb", "normals"])
    specs = {CameraDataType.RGBA: OutputSpec(4, torch.uint8), CameraDataType.RGB: OutputSpec(3, torch.uint8)}
    data = CameraData.allocate(
        data_types=cfg.data_types, height=4, width=4, num_views=1, device="cpu", supported_specs=specs
    )

    assert "normals" not in data.output
    assert {"rgb", "rgba"} <= set(data.output.keys())


def test_camera_data_no_arg_construction_yields_empty_container():
    """Bare ``CameraData()`` continues to produce an all-``None`` container (back-compat)."""
    data = CameraData()
    assert data.pos_w is None
    assert data.quat_w_world is None
    assert data.intrinsic_matrices is None
    assert data.output is None
    assert data.info is None
    assert data.image_shape is None


def test_camera_data_segmentation_dtype_follows_supported_spec():
    """``CameraData`` consumes the layout fact (dtype) without knowing about ``colorize_*`` flags."""
    cfg = _make_camera_cfg(["instance_segmentation_fast"])
    raw_specs = {CameraDataType.INSTANCE_SEGMENTATION_FAST: OutputSpec(1, torch.int32)}
    colorized_specs = {CameraDataType.INSTANCE_SEGMENTATION_FAST: OutputSpec(4, torch.uint8)}

    raw = CameraData.allocate(
        data_types=cfg.data_types, height=4, width=4, num_views=1, device="cpu", supported_specs=raw_specs
    )
    colorized = CameraData.allocate(
        data_types=cfg.data_types, height=4, width=4, num_views=1, device="cpu", supported_specs=colorized_specs
    )

    assert raw.output["instance_segmentation_fast"].dtype == torch.int32
    assert raw.output["instance_segmentation_fast"].shape == (1, 4, 4, 1)
    assert colorized.output["instance_segmentation_fast"].dtype == torch.uint8
    assert colorized.output["instance_segmentation_fast"].shape == (1, 4, 4, 4)


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
    pytest.importorskip("isaaclab_ov")
    pytest.importorskip("ovrtx")
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer

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
