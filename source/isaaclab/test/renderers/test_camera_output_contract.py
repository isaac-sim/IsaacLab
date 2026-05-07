# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the renderer→camera output contract."""

import warnings

import pytest
import torch

pytest.importorskip("isaaclab_physx")

from isaaclab.sensors.camera import CameraCfg, TiledCameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


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
    """Deprecated CameraCfg field is forwarded to renderer_cfg with a warning."""
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
    """TiledCameraCfg skips CameraCfg's per-field forwarder."""
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


def test_newton_warp_supported_output_types_key_set():
    """NewtonWarpRenderer publishes the documented key set."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("newton")
    from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    renderer = NewtonWarpRenderer.__new__(NewtonWarpRenderer)
    renderer.cfg = NewtonWarpRendererCfg()
    specs = renderer.supported_output_types()

    assert set(specs.keys()) == {
        RenderBufferKind.RGB,
        RenderBufferKind.RGBA,
        RenderBufferKind.ALBEDO,
        RenderBufferKind.DEPTH,
        RenderBufferKind.NORMALS,
        RenderBufferKind.INSTANCE_SEGMENTATION_FAST,
    }


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def test_camera_data_allocates_supported_subset_and_aliases_rgb():
    """CameraData allocates the intersection of requested + supported and aliases rgb into rgba."""
    cfg = _make_camera_cfg(["rgb", "rgba", "depth"])
    specs = {
        RenderBufferKind.RGBA: RenderBufferSpec(4, torch.uint8),
        RenderBufferKind.RGB: RenderBufferSpec(3, torch.uint8),
        RenderBufferKind.DEPTH: RenderBufferSpec(1, torch.float32),
        RenderBufferKind.NORMALS: RenderBufferSpec(3, torch.float32),
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
    """Requested types absent from supported_specs are absent from data.output."""
    cfg = _make_camera_cfg(["rgb", "normals"])
    specs = {
        RenderBufferKind.RGBA: RenderBufferSpec(4, torch.uint8),
        RenderBufferKind.RGB: RenderBufferSpec(3, torch.uint8),
    }
    data = CameraData.allocate(
        data_types=cfg.data_types, height=4, width=4, num_views=1, device="cpu", supported_specs=specs
    )

    assert "normals" not in data.output
    assert {"rgb", "rgba"} <= set(data.output.keys())


def test_camera_data_no_arg_construction_yields_empty_container():
    """Bare CameraData() produces an all-None container."""
    data = CameraData()
    assert data.pos_w is None
    assert data.quat_w_world is None
    assert data.intrinsic_matrices is None
    assert data.output is None
    assert data.info is None
    assert data.image_shape is None


def test_camera_data_segmentation_dtype_follows_supported_spec():
    """CameraData consumes the layout dtype declared by the renderer spec."""
    cfg = _make_camera_cfg(["instance_segmentation_fast"])
    raw_specs = {RenderBufferKind.INSTANCE_SEGMENTATION_FAST: RenderBufferSpec(1, torch.int32)}
    colorized_specs = {RenderBufferKind.INSTANCE_SEGMENTATION_FAST: RenderBufferSpec(4, torch.uint8)}

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


def test_camera_data_allocate_raises_on_unknown_name():
    """An unknown data_types name raises ValueError naming the offender."""
    supported_specs = {RenderBufferKind.RGBA: RenderBufferSpec(4, torch.uint8)}
    with pytest.raises(ValueError) as exc_info:
        CameraData.allocate(
            data_types=["not_a_real_type"],
            height=4,
            width=4,
            num_views=1,
            device="cpu",
            supported_specs=supported_specs,
        )
    assert "not_a_real_type" in str(exc_info.value)
    assert "RenderBufferKind" in str(exc_info.value)
