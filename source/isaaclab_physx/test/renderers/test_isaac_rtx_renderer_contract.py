# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Isaac RTX renderer output contract."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import warp as wp
from packaging import version

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec

pytestmark = pytest.mark.isaacsim_ci


def _install_omni_stubs(monkeypatch):
    omni_module = sys.modules.get("omni", types.ModuleType("omni"))
    replicator_module = types.ModuleType("omni.replicator")
    replicator_core_module = types.ModuleType("omni.replicator.core")
    syntheticdata_module = types.ModuleType("omni.syntheticdata")
    usd_module = MagicMock()

    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator_module)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", replicator_core_module)
    monkeypatch.setitem(sys.modules, "omni.syntheticdata", syntheticdata_module)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_module)
    monkeypatch.setattr(omni_module, "replicator", replicator_module, raising=False)
    monkeypatch.setattr(omni_module, "syntheticdata", syntheticdata_module, raising=False)
    monkeypatch.setattr(omni_module, "usd", usd_module, raising=False)
    monkeypatch.setattr(replicator_module, "core", replicator_core_module, raising=False)

    return replicator_core_module, syntheticdata_module


def test_isaac_rtx_supported_output_types_include_rgb_hdr(monkeypatch):
    """Isaac RTX advertises RGB_HDR as a 3-channel float renderer output."""
    _install_omni_stubs(monkeypatch)
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)
    renderer.cfg = IsaacRtxRendererCfg()
    with patch("isaaclab_physx.renderers.isaac_rtx_renderer.get_isaac_sim_version", return_value=version.parse("6.0")):
        specs = renderer.supported_output_types()

    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32)


def test_create_render_data_uses_unique_sdf_safe_render_product_name(monkeypatch):
    """Each tiled render product gets a fresh ``rp_<uuid4.hex>`` name.

    Unique names avoid collisions across concurrent tiled cameras and sequential
    create/destroy cycles in one Kit process (e.g. ``simple_shading_*`` pytest).
    uuid4 provides 122 random bits, so birthday-paradox collision chance among n
    names is ~n^2 / 2^123 — negligible for Isaac Lab workloads.
    """
    replicator_core_module, syntheticdata_module = _install_omni_stubs(monkeypatch)
    monkeypatch.setattr(syntheticdata_module, "SyntheticData", MagicMock(), raising=False)

    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    from pxr import Sdf, UsdGeom

    import isaaclab.sim.utils.stage as stage_utils

    # Stub Kit settings / stage so create_render_data can run without Isaac Sim.
    # has_gui=False keeps the depth-only color-render branch inactive for rgb cameras.
    settings = MagicMock()
    settings.get.return_value = False
    stage = MagicMock()
    # Pass the Camera prim check that gates render-product creation.
    stage.GetPrimAtPath.return_value.IsA.side_effect = lambda typ: typ is UsdGeom.Camera

    # Capture the ``name=`` kwarg passed to Replicator; the returned HydraTexture
    # and annotator registry only need to exist so create_render_data can finish.
    rp = MagicMock()
    rp.path = "/Render/rp_test"
    create_tiled = MagicMock(return_value=rp)
    annotator = MagicMock()
    registry = MagicMock()
    registry.get_annotator.return_value = annotator
    replicator_core_module.create = SimpleNamespace(render_product_tiled=create_tiled)
    replicator_core_module.AnnotatorRegistry = registry

    # Minimal CameraRenderSpec: one rgb tiled camera is enough to exercise naming.
    spec = SimpleNamespace(
        camera_prim_paths=["/World/envs/env_0/Camera"],
        device="cpu",
        cfg=SimpleNamespace(
            data_types=["rgb"],
            width=64,
            height=64,
            isp_cfg=None,
            colorize_semantic_segmentation=False,
            colorize_instance_segmentation=False,
            colorize_instance_id_segmentation=False,
        ),
    )
    renderer = rtx_renderer.IsaacRtxRenderer.__new__(rtx_renderer.IsaacRtxRenderer)
    renderer.cfg = IsaacRtxRendererCfg()

    # Create many products with the same spec: names must still all differ (the
    # sequential simple_shading_* / multi-camera collision case this fix targets).
    num_names = 256
    names: list[str] = []
    with (
        patch.object(rtx_renderer, "get_settings_manager", return_value=settings),
        patch.object(rtx_renderer, "get_isaac_sim_version", return_value=version.parse("6.0")),
        patch.object(stage_utils, "get_current_stage", return_value=stage),
    ):
        for _ in range(num_names):
            renderer.create_render_data(spec)
            names.append(create_tiled.call_args.kwargs["name"])

    # Every call must mint a distinct name — a reused default was the original bug.
    assert len(set(names)) == num_names
    for name in names:
        # Contract: ``rp_`` + uuid4().hex so the token is a valid USD identifier
        # (no hyphens) and cannot collide with path-derived names.
        assert name.startswith("rp_")
        hex_part = name.removeprefix("rp_")
        # uuid4().hex is 32 lowercase hex digits (128 bits; 122 of them random).
        assert len(hex_part) == 32
        assert all(c in "0123456789abcdef" for c in hex_part)
        # Replicator builds a USD prim from this name; reject illegal identifiers.
        assert Sdf.Path.IsValidIdentifier(name)
        assert Sdf.Path.IsValidPathString(f"/Render/{name}")


def test_render_product_uuid_name_format_is_sdf_safe():
    """``rp_{uuid4().hex}`` matches the create_render_data naming contract and is SDF-safe."""
    import uuid

    from pxr import Sdf

    names = [f"rp_{uuid.uuid4().hex}" for _ in range(64)]
    assert len(set(names)) == len(names)
    for name in names:
        assert name.startswith("rp_")
        hex_part = name.removeprefix("rp_")
        assert len(hex_part) == 32
        int(hex_part, 16)  # raises if not hex
        assert "-" not in name
        assert Sdf.Path.IsValidIdentifier(name)
        assert Sdf.Path.IsValidPathString(f"/Render/{name}")


@pytest.mark.parametrize(
    ("has_gui", "expected_disable_color_render"),
    [
        pytest.param(True, False, id="gui-keeps-color-rendering"),
        pytest.param(False, True, id="headless-uses-depth-only-rendering"),
    ],
)
def test_depth_only_camera_color_render_setting(monkeypatch, has_gui, expected_disable_color_render):
    """Depth-only cameras must not disable color rendering for an active GUI.

    ``disableColorRender`` is a global RTX setting, so enabling it for a depth-only
    camera also blacks out the viewport. Headless execution should retain the
    depth-only optimization.
    """
    _, syntheticdata_module = _install_omni_stubs(monkeypatch)
    monkeypatch.setattr(syntheticdata_module, "SyntheticData", MagicMock(), raising=False)

    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    import isaaclab.sim.utils.stage as stage_utils

    settings = MagicMock()
    settings.get.return_value = has_gui

    # Camera validation terminates create_render_data immediately after the
    # color-render setting is selected, keeping this a lightweight unit test.
    stage = MagicMock()
    stage.GetPrimAtPath.return_value.IsA.return_value = False
    spec = SimpleNamespace(
        camera_prim_paths=["/World/NotACamera"],
        cfg=SimpleNamespace(data_types=["depth"]),
    )
    renderer = rtx_renderer.IsaacRtxRenderer.__new__(rtx_renderer.IsaacRtxRenderer)
    renderer.cfg = IsaacRtxRendererCfg()

    with (
        patch.object(rtx_renderer, "get_settings_manager", return_value=settings),
        patch.object(rtx_renderer, "get_isaac_sim_version", return_value=version.parse("6.0")),
        patch.object(stage_utils, "get_current_stage", return_value=stage),
        pytest.raises(RuntimeError, match="is not a Camera"),
    ):
        renderer.create_render_data(spec)

    color_render_calls = [
        setting_call
        for setting_call in settings.set_bool.call_args_list
        if setting_call.args[0] == "/rtx/sdg/force/disableColorRender"
    ]
    assert color_render_calls[-1] == call("/rtx/sdg/force/disableColorRender", expected_disable_color_render)


_MISSING = object()


@pytest.mark.parametrize(
    ("stored", "expected_called"),
    [
        pytest.param(True, True, id="deterministic-true-applies-settings"),
        pytest.param(False, False, id="deterministic-false-skips-settings"),
        pytest.param(_MISSING, False, id="deterministic-missing-skips-settings"),
    ],
)
def test_deterministic_flag_gates_rtx_determinism_settings(monkeypatch, stored, expected_called):
    """IsaacRtxRenderer applies RTX determinism settings only when ``/isaaclab/render/deterministic`` is true."""
    _install_omni_stubs(monkeypatch)
    import isaaclab_physx.renderers.isaac_rtx_renderer as rtx_renderer
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg

    # RTX rendering requires cameras to be enabled.
    settings_values = {"/isaaclab/cameras_enabled": True}
    if stored is not _MISSING:
        settings_values["/isaaclab/render/deterministic"] = stored

    settings = MagicMock()
    settings.get.side_effect = settings_values.get
    determinism_mock = MagicMock()

    with (
        patch.object(rtx_renderer, "get_settings_manager", return_value=settings),
        patch.object(rtx_renderer, "apply_isaac_rtx_global_settings"),
        patch.object(rtx_renderer, "apply_isaac_rtx_determinism_settings", determinism_mock),
        patch.object(rtx_renderer, "ensure_rtx_hydra_engine_attached"),
    ):
        rtx_renderer.IsaacRtxRenderer(IsaacRtxRendererCfg())

    assert determinism_mock.called is expected_called
    if expected_called:
        determinism_mock.assert_called_once_with(settings)


def test_isaac_rtx_read_output_clears_stale_metadata_and_keeps_seeded_keys(monkeypatch):
    """read_output replaces (not merges): a dropped annotator info resets its info entry, seeded keys persist."""
    _install_omni_stubs(monkeypatch)
    from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderer

    from isaaclab.sensors.camera.camera_data import CameraData

    renderer = IsaacRtxRenderer.__new__(IsaacRtxRenderer)

    # ``camera_data.info`` is seeded with one key per output (mirrors ``camera_data.output``); both start None.
    camera_data = CameraData()
    camera_data.info = {"rgb": None, "semantic_segmentation": None}

    # Frame 1: the segmentation annotator emits metadata, so its info lands in camera_data.info.
    id_to_labels = {"2": {"class": "cartpole"}}
    render_data = SimpleNamespace(renderer_info={"semantic_segmentation": {"idToLabels": id_to_labels}})
    renderer.read_output(render_data, camera_data)
    assert camera_data.info["semantic_segmentation"] == {"idToLabels": id_to_labels}

    # Frame 2: the annotator emits no info (``renderer_info`` value is None or the key is gone).
    render_data = SimpleNamespace(renderer_info={"semantic_segmentation": None})
    renderer.read_output(render_data, camera_data)

    # The stale idToLabels must be cleared, and the seeded keys (rgb, semantic_segmentation) must remain.
    assert camera_data.info == {"rgb": None, "semantic_segmentation": None}
