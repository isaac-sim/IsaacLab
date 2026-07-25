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
