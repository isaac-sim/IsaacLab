# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Isaac RTX renderer output contract."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

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
