# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TiledCamera renderer backend and env.scene variant contract.

Run with: pytest source/isaaclab/test/sensors/test_tiled_camera_renderer_backend.py -v
(from repo root, with Isaac Lab env active). The contract tests run without Isaac Sim;
the TiledCameraCfg default test requires the full env (imports isaaclab.sensors.camera).

Renderer is dictated by env.scene=: tiled variants (e.g. 64x64tiled_rgb) use RTX,
newton variants (e.g. 64x64newton_rgb) use Warp. train.py does not set --renderer_backend.
"""

import pytest

# env.scene variant names that select each backend (task defines these in scene variants)
ENV_SCENE_RTX = "64x64tiled_rgb"
ENV_SCENE_WARP = "64x64newton_rgb"


class TestEnvSceneVariantContract:
    """Enforce env.scene= variant names for RTX vs Warp (no Isaac Sim required)."""

    def test_tiled_variant_is_rtx(self):
        """Tiled variant name (64x64tiled_rgb) selects RTX backend."""
        assert "tiled" in ENV_SCENE_RTX

    def test_newton_variant_is_warp(self):
        """Newton variant name (64x64newton_rgb) selects Warp backend."""
        assert "newton" in ENV_SCENE_WARP


class TestTiledCameraCfgDefault:
    """Test TiledCameraCfg default (skipped when Isaac Sim not available)."""

    def test_tiled_camera_cfg_default_renderer_is_none(self):
        """Default renderer_type must be None (meaning RTX)."""
        pytest.importorskip("omni.usd", reason="Isaac Sim required for camera imports")
        from isaaclab.sensors.camera import TiledCameraCfg

        cfg = TiledCameraCfg(prim_path="/World/cam", data_types=["rgb"])
        assert cfg.renderer_type is None
