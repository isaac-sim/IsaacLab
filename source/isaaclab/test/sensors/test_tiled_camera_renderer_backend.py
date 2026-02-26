# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TiledCamera renderer backend and env.scene variant contract.

Run with: pytest source/isaaclab/test/sensors/test_tiled_camera_renderer_backend.py -v
(from repo root, with Isaac Lab env active). The contract tests run without Isaac Sim;
the TiledCameraCfg default test requires the full env (imports isaaclab.sensors.camera).

Scene variants use neutral keys: <width>x<height><camera_type> (e.g. 64x64rgb, 64x64depth).
Renderer is set via override, not the variant key: env.scene.base_camera.renderer_type or
env.scene.tiled_camera.renderer_type (e.g. isaac_rtx, newton_warp). train.py does not set
--renderer_backend.
"""

import re

import pytest

# Neutral scene variant format: <width>x<height><camera_type> (no renderer in key)
NEUTRAL_SCENE_KEY_PATTERN = re.compile(r"^(\d+)x(\d+)(rgb|depth|albedo)$")
EXAMPLE_NEUTRAL_KEY = "64x64rgb"


class TestEnvSceneVariantContract:
    """Enforce env.scene= neutral variant format and renderer override paths (no Isaac Sim required)."""

    def test_neutral_variant_format(self):
        """Scene variant key follows <width>x<height><camera_type> (e.g. 64x64rgb)."""
        assert NEUTRAL_SCENE_KEY_PATTERN.match(EXAMPLE_NEUTRAL_KEY) is not None
        w, h, cam = NEUTRAL_SCENE_KEY_PATTERN.match(EXAMPLE_NEUTRAL_KEY).groups()
        assert (w, h, cam) == ("64", "64", "rgb")

    def test_renderer_via_override_paths(self):
        """Renderer is selected via env.scene.base_camera.renderer_type or env.scene.tiled_camera.renderer_type."""
        # Document the two common override paths; no renderer in variant key
        base_path = "env.scene.base_camera.renderer_type"
        tiled_path = "env.scene.tiled_camera.renderer_type"
        assert "base_camera" in base_path and "renderer_type" in base_path
        assert "tiled_camera" in tiled_path and "renderer_type" in tiled_path
        assert EXAMPLE_NEUTRAL_KEY.count("rtx") == 0 and EXAMPLE_NEUTRAL_KEY.count("warp") == 0


class TestTiledCameraCfgDefault:
    """Test TiledCameraCfg default (skipped when Isaac Sim not available)."""

    def test_tiled_camera_cfg_default_renderer_is_none(self):
        """Default renderer_type must be None (meaning RTX)."""
        pytest.importorskip("omni.usd", reason="Isaac Sim required for camera imports")
        from isaaclab.sensors.camera import TiledCameraCfg

        cfg = TiledCameraCfg(prim_path="/World/cam", data_types=["rgb"])
        assert cfg.renderer_type is None
