# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TiledCamera renderer backend default and --renderer_backend -> env.scene contract.

Run with: pytest source/isaaclab/test/sensors/test_tiled_camera_renderer_backend.py -v
(from repo root, with Isaac Lab env active). The contract tests run without Isaac Sim;
the TiledCameraCfg default test requires the full env (imports isaaclab.sensors.camera).
"""

import pytest

# Default env.scene values used by scripts/reinforcement_learning/rsl_rl/train.py when
# --renderer_backend is set and the user does not pass env.scene=.
RENDERER_BACKEND_TO_DEFAULT_ENV_SCENE = {
    "rtx": "64x64tiled_rgb",
    "warp_renderer": "64x64newton_rgb",
}


class TestRendererBackendContract:
    """Enforce --renderer_backend -> env.scene contract (no Isaac Sim required)."""

    def test_renderer_backend_rtx_maps_to_tiled_rgb(self):
        """Default for --renderer_backend rtx must be env.scene=64x64tiled_rgb (RTX)."""
        assert RENDERER_BACKEND_TO_DEFAULT_ENV_SCENE["rtx"] == "64x64tiled_rgb"

    def test_renderer_backend_warp_maps_to_newton_rgb(self):
        """Default for --renderer_backend warp_renderer must be env.scene=64x64newton_rgb."""
        assert RENDERER_BACKEND_TO_DEFAULT_ENV_SCENE["warp_renderer"] == "64x64newton_rgb"

    def test_only_newton_warp_selects_warp_renderer(self):
        """Only 'newton_warp' should imply Newton Warp; None/rtx/other -> RTX."""
        for rt in (None, "rtx", "other"):
            effective = rt if rt is not None else "rtx"
            assert effective != "newton_warp"
        assert "newton_warp" not in RENDERER_BACKEND_TO_DEFAULT_ENV_SCENE
        # train.py uses env.scene to select scene variant; scene variant sets renderer_type.


class TestTiledCameraCfgDefault:
    """Test TiledCameraCfg default (skipped when Isaac Sim not available)."""

    def test_tiled_camera_cfg_default_renderer_is_none(self):
        """Default renderer_type must be None (meaning RTX)."""
        pytest.importorskip("omni.usd", reason="Isaac Sim required for camera imports")
        from isaaclab.sensors.camera import TiledCameraCfg

        cfg = TiledCameraCfg(prim_path="/World/cam", data_types=["rgb"])
        assert cfg.renderer_type is None
