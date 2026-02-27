# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for TiledCamera renderer backend.

Run with: pytest source/isaaclab/test/sensors/test_tiled_camera_renderer_backend.py -v
(from repo root, with Isaac Lab env active). The TiledCameraCfg default test requires the full env
(imports isaaclab.sensors.camera). Renderer is set via the top-level Hydra group: renderer=isaac_rtx
or renderer=newton_warp (config store applies the preset to all cameras).
"""

import pytest


class TestTiledCameraCfgDefault:
    """Test TiledCameraCfg default (skipped when Isaac Sim not available)."""

    def test_tiled_camera_cfg_has_renderer_cfg(self):
        """TiledCameraCfg inherits renderer_cfg from CameraCfg (default isaac_rtx)."""
        pytest.importorskip("omni.usd", reason="Isaac Sim required for camera imports")
        from isaaclab.sensors.camera import TiledCameraCfg

        cfg = TiledCameraCfg(prim_path="/World/cam", data_types=["rgb"])
        assert cfg.renderer_cfg is not None
        assert getattr(cfg.renderer_cfg, "renderer_type", None) == "isaac_rtx"
