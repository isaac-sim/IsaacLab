# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for renderer configurations."""

import pytest

from isaaclab.renderer import (
    NewtonWarpRendererCfg,
    OVRTXRendererCfg,
    RendererCfg,
    get_renderer_class,
)


class TestRendererCfg:
    """Tests for base RendererCfg."""

    def test_default_values(self):
        """Test default configuration values."""
        cfg = RendererCfg()
        assert cfg.renderer_type == "base"
        assert cfg.height == 1024
        assert cfg.width == 1024
        assert cfg.num_envs == 1
        assert cfg.num_cameras == 1
        assert cfg.data_types == []

    def test_getters(self):
        """Test config getter methods."""
        cfg = RendererCfg(height=480, width=640, num_envs=8)
        assert cfg.get_height() == 480
        assert cfg.get_width() == 640
        assert cfg.get_num_envs() == 8
        assert cfg.get_renderer_type() == "base"


class TestOVRTXRendererCfg:
    """Tests for OVRTXRendererCfg."""

    def test_inherits_renderer_cfg(self):
        """Test OVRTX config inherits from RendererCfg."""
        cfg = OVRTXRendererCfg(width=128, height=128, num_envs=4)
        assert cfg.renderer_type == "ov_rtx"
        assert cfg.width == 128
        assert cfg.height == 128
        assert cfg.num_envs == 4

    def test_ovrtx_specific_fields(self):
        """Test OVRTX-specific configuration fields."""
        cfg = OVRTXRendererCfg()
        assert hasattr(cfg, "simple_shading_mode")
        assert cfg.simple_shading_mode is True
        assert hasattr(cfg, "temp_usd_dir")
        assert hasattr(cfg, "temp_usd_suffix")


class TestNewtonWarpRendererCfg:
    """Tests for NewtonWarpRendererCfg."""

    def test_inherits_renderer_cfg(self):
        """Test Newton config inherits from RendererCfg."""
        cfg = NewtonWarpRendererCfg(width=100, height=100, num_envs=16)
        assert cfg.renderer_type == "newton_warp"
        assert cfg.width == 100
        assert cfg.height == 100
        assert cfg.num_envs == 16


class TestGetRendererClass:
    """Tests for get_renderer_class dispatch."""

    def test_newton_warp_dispatch(self):
        """Test get_renderer_class returns NewtonWarpRenderer for NewtonWarpRendererCfg."""
        pytest.importorskip("newton")
        cfg = NewtonWarpRendererCfg()
        cls = get_renderer_class(cfg)
        assert cls is not None
        assert cls.__name__ == "NewtonWarpRenderer"

    def test_ovrtx_dispatch(self):
        """Test get_renderer_class returns OVRTXRenderer for OVRTXRendererCfg."""
        pytest.importorskip("ovrtx")
        cfg = OVRTXRendererCfg()
        cls = get_renderer_class(cfg)
        assert cls is not None
        assert cls.__name__ == "OVRTXRenderer"
