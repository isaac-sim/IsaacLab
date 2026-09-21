# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Isaac RTX renderer configuration."""

import isaaclab_physx.renderers as renderers
from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg


def test_global_settings_cfg_is_publicly_exported():
    """Test that the global settings config is exported from the renderer package."""
    assert renderers.IsaacRtxRendererGlobalSettingsCfg is IsaacRtxRendererGlobalSettingsCfg
    assert "IsaacRtxRendererGlobalSettingsCfg" in renderers.__all__


def test_global_view_preserves_launch_setting_by_default():
    """Renderer construction should preserve AppLauncher's spectator-view choice."""
    assert IsaacRtxRendererCfg().global_settings.show_all_partitions_by_default is None
