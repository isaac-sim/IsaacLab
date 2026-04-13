# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for built-in RTX rendering mode preset data."""

from __future__ import annotations

import pytest

from isaaclab_physx.rendering.rtx_rendering_mode_presets import get_builtin_rtx_rendering_mode_preset


def test_builtin_rtx_presets_cover_three_modes():
    for name in ("performance", "balanced", "quality"):
        d = get_builtin_rtx_rendering_mode_preset(name)
        assert isinstance(d, dict)
        assert "/rtx/shadows/enabled" in d


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_builtin_rtx_rendering_mode_preset("not_a_mode")
