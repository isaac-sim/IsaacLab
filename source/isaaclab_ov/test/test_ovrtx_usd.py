# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVRTX USD render product authoring."""

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_ov", "pxr")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_usd import (  # noqa: E402
        build_render_scope_usd,
        get_render_var_config,
        get_render_var_configs,
    )
else:
    build_render_scope_usd = None
    get_render_var_config = None
    get_render_var_configs = None


def test_ovrtx_rgb_hdr_uses_hdr_color_render_var():
    """Requesting RGB_HDR from OVRTX selects the HdrColor render variable."""
    assert get_render_var_config(["rgb_hdr"]) == ("/Render/Vars/HdrColor", "HdrColor", "HdrColor")


def test_ovrtx_rgb_and_rgb_hdr_author_both_render_vars():
    """Requesting LDR RGB and RGB_HDR keeps both OVRTX render variables."""
    render_var_configs = get_render_var_configs(["rgb", "rgb_hdr"])

    assert render_var_configs == [
        ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
        ("/Render/Vars/HdrColor", "HdrColor", "HdrColor"),
    ]

    render_scope = build_render_scope_usd(
        camera_paths=["/World/envs/env_0/Camera"],
        render_product_name="RenderProduct",
        render_var_path=render_var_configs[0][0],
        render_var_name=render_var_configs[0][1],
        source_name=render_var_configs[0][2],
        tiled_width=16,
        tiled_height=8,
        render_var_configs=render_var_configs,
    )

    assert "rel orderedVars = [</Render/Vars/LdrColor>, </Render/Vars/HdrColor>]" in render_scope
    assert 'def RenderVar "LdrColor"' in render_scope
    assert 'def RenderVar "HdrColor"' in render_scope
