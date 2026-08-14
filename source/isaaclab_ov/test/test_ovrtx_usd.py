# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVRTX USD render product authoring."""

from __future__ import annotations

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_ov",)
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
        build_render_product_as_string,
        build_render_scope_usd,
        get_render_var_config,
        get_render_var_configs,
    )
else:
    build_render_product_as_string = None
    build_render_scope_usd = None
    get_render_var_config = None
    get_render_var_configs = None


def test_build_render_scope_usd_default_background_is_dome_light():
    """Default background (background_color=None) uses domeLight source type."""
    render_scope = build_render_scope_usd(
        camera_paths=["/World/envs/env_0/Camera"],
        render_product_name="RenderProduct",
        render_var_path="/Render/Vars/LdrColor",
        render_var_name="LdrColor",
        source_name="LdrColor",
        tiled_width=16,
        tiled_height=8,
    )
    assert 'token omni:rtx:background:source:type = "domeLight"' in render_scope
    assert "omni:rtx:background:source:color" not in render_scope


def test_build_render_scope_usd_solid_background_color():
    """Providing background_color emits color source type and the color attribute."""
    render_scope = build_render_scope_usd(
        camera_paths=["/World/envs/env_0/Camera"],
        render_product_name="RenderProduct",
        render_var_path="/Render/Vars/LdrColor",
        render_var_name="LdrColor",
        source_name="LdrColor",
        tiled_width=16,
        tiled_height=8,
        background_color=(1.0, 0.0, 0.5),
    )
    assert 'token omni:rtx:background:source:type = "color"' in render_scope
    assert "color3f omni:rtx:background:source:color = (1.0, 0.0, 0.5)" in render_scope
    assert 'token omni:rtx:background:source:type = "domeLight"' not in render_scope


def test_ovrtx_rgb_hdr_uses_hdr_color_render_var():
    """Requesting RGB_HDR from OVRTX selects the HdrColor render variable."""
    assert get_render_var_config(["rgb_hdr"]) == ("/Render/Vars/HdrColor", "HdrColor", "HdrColor")


def test_ovrtx_instance_segmentation_uses_non_stable_instance_segmentation_render_var():
    """Requesting instance_segmentation from OVRTX selects the NonStableInstanceSegmentation render var."""
    assert get_render_var_config(["instance_segmentation"]) == (
        "/Render/Vars/NonStableInstanceSegmentation",
        "NonStableInstanceSegmentation",
        "NonStableInstanceSegmentation",
    )


def test_ovrtx_motion_vectors_uses_target_motion_render_var():
    """Requesting motion vectors from OVRTX selects the TargetMotionSD render variable."""
    assert get_render_var_config(["motion_vectors"]) == (
        "/Render/Vars/TargetMotionSD",
        "TargetMotionSD",
        "TargetMotionSD",
    )


def test_ovrtx_motion_vectors_with_rgb_falls_back_to_rgb():
    """OVRTX only supports one main AOV at a time; combining motion vectors with RGB keeps RGB."""
    assert get_render_var_config(["rgb", "motion_vectors"]) == ("/Render/Vars/LdrColor", "LdrColor", "LdrColor")


def test_render_product_initially_targets_only_the_resolvable_source_camera():
    """Multi-environment RenderProducts initially target the authored prototype camera."""
    render_product, render_product_path = build_render_product_as_string(
        width=16,
        height=8,
        num_envs=4,
        data_types=["rgb"],
        camera_path="/World/scenes/scene_7/Robot/head_cam",
    )

    assert render_product_path == "/Render/RenderProduct"
    assert "rel camera = [</World/scenes/scene_7/Robot/head_cam>]" in render_product
    assert "/World/envs/" not in render_product
    assert "uniform int2 resolution = (32, 16)" in render_product


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


def test_ovrtx_semantic_segmentation_authors_semantic_and_id_map_render_vars():
    """Requesting semantic segmentation authors both SemanticSegmentation and SemanticIdMap render vars."""
    render_var_configs = get_render_var_configs(["semantic_segmentation"])

    assert render_var_configs == [
        ("/Render/Vars/semantic", "semantic", "SemanticSegmentation"),
        ("/Render/Vars/SemanticIdMap", "SemanticIdMap", "SemanticIdMap"),
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

    assert "rel orderedVars = [</Render/Vars/semantic>, </Render/Vars/SemanticIdMap>]" in render_scope
    assert 'uniform string sourceName = "SemanticSegmentation"' in render_scope
    assert 'uniform string sourceName = "SemanticIdMap"' in render_scope


def test_ovrtx_instance_segmentation_authors_pixel_and_map_render_vars():
    """Requesting instance segmentation authors the pixel AOV plus the three ID/label map render vars."""
    render_var_configs = get_render_var_configs(["instance_segmentation"])

    assert render_var_configs == [
        (
            "/Render/Vars/NonStableInstanceSegmentation",
            "NonStableInstanceSegmentation",
            "NonStableInstanceSegmentation",
        ),
        ("/Render/Vars/StableIdSemanticIdMap", "StableIdSemanticIdMap", "StableIdSemanticIdMap"),
        ("/Render/Vars/StableIdMap", "StableIdMap", "StableIdMap"),
        ("/Render/Vars/SemanticIdMap", "SemanticIdMap", "SemanticIdMap"),
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

    assert (
        "rel orderedVars = [</Render/Vars/NonStableInstanceSegmentation>, </Render/Vars/StableIdSemanticIdMap>,"
        " </Render/Vars/StableIdMap>, </Render/Vars/SemanticIdMap>]" in render_scope
    )
    assert 'uniform string sourceName = "StableIdSemanticIdMap"' in render_scope
    assert 'uniform string sourceName = "StableIdMap"' in render_scope


def test_ovrtx_semantic_and_instance_segmentation_share_a_single_semantic_id_map():
    """Requesting both segmentation outputs authors ``SemanticIdMap`` exactly once (it is shared)."""
    render_var_configs = get_render_var_configs(["semantic_segmentation", "instance_segmentation"])

    sources = [source for _, _, source in render_var_configs]
    assert sources.count("SemanticIdMap") == 1
    # Both segmentations' map render vars are authored regardless of which AOV get_render_var_config resolves.
    assert {"SemanticIdMap", "StableIdSemanticIdMap", "StableIdMap"} <= set(sources)
