# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVRTX USD render product authoring and stage export."""

from __future__ import annotations

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
        create_scene_partition_attributes,
        export_stage_to_string,
        get_render_var_config,
        get_render_var_configs,
    )

    from pxr import Sdf, Usd, UsdGeom  # noqa: E402
else:
    Sdf = None
    Usd = None
    UsdGeom = None
    build_render_scope_usd = None
    create_scene_partition_attributes = None
    export_stage_to_string = None
    get_render_var_config = None
    get_render_var_configs = None


def _make_multi_env_stage(num_envs: int) -> Usd.Stage:
    """Build an in-memory stage with distinguishable content per environment."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")

    for env_idx in range(num_envs):
        env_path = f"/World/envs/env_{env_idx}"
        UsdGeom.Xform.Define(stage, env_path)
        UsdGeom.Xform.Define(stage, f"{env_path}/Robot")
        UsdGeom.Xform.Define(stage, f"{env_path}/Object_env{env_idx}_only")
        UsdGeom.Camera.Define(stage, f"{env_path}/Camera")

    return stage


def _assert_export_contains_env_roots_and_children(exported: str, env_indices: range | list[int]) -> None:
    """Listed environment roots appear in the stage export."""
    for env_idx in env_indices:
        assert f'def Xform "env_{env_idx}"' in exported
        assert f'def Xform "Object_env{env_idx}_only"' in exported

    assert exported.count('def Xform "Robot"') == len(env_indices)
    assert exported.count('def Camera "Camera"') == len(env_indices)


def _assert_export_omits_env_children(exported: str, env_indices: range | list[int]) -> None:
    """Listed environments keep their roots but omit prototype children from the stage export."""
    for env_idx in env_indices:
        assert f'def Xform "env_{env_idx}"' in exported
        assert f'def Xform "Object_env{env_idx}_only"' not in exported


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


def test_export_stage_keeps_all_env_content_when_all_roots_are_sources():
    """Listing every env root as a source preserves the full stage content."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        num_envs=num_envs,
        source_paths=tuple(f"/World/envs/env_{env_idx}" for env_idx in range(num_envs)),
    )

    _assert_export_contains_env_roots_and_children(exported, range(num_envs))


def test_export_stage_full_when_single_env():
    """Single-environment stages are exported without trimming."""
    num_envs = 1
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        num_envs=num_envs,
        source_paths=("/World/envs/env_0",),
    )

    _assert_export_contains_env_roots_and_children(exported, range(num_envs))


def test_export_stage_homogeneous_keeps_only_env0_prototype():
    """Homogeneous cloning exports only the env_0 prototype subtree."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        num_envs=num_envs,
        source_paths=("/World/envs/env_0",),
    )

    _assert_export_contains_env_roots_and_children(exported, [0])
    _assert_export_omits_env_children(exported, range(1, num_envs))


def test_export_stage_heterogeneous_keeps_multiple_sources():
    """Heterogeneous source paths export only prototype env subtrees."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        num_envs=num_envs,
        source_paths=("/World/envs/env_0/Object_env0_only", "/World/envs/env_3/Object_env3_only"),
    )

    # Only the source subtrees are exported:
    assert 'def Xform "env_0"' in exported
    assert 'def Xform "Object_env0_only"' in exported
    assert 'def Xform "env_3"' in exported
    assert 'def Xform "Object_env3_only"' in exported

    # Other env roots remain, but their prototype children are omitted.
    _assert_export_omits_env_children(exported, [1, 2])
    assert 'def Xform "Robot"' not in exported
    assert 'def Camera "Camera"' not in exported


def test_export_stage_restores_active_state():
    """Export temporarily deactivates prims but restores them afterward."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    for env_idx in range(num_envs):
        env_path = f"/World/envs/env_{env_idx}"
        assert stage.GetPrimAtPath(env_path).IsActive()
        assert stage.GetPrimAtPath(f"{env_path}/Object_env{env_idx}_only").IsActive()

    export_stage_to_string(
        stage,
        num_envs=num_envs,
        source_paths=("/World/envs/env_0",),
    )

    for env_idx in range(num_envs):
        env_path = f"/World/envs/env_{env_idx}"
        assert stage.GetPrimAtPath(env_path).IsActive()
        assert stage.GetPrimAtPath(f"{env_path}/Object_env{env_idx}_only").IsActive()


def test_create_scene_partition_attributes_all_envs():
    """Scene partition attributes are authored on every env root and camera."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    create_scene_partition_attributes(stage, num_envs)

    root_layer = stage.GetRootLayer()
    for env_idx in range(num_envs):
        env_partition_attr = root_layer.GetAttributeAtPath(
            Sdf.Path(f"/World/envs/env_{env_idx}").AppendProperty("primvars:omni:scenePartition")
        )
        camera_partition_attr = root_layer.GetAttributeAtPath(
            Sdf.Path(f"/World/envs/env_{env_idx}/Camera").AppendProperty("omni:scenePartition")
        )
        assert env_partition_attr is not None
        assert env_partition_attr.default == f"env_{env_idx}"
        assert camera_partition_attr is not None
        assert camera_partition_attr.default == f"env_{env_idx}"
