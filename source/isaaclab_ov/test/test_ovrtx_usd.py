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
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_usd import (  # noqa: E402
        build_render_product_as_string,
        build_render_scope_usd,
        create_scene_partition_attributes,
        export_stage_to_string,
        get_render_var_config,
        get_render_var_configs,
        render_var_prim_paths_by_source,
    )

    from pxr import Sdf, Usd, UsdGeom  # noqa: E402
else:
    Sdf = None
    Usd = None
    UsdGeom = None
    build_render_product_as_string = None
    build_render_scope_usd = None
    create_scene_partition_attributes = None
    export_stage_to_string = None
    get_render_var_config = None
    get_render_var_configs = None
    render_var_prim_paths_by_source = None


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


def test_render_var_prim_paths_cover_every_authored_render_var():
    """Every render-var config this module authors is reachable by its source name."""
    prim_paths = render_var_prim_paths_by_source()

    authored = get_render_var_configs(
        ["rgb", "rgb_hdr", "albedo", "depth", "distance_to_camera", "normals", "motion_vectors"]
    ) + get_render_var_configs(["semantic_segmentation", "instance_segmentation"])

    for path, _, source in authored:
        assert prim_paths[source] == path


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


def test_ovrtx_primary_render_var_follows_the_first_requested_data_type():
    """The primary render var seeds the single-var arguments from the first requested data type."""
    assert get_render_var_config(["rgb", "motion_vectors"]) == ("/Render/Vars/LdrColor", "LdrColor", "LdrColor")
    assert get_render_var_config(["motion_vectors", "rgb"]) == (
        "/Render/Vars/TargetMotionSD",
        "TargetMotionSD",
        "TargetMotionSD",
    )


def test_ovrtx_authors_one_render_var_per_requested_data_type():
    """Every requested AOV is authored, so combining them no longer drops any."""
    data_types = [
        "rgb",
        "albedo",
        "semantic_segmentation",
        "instance_segmentation",
        "depth",
        "distance_to_camera",
        "normals",
        "motion_vectors",
    ]

    sources = [source for _, _, source in get_render_var_configs(data_types)]

    assert sources == [
        "LdrColor",
        "DiffuseAlbedoSD",
        "SemanticSegmentation",
        "NonStableInstanceSegmentation",
        "DistanceToImagePlaneSD",
        "DistanceToCameraSD",
        "NormalSD",
        "TargetMotionSD",
        "StableIdSemanticIdMap",
        "StableIdMap",
        "SemanticIdMap",
    ]


def test_ovrtx_data_types_sharing_a_source_author_one_render_var():
    """``rgb``/``rgba`` and ``depth``/``distance_to_image_plane`` collapse onto one render var each."""
    render_var_configs = get_render_var_configs(["rgb", "rgba", "depth", "distance_to_image_plane"])

    assert render_var_configs == [
        ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
        ("/Render/Vars/depth", "depth", "DistanceToImagePlaneSD"),
    ]


def test_ovrtx_depth_and_distance_to_camera_author_distinct_render_vars():
    """Image-plane depth and distance-to-camera are different sources and get separate prims."""
    render_scope = build_render_scope_usd(
        camera_paths=["/World/envs/env_0/Camera"],
        render_product_name="RenderProduct",
        render_var_path="/Render/Vars/depth",
        render_var_name="depth",
        source_name="DistanceToImagePlaneSD",
        tiled_width=16,
        tiled_height=8,
        render_var_configs=get_render_var_configs(["depth", "distance_to_camera"]),
    )

    assert "rel orderedVars = [</Render/Vars/depth>, </Render/Vars/DistanceToCameraSD>]" in render_scope
    assert 'uniform string sourceName = "DistanceToImagePlaneSD"' in render_scope
    assert 'uniform string sourceName = "DistanceToCameraSD"' in render_scope


def test_ovrtx_unsupported_data_type_is_skipped_and_falls_back_to_ldr_color():
    """Unsupported data types author no render var; an otherwise empty product keeps LdrColor."""
    assert get_render_var_configs(["instance_id_segmentation_fast"]) == [
        ("/Render/Vars/LdrColor", "LdrColor", "LdrColor")
    ]
    assert get_render_var_configs(["normals", "instance_id_segmentation_fast"]) == [
        ("/Render/Vars/NormalSD", "NormalSD", "NormalSD")
    ]


def test_ovrtx_rejects_color_combined_with_simple_shading():
    """Color and simple shading both read LdrColor, so one render product cannot serve both."""
    with pytest.raises(ValueError, match="simple shading"):
        get_render_var_configs(["rgb", "simple_shading_full_mdl"])


def test_ovrtx_rejects_multiple_simple_shading_data_types():
    """RTX Minimal mode is per render product, so only one simple shading output is possible."""
    with pytest.raises(ValueError, match="at most one simple shading"):
        get_render_var_configs(["simple_shading_constant_diffuse", "simple_shading_full_mdl"])


def test_ovrtx_simple_shading_alone_uses_ldr_color():
    """A lone simple shading request still reads LdrColor, shaded by RTX Minimal mode."""
    assert get_render_var_configs(["simple_shading_diffuse_mdl"]) == [("/Render/Vars/LdrColor", "LdrColor", "LdrColor")]


def test_ovrtx_duplicate_simple_shading_data_types_collapse():
    """Repeated identical simple-shading requests share one LdrColor render var."""
    assert get_render_var_configs(["simple_shading_full_mdl", "simple_shading_full_mdl"]) == [
        ("/Render/Vars/LdrColor", "LdrColor", "LdrColor")
    ]


def test_render_product_initially_targets_only_the_resolvable_source_camera():
    """Multi-environment RenderProducts initially target env zero while retaining tiled resolution."""
    render_product, render_product_path = build_render_product_as_string(
        width=16,
        height=8,
        num_envs=4,
        data_types=["rgb"],
        camera_path="/World/scenes/scene_2/Robot/head_cam",
    )

    assert render_product_path == "/Render/RenderProduct"
    assert "rel camera = [</World/scenes/scene_2/Robot/head_cam>]" in render_product
    assert "/World/scenes/scene_5/Robot/head_cam" not in render_product
    assert "uniform int2 resolution = (32, 16)" in render_product


def test_render_product_pins_device_ids_to_the_requested_cuda_device():
    """``device_id`` is authored as ``deviceIds`` so OVRTX allocates buffers on the reader's device."""
    render_product, render_product_path = build_render_product_as_string(
        width=16,
        height=8,
        num_envs=4,
        data_types=["rgb"],
        camera_path="/World/envs/env_0/Camera",
        device_id=1,
    )

    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString("#usda 1.0\n" + render_product)
    device_ids = layer.GetAttributeAtPath(f"{render_product_path}.deviceIds")
    assert device_ids is not None
    assert device_ids.typeName == Sdf.ValueTypeNames.UIntArray
    assert list(device_ids.default) == [1]


def test_render_product_omits_device_ids_when_no_device_is_given():
    """Without a device index the render product keeps OVRTX's automatic device assignment."""
    render_product, _ = build_render_product_as_string(
        width=16, height=8, num_envs=4, data_types=["rgb"], camera_path="/World/envs/env_0/Camera"
    )

    assert "deviceIds" not in render_product


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


def test_export_stage_keeps_all_env_content_when_all_roots_are_sources():
    """Listing every env root as a source preserves the full stage content."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        tuple(f"/World/envs/env_{env_idx}" for env_idx in range(num_envs)),
        source_paths=tuple(f"/World/envs/env_{env_idx}" for env_idx in range(num_envs)),
    )

    _assert_export_contains_env_roots_and_children(exported, range(num_envs))


def test_export_stage_full_when_single_env():
    """Single-environment stages are exported without trimming."""
    num_envs = 1
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        ("/World/envs/env_0",),
        source_paths=("/World/envs/env_0",),
    )

    _assert_export_contains_env_roots_and_children(exported, range(num_envs))


def test_export_stage_homogeneous_keeps_only_env0_prototype():
    """Homogeneous cloning exports only the env_0 prototype subtree."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        tuple(f"/World/envs/env_{env_idx}" for env_idx in range(num_envs)),
        source_paths=("/World/envs/env_0",),
    )

    _assert_export_contains_env_roots_and_children(exported, [0])
    _assert_export_omits_env_children(exported, range(1, num_envs))


def test_export_stage_trims_environment_siblings_outside_the_plan():
    """An unrelated staged environment cannot leak geometry into the detached renderer."""
    stage = _make_multi_env_stage(7)

    exported = export_stage_to_string(
        stage,
        ("/World/envs/env_2", "/World/envs/env_5"),
        source_paths=("/World/envs/env_2/Robot",),
    )

    assert 'def Xform "env_2"' in exported
    assert 'def Xform "Robot"' in exported
    assert 'def Xform "env_5"' in exported
    assert 'def Xform "Object_env5_only"' not in exported
    for env_idx in (0, 1, 3, 4, 6):
        assert f'def Xform "env_{env_idx}"' not in exported
        assert f'def Xform "Object_env{env_idx}_only"' not in exported


def test_export_stage_without_keep_env_roots_trims_non_source_env_roots():
    """The ovstage clone path also trims the non-source env roots themselves.

    ``ovstage.Stage.clone`` requires every target path to not already exist, so the exported stage
    must not retain env roots that the clone will recreate. This is the only difference from the
    legacy ``renderer.clone_usd`` path, which keeps the roots as placeholders.
    """
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        tuple(f"/World/envs/env_{env_idx}" for env_idx in range(num_envs)),
        source_paths=("/World/envs/env_0",),
        keep_env_roots=False,
    )

    _assert_export_contains_env_roots_and_children(exported, [0])
    for env_idx in range(1, num_envs):
        assert f'def Xform "env_{env_idx}"' not in exported
        assert f'def Xform "Object_env{env_idx}_only"' not in exported


def test_export_stage_heterogeneous_keeps_multiple_sources():
    """Heterogeneous source paths export only prototype env subtrees."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        tuple(f"/World/envs/env_{env_idx}" for env_idx in range(num_envs)),
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
        tuple(f"/World/envs/env_{env_idx}" for env_idx in range(num_envs)),
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

    env_paths = tuple(f"/World/envs/env_{env_idx}" for env_idx in range(num_envs))
    camera_paths = tuple(f"{env_path}/Camera" for env_path in env_paths)
    create_scene_partition_attributes(stage, dict(zip(env_paths, camera_paths, strict=True)))

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
