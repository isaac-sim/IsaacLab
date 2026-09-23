# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVRTX USD render product authoring and stage export."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

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
        render_var_prim_names_by_source,
        render_var_prim_paths_by_source,
    )

    from pxr import Sdf, Usd, UsdGeom  # noqa: E402

    from isaaclab.renderers.camera_render_spec import CameraRenderSpec  # noqa: E402
    from isaaclab.sensors.camera import CameraCfg  # noqa: E402
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
    render_var_prim_names_by_source = None
    render_var_prim_paths_by_source = None


@pytest.fixture
def camera_spec():
    cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/head_cam",
        spawn=None,
        width=16,
        height=8,
        data_types=["rgb"],
    )
    return CameraRenderSpec(
        cfg=cfg,
        device="cpu",
        num_instances=4,
        camera_prim_paths=tuple(f"/World/envs/env_{i}/Robot/head_cam" for i in range(4)),
        view_count=4,
        camera_path_relative_to_env_0="Robot/head_cam",
    )


@pytest.fixture
def render_data():
    return SimpleNamespace(render_scope_name="RenderCamera_0", render_product_name="RenderProduct")


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


def test_render_product_default_background_is_dome_light(camera_spec, render_data):
    """Default background (background_color=None) uses domeLight source type."""
    render_scope = build_render_product_as_string(camera_spec, render_data)
    assert 'token omni:rtx:background:source:type = "domeLight"' in render_scope
    assert "omni:rtx:background:source:color" not in render_scope
    assert 'token omni:rtx:rendermode = "RealTimePathTracing"' in render_scope
    assert "omni:rtx:minimal:" not in render_scope


def test_render_product_solid_background_color(camera_spec, render_data):
    """Providing background_color emits color source type and the color attribute."""
    camera_spec.cfg.background_color = (1.0, 0.0, 0.5)
    render_scope = build_render_product_as_string(camera_spec, render_data)
    assert 'token omni:rtx:background:source:type = "color"' in render_scope
    assert "color3f omni:rtx:background:source:color = (1.0, 0.0, 0.5)" in render_scope
    assert 'token omni:rtx:background:source:type = "domeLight"' not in render_scope


def test_build_render_scope_usd_authors_ovrtx_rtx_settings(camera_spec, render_data):
    """OVRTX settings are authored on the RenderProduct without its settings extension."""
    render_scope = build_render_scope_usd(
        camera_spec,
        render_data,
        render_mode="PathTracing",
        enable_accumulation=True,
        accumulation_limit=7,
        gaussian_accumulated_albedo=True,
        gaussian_skip_tonemapping=True,
    )

    assert (
        'prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1", "OmniRtxSettingsRtAPI_1", '
        '"OmniRtxSettingsParticleFieldAPI_1"]'
    ) in render_scope
    assert 'token omni:rtx:rendermode = "PathTracing"' in render_scope
    assert "bool omni:rtx:rt:accumulation:enabled = true" in render_scope
    assert "int omni:rtx:rt:accumulationLimit = 7" in render_scope
    assert "bool omni:rtx:rtpt:gaussian:accumulatedAlbedo:enabled = true" in render_scope
    assert "bool omni:rtx:rtpt:gaussian:skipTonemapping:enabled = true" in render_scope


def test_build_render_scope_usd_rejects_minimal_without_simple_shading(camera_spec, render_data):
    """Minimal mode needs the simple-shading mode that selects its output."""
    with pytest.raises(ValueError, match="requires a simple-shading output"):
        build_render_scope_usd(
            camera_spec,
            render_data,
            render_mode="Minimal",
        )


def test_ovrtx_rgb_hdr_uses_hdr_color_render_var():
    """Requesting RGB_HDR from OVRTX selects the HdrColor render variable."""
    assert get_render_var_config(["rgb_hdr"], render_scope_name="RenderCamera_0") == (
        "/RenderCamera_0/Vars/HdrColor",
        "HdrColor",
        "HdrColor",
    )


def test_render_var_prim_names_are_read_only():
    with pytest.raises(TypeError):
        render_var_prim_names_by_source()["LdrColor"] = "mutated"  # type: ignore[index]


@pytest.mark.parametrize("render_scope_name", ["RenderCamera_0", "RenderCamera_1"])
def test_render_var_prim_paths_cover_every_authored_render_var(render_scope_name: str, camera_spec, render_data):
    """Every output and metadata path resolves to its authored prim under the requested scope."""
    data_types = [
        "rgb",
        "rgb_hdr",
        "albedo",
        "depth",
        "distance_to_camera",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation",
    ]
    prim_paths = render_var_prim_paths_by_source(render_scope_name=render_scope_name)
    authored = get_render_var_configs(data_types, render_scope_name=render_scope_name)
    camera_spec.cfg.data_types = data_types
    render_data.render_product_name = "CameraOutput"
    render_data.render_scope_name = render_scope_name
    render_scope = build_render_product_as_string(camera_spec, render_data)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(render_scope)
    stage = Usd.Stage.Open(layer)
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path(f"/{render_scope_name}")
    product = stage.GetPrimAtPath(f"/{render_scope_name}/CameraOutput")
    assert product.GetTypeName() == "RenderProduct"
    targets = product.GetRelationship("orderedVars").GetTargets()
    assert targets == [Sdf.Path(path) for path, _, _ in authored]
    assert len(targets) == len(prim_paths)

    for path, name, source in authored:
        prim = stage.GetPrimAtPath(path)
        assert prim.GetName() == name
        assert prim.GetAttribute("sourceName").Get() == source
        assert prim.GetParent().GetPath() == Sdf.Path(f"/{render_scope_name}/Vars")
        assert prim_paths[source] == path


def test_ovrtx_instance_segmentation_uses_non_stable_instance_segmentation_render_var():
    """Requesting instance_segmentation from OVRTX selects the NonStableInstanceSegmentation render var."""
    assert get_render_var_config(["instance_segmentation"], render_scope_name="RenderCamera_0") == (
        "/RenderCamera_0/Vars/NonStableInstanceSegmentation",
        "NonStableInstanceSegmentation",
        "NonStableInstanceSegmentation",
    )


def test_ovrtx_motion_vectors_uses_target_motion_render_var():
    """Requesting motion vectors from OVRTX selects the TargetMotionSD render variable."""
    assert get_render_var_config(["motion_vectors"], render_scope_name="RenderCamera_0") == (
        "/RenderCamera_0/Vars/TargetMotionSD",
        "TargetMotionSD",
        "TargetMotionSD",
    )


def test_ovrtx_primary_render_var_follows_the_first_requested_data_type():
    """The primary render var is the first supported configuration in the requested data types."""
    assert get_render_var_config(["rgb", "motion_vectors"], render_scope_name="RenderCamera_0") == (
        "/RenderCamera_0/Vars/LdrColor",
        "LdrColor",
        "LdrColor",
    )
    assert get_render_var_config(["motion_vectors", "rgb"], render_scope_name="RenderCamera_0") == (
        "/RenderCamera_0/Vars/TargetMotionSD",
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

    sources = [source for _, _, source in get_render_var_configs(data_types, render_scope_name="RenderCamera_0")]

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
    render_var_configs = get_render_var_configs(
        ["rgb", "rgba", "depth", "distance_to_image_plane"], render_scope_name="RenderCamera_0"
    )

    assert render_var_configs == [
        ("/RenderCamera_0/Vars/LdrColor", "LdrColor", "LdrColor"),
        ("/RenderCamera_0/Vars/depth", "depth", "DistanceToImagePlaneSD"),
    ]


def test_ovrtx_depth_and_distance_to_camera_author_distinct_render_vars(camera_spec, render_data):
    """Image-plane depth and distance-to-camera are different sources and get separate prims."""
    camera_spec.cfg.data_types = ["depth", "distance_to_camera"]
    render_scope = build_render_scope_usd(camera_spec, render_data)

    assert "rel orderedVars = [</RenderCamera_0/Vars/depth>, </RenderCamera_0/Vars/DistanceToCameraSD>]" in render_scope
    assert 'uniform string sourceName = "DistanceToImagePlaneSD"' in render_scope
    assert 'uniform string sourceName = "DistanceToCameraSD"' in render_scope


def test_ovrtx_unsupported_data_type_is_skipped_and_falls_back_to_ldr_color():
    """Unsupported data types author no render var; an otherwise empty product keeps LdrColor."""
    assert get_render_var_configs(["instance_id_segmentation_fast"], render_scope_name="RenderCamera_0") == [
        ("/RenderCamera_0/Vars/LdrColor", "LdrColor", "LdrColor")
    ]
    assert get_render_var_configs(["normals", "instance_id_segmentation_fast"], render_scope_name="RenderCamera_0") == [
        ("/RenderCamera_0/Vars/NormalSD", "NormalSD", "NormalSD")
    ]


def test_ovrtx_rejects_color_combined_with_simple_shading():
    """Color and simple shading both read LdrColor, so one render product cannot serve both."""
    with pytest.raises(ValueError, match="simple shading"):
        get_render_var_configs(["rgb", "simple_shading_full_mdl"], render_scope_name="RenderCamera_0")


def test_ovrtx_rejects_multiple_simple_shading_data_types():
    """RTX Minimal mode is per render product, so only one simple shading output is possible."""
    with pytest.raises(ValueError, match="at most one simple shading"):
        get_render_var_configs(
            ["simple_shading_constant_diffuse", "simple_shading_full_mdl"], render_scope_name="RenderCamera_0"
        )


@pytest.mark.parametrize(
    ("data_type", "minimal_mode", "enable_shadows"),
    [
        ("simple_shading_constant_diffuse", 1, False),
        ("simple_shading_diffuse_mdl", 2, True),
        ("simple_shading_full_mdl", 3, False),
    ],
)
def test_ovrtx_simple_shading_alone_uses_ldr_color(camera_spec, render_data, data_type, minimal_mode, enable_shadows):
    """Simple shading selects LdrColor and the matching RTX Minimal mode with the configured shadows."""
    assert get_render_var_configs([data_type], render_scope_name="RenderCamera_0") == [
        ("/RenderCamera_0/Vars/LdrColor", "LdrColor", "LdrColor")
    ]
    camera_spec.cfg.data_types = [data_type]
    render_product = build_render_product_as_string(camera_spec, render_data, enable_shadows=enable_shadows)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(render_product)
    assert layer.GetAttributeAtPath("/RenderCamera_0/RenderProduct.omni:rtx:rendermode").default == "Minimal"
    assert layer.GetAttributeAtPath("/RenderCamera_0/RenderProduct.omni:rtx:minimal:mode").default == minimal_mode
    assert (
        layer.GetAttributeAtPath("/RenderCamera_0/RenderProduct.omni:rtx:minimal:castShadows").default == enable_shadows
    )


def test_ovrtx_duplicate_simple_shading_data_types_collapse():
    """Repeated identical simple-shading requests share one LdrColor render var."""
    assert get_render_var_configs(
        ["simple_shading_full_mdl", "simple_shading_full_mdl"], render_scope_name="RenderCamera_0"
    ) == [("/RenderCamera_0/Vars/LdrColor", "LdrColor", "LdrColor")]


def test_render_product_initially_targets_only_the_resolvable_source_camera(camera_spec, render_data):
    """Multi-environment RenderProducts initially target env zero while retaining tiled resolution."""
    render_product = build_render_product_as_string(camera_spec, render_data)

    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(render_product)
    assert layer.GetPrimAtPath("/RenderCamera_0/RenderProduct").typeName == "RenderProduct"
    assert "rel camera = [</World/envs/env_0/Robot/head_cam>]" in render_product
    assert "/World/envs/env_1/Robot/head_cam" not in render_product
    assert "uniform int2 resolution = (32, 16)" in render_product


def test_render_product_pins_device_ids_to_the_requested_cuda_device(camera_spec, render_data):
    """``device_id`` is authored as ``deviceIds`` so OVRTX allocates buffers on the reader's device."""
    render_product = build_render_product_as_string(camera_spec, render_data, device_id=1)

    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(render_product)
    device_ids = layer.GetAttributeAtPath("/RenderCamera_0/RenderProduct.deviceIds")
    assert device_ids is not None
    assert device_ids.typeName == Sdf.ValueTypeNames.UIntArray
    assert list(device_ids.default) == [1]


def test_render_product_omits_device_ids_when_no_device_is_given(camera_spec, render_data):
    """Without a device index the render product keeps OVRTX's automatic device assignment."""
    render_product = build_render_product_as_string(camera_spec, render_data)

    assert "deviceIds" not in render_product


@pytest.mark.parametrize("data_types", [["rgb"], ["rgb", "rgb_hdr"], []])
def test_render_product_isp_requests_hdr_without_mutating_camera_outputs(camera_spec, render_data, data_types):
    """ISP receives one HDR source while the camera's requested outputs remain unchanged."""
    camera_spec.cfg.data_types = data_types.copy()
    camera_spec.cfg.isp_cfg = object()
    render_product = build_render_product_as_string(camera_spec, render_data)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(render_product)
    ordered_vars = layer.GetRelationshipAtPath("/RenderCamera_0/RenderProduct.orderedVars")
    assert list(ordered_vars.targetPathList.explicitItems) == [
        Sdf.Path("/RenderCamera_0/Vars/LdrColor"),
        Sdf.Path("/RenderCamera_0/Vars/HdrColor"),
    ]
    assert camera_spec.cfg.data_types == data_types


def test_ovrtx_rgb_and_rgb_hdr_author_both_render_vars(camera_spec, render_data):
    """Requesting LDR RGB and RGB_HDR keeps both OVRTX render variables."""
    render_var_configs = get_render_var_configs(["rgb", "rgb_hdr"], render_scope_name="RenderCamera_0")

    assert render_var_configs == [
        ("/RenderCamera_0/Vars/LdrColor", "LdrColor", "LdrColor"),
        ("/RenderCamera_0/Vars/HdrColor", "HdrColor", "HdrColor"),
    ]

    camera_spec.cfg.data_types = ["rgb", "rgb_hdr"]
    render_scope = build_render_scope_usd(camera_spec, render_data)

    assert "rel orderedVars = [</RenderCamera_0/Vars/LdrColor>, </RenderCamera_0/Vars/HdrColor>]" in render_scope
    assert 'def RenderVar "LdrColor"' in render_scope
    assert 'def RenderVar "HdrColor"' in render_scope


def test_ovrtx_hdr_render_product_disables_gaussian_skip_tonemapping(camera_spec, render_data):
    """OVRTX 0.5 reads Gaussian tonemapping from the RenderProduct, not global settings."""
    camera_spec.cfg.data_types = ["rgb_hdr"]

    render_scope = build_render_scope_usd(camera_spec, render_data, gaussian_skip_tonemapping=True)

    assert "bool omni:rtx:rtpt:gaussian:skipTonemapping:enabled = false" in render_scope


def test_ovrtx_semantic_segmentation_authors_semantic_and_id_map_render_vars(camera_spec, render_data):
    """Requesting semantic segmentation authors both SemanticSegmentation and SemanticIdMap render vars."""
    render_var_configs = get_render_var_configs(["semantic_segmentation"], render_scope_name="RenderCamera_0")

    assert render_var_configs == [
        ("/RenderCamera_0/Vars/semantic", "semantic", "SemanticSegmentation"),
        ("/RenderCamera_0/Vars/SemanticIdMap", "SemanticIdMap", "SemanticIdMap"),
    ]

    camera_spec.cfg.data_types = ["semantic_segmentation"]
    render_scope = build_render_scope_usd(camera_spec, render_data)

    assert "rel orderedVars = [</RenderCamera_0/Vars/semantic>, </RenderCamera_0/Vars/SemanticIdMap>]" in render_scope
    assert 'uniform string sourceName = "SemanticSegmentation"' in render_scope
    assert 'uniform string sourceName = "SemanticIdMap"' in render_scope


def test_ovrtx_instance_segmentation_authors_pixel_and_map_render_vars(camera_spec, render_data):
    """Requesting instance segmentation authors the pixel AOV plus the three ID/label map render vars."""
    render_var_configs = get_render_var_configs(["instance_segmentation"], render_scope_name="RenderCamera_0")

    assert render_var_configs == [
        (
            "/RenderCamera_0/Vars/NonStableInstanceSegmentation",
            "NonStableInstanceSegmentation",
            "NonStableInstanceSegmentation",
        ),
        ("/RenderCamera_0/Vars/StableIdSemanticIdMap", "StableIdSemanticIdMap", "StableIdSemanticIdMap"),
        ("/RenderCamera_0/Vars/StableIdMap", "StableIdMap", "StableIdMap"),
        ("/RenderCamera_0/Vars/SemanticIdMap", "SemanticIdMap", "SemanticIdMap"),
    ]

    camera_spec.cfg.data_types = ["instance_segmentation"]
    render_scope = build_render_scope_usd(camera_spec, render_data)

    assert (
        "rel orderedVars = [</RenderCamera_0/Vars/NonStableInstanceSegmentation>,"
        " </RenderCamera_0/Vars/StableIdSemanticIdMap>,"
        " </RenderCamera_0/Vars/StableIdMap>, </RenderCamera_0/Vars/SemanticIdMap>]" in render_scope
    )
    assert 'uniform string sourceName = "StableIdSemanticIdMap"' in render_scope
    assert 'uniform string sourceName = "StableIdMap"' in render_scope


def test_ovrtx_semantic_and_instance_segmentation_share_a_single_semantic_id_map():
    """Requesting both segmentation outputs authors ``SemanticIdMap`` exactly once (it is shared)."""
    render_var_configs = get_render_var_configs(
        ["semantic_segmentation", "instance_segmentation"], render_scope_name="RenderCamera_0"
    )

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
        num_envs,
        source_paths=tuple(f"/World/envs/env_{env_idx}" for env_idx in range(num_envs)),
    )

    _assert_export_contains_env_roots_and_children(exported, range(num_envs))


def test_export_stage_full_when_single_env():
    """Single-environment stages are exported without trimming."""
    num_envs = 1
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        num_envs,
        source_paths=("/World/envs/env_0",),
    )

    _assert_export_contains_env_roots_and_children(exported, range(num_envs))


def test_export_stage_homogeneous_keeps_only_env0_prototype():
    """Homogeneous cloning exports only the env_0 prototype subtree."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        num_envs,
        source_paths=("/World/envs/env_0",),
    )

    _assert_export_contains_env_roots_and_children(exported, [0])
    _assert_export_omits_env_children(exported, range(1, num_envs))


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
        num_envs,
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
        num_envs,
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
        num_envs,
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
