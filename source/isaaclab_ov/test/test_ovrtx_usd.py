# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVRTX USD render product authoring and stage export."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab_ov", reason="requires optional isaaclab_ov package")
pytest.importorskip("pxr", reason="requires optional USD bindings")

pytestmark = pytest.mark.ov

from isaaclab_ov.renderers import (  # noqa: E402
    RtxMinimalShadingMode,
    RtxRenderMode,
    apply_minimal_shading_mode,
    apply_render_mode,
)
from isaaclab_ov.renderers.ovrtx_usd import (  # noqa: E402
    build_render_product_as_string,
    get_render_var_configs,
    render_var_prim_names_by_source,
    render_var_prim_paths_by_source,
)
from isaaclab_ov.stage import export_stage_to_string  # noqa: E402

from pxr import Gf, Sdf, Usd, UsdGeom, UsdRender  # noqa: E402

from isaaclab.renderers.camera_render_spec import CameraRenderSpec  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402


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
    )


@pytest.fixture
def render_data(camera_spec):
    return SimpleNamespace(
        render_scope_name="RenderCamera_0",
        render_product_path="/RenderCamera_0/RenderProduct",
        camera_paths=list(camera_spec.camera_prim_paths),
        width=16,
        height=8,
        num_cols=2,
        num_rows=2,
    )


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
    render_scope = build_render_product_as_string(camera_spec, render_data, device_id=0)
    assert 'token omni:rtx:background:source:type = "domeLight"' in render_scope
    assert "omni:rtx:background:source:color" not in render_scope
    assert 'token omni:rtx:rendermode = "RealTimePathTracing"' in render_scope
    assert "omni:rtx:minimal:" not in render_scope


def test_render_product_solid_background_color(camera_spec, render_data):
    """Providing background_color emits color source type and the color attribute."""
    camera_spec.cfg.background_color = (1.0, 0.0, 0.5)
    render_scope = build_render_product_as_string(camera_spec, render_data, device_id=0)
    assert 'token omni:rtx:background:source:type = "color"' in render_scope
    assert "color3f omni:rtx:background:source:color = (1.0, 0.0, 0.5)" in render_scope
    assert 'token omni:rtx:background:source:type = "domeLight"' not in render_scope


def test_render_var_prim_names_are_read_only():
    with pytest.raises(TypeError):
        render_var_prim_names_by_source()["LdrColor"] = "mutated"  # type: ignore[index]


def test_render_var_prim_paths_cover_every_authored_render_var(camera_spec, render_data):
    """Every output and metadata path resolves to its authored prim under the requested scope."""
    # A non-default scope name also covers the default one: only the path prefix changes.
    render_scope_name = "RenderCamera_1"
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
    render_data.render_scope_name = render_scope_name
    render_data.render_product_path = f"/{render_scope_name}/RenderProduct"
    render_scope = build_render_product_as_string(camera_spec, render_data, device_id=0)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(render_scope)
    stage = Usd.Stage.Open(layer)
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path(f"/{render_scope_name}")
    product = stage.GetPrimAtPath(f"/{render_scope_name}/RenderProduct")
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


def test_ovrtx_primary_render_var_follows_the_first_requested_data_type():
    """The primary render var follows the order of the requested outputs."""
    assert get_render_var_configs(["rgb", "motion_vectors"], render_scope_name="RenderCamera_0")[0] == (
        "/RenderCamera_0/Vars/LdrColor",
        "LdrColor",
        "LdrColor",
    )
    assert get_render_var_configs(["motion_vectors", "rgb"], render_scope_name="RenderCamera_0")[0] == (
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


@pytest.mark.parametrize("data_types", [[], ["instance_id_segmentation_fast"], ["normals", "unknown"]])
def test_ovrtx_rejects_missing_or_unsupported_outputs(data_types):
    """Invalid requests fail instead of silently rendering RGB or dropping an output."""
    with pytest.raises(ValueError, match="output"):
        get_render_var_configs(data_types, render_scope_name="RenderCamera_0")


@pytest.mark.parametrize(
    ("data_types", "mode", "level", "message"),
    [
        (
            ["simple_shading_constant_diffuse", "simple_shading_full_mdl"],
            RtxRenderMode.MINIMAL,
            RtxMinimalShadingMode.CONSTANT_DIFFUSE,
            "at most one",
        ),
        (["rgb_hdr"], RtxRenderMode.MINIMAL, RtxMinimalShadingMode.FULL_MDL, "does not support 'rgb_hdr'"),
        (
            ["rgb_hdr", "simple_shading_full_mdl"],
            RtxRenderMode.REAL_TIME_PATH_TRACING,
            RtxMinimalShadingMode.FULL_MDL,
            "does not support 'rgb_hdr'",
        ),
        (["rgb"], "Unknown", RtxMinimalShadingMode.FULL_MDL, "Unsupported render mode"),
        (["depth"], RtxRenderMode.MINIMAL, "no_color", "Minimal shading mode must be"),
        (["rgb"], RtxRenderMode.MINIMAL, 0, "Unsupported Minimal shading mode"),
    ],
)
def test_ovrtx_rejects_incompatible_mode_outputs(camera_spec, render_data, data_types, mode, level, message):
    """Invalid mode/output requests fail at product creation rather than silently selecting a mode."""
    camera_spec.cfg.data_types = data_types
    camera_spec.cfg.render_settings = {
        "omni:rtx:rendermode": ("Token", mode),
        "omni:rtx:minimal:mode": ("Int", level),
    }
    with pytest.raises(ValueError, match=message):
        build_render_product_as_string(camera_spec, render_data, device_id=0)


@pytest.mark.parametrize(
    ("data_type", "shading_mode", "minimal_mode", "enable_shadows"),
    [
        (
            "simple_shading_constant_diffuse",
            RtxMinimalShadingMode.FULL_MDL,
            RtxMinimalShadingMode.CONSTANT_DIFFUSE,
            False,
        ),
        ("simple_shading_diffuse_mdl", RtxMinimalShadingMode.FULL_MDL, RtxMinimalShadingMode.DIFFUSE_MDL, True),
        ("simple_shading_full_mdl", RtxMinimalShadingMode.CONSTANT_DIFFUSE, RtxMinimalShadingMode.FULL_MDL, False),
    ],
)
def test_ovrtx_simple_shading_alone_uses_ldr_color(
    camera_spec, render_data, data_type, shading_mode, minimal_mode, enable_shadows
):
    """Simple shading overrides configured execution and shading modes, retaining the configured shadows."""
    assert get_render_var_configs([data_type], render_scope_name="RenderCamera_0") == [
        ("/RenderCamera_0/Vars/LdrColor", "LdrColor", "LdrColor")
    ]
    camera_spec.cfg.data_types = [data_type]
    apply_render_mode(camera_spec.cfg, RtxRenderMode.REAL_TIME_PATH_TRACING)
    apply_minimal_shading_mode(camera_spec.cfg, shading_mode)
    usd = build_render_product_as_string(camera_spec, render_data, device_id=0, enable_shadows=enable_shadows)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(usd)
    assert layer.GetAttributeAtPath("/RenderCamera_0/RenderProduct.omni:rtx:rendermode").default == "Minimal"
    assert layer.GetAttributeAtPath("/RenderCamera_0/RenderProduct.omni:rtx:minimal:mode").default == minimal_mode
    assert layer.GetAttributeAtPath("/RenderCamera_0/RenderProduct.omni:rtx:rt:ecoMode:enabled").default is False
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
    render_product = build_render_product_as_string(camera_spec, render_data, device_id=0)

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


@pytest.mark.parametrize("data_types", [["rgb"], ["rgb", "rgb_hdr"], []])
def test_render_product_isp_requests_hdr_without_mutating_camera_outputs(camera_spec, render_data, data_types):
    """ISP receives one HDR source while the camera's requested outputs remain unchanged."""
    camera_spec.cfg.data_types = data_types.copy()
    camera_spec.cfg.isp_cfg = object()
    render_product = build_render_product_as_string(camera_spec, render_data, device_id=0)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(render_product)
    ordered_vars = layer.GetRelationshipAtPath("/RenderCamera_0/RenderProduct.orderedVars")
    expected_vars = ("LdrColor", "HdrColor") if data_types else ("HdrColor",)
    assert list(ordered_vars.targetPathList.explicitItems) == [
        Sdf.Path(f"/RenderCamera_0/Vars/{name}") for name in expected_vars
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
    render_scope = build_render_product_as_string(camera_spec, render_data, device_id=0)

    assert "rel orderedVars = [</RenderCamera_0/Vars/LdrColor>, </RenderCamera_0/Vars/HdrColor>]" in render_scope
    assert 'def RenderVar "LdrColor"' in render_scope
    assert 'def RenderVar "HdrColor"' in render_scope


def test_ovrtx_semantic_segmentation_authors_semantic_and_id_map_render_vars(camera_spec, render_data):
    """Requesting semantic segmentation authors both SemanticSegmentation and SemanticIdMap render vars."""
    render_var_configs = get_render_var_configs(["semantic_segmentation"], render_scope_name="RenderCamera_0")

    assert render_var_configs == [
        ("/RenderCamera_0/Vars/semantic", "semantic", "SemanticSegmentation"),
        ("/RenderCamera_0/Vars/SemanticIdMap", "SemanticIdMap", "SemanticIdMap"),
    ]

    camera_spec.cfg.data_types = ["semantic_segmentation"]
    render_scope = build_render_product_as_string(camera_spec, render_data, device_id=0)

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
    render_scope = build_render_product_as_string(camera_spec, render_data, device_id=0)

    assert (
        "rel orderedVars = [</RenderCamera_0/Vars/NonStableInstanceSegmentation>,"
        " </RenderCamera_0/Vars/StableIdSemanticIdMap>,"
        " </RenderCamera_0/Vars/StableIdMap>, </RenderCamera_0/Vars/SemanticIdMap>]" in render_scope
    )
    assert 'uniform string sourceName = "StableIdSemanticIdMap"' in render_scope
    assert 'uniform string sourceName = "StableIdMap"' in render_scope


def test_export_stage_keeps_all_env_content_when_all_roots_are_sources():
    """Listing every env root as a source preserves the full stage content."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        tuple(f"/World/envs/env_{i}" for i in range(num_envs)),
        source_paths=tuple(f"/World/envs/env_{env_idx}" for env_idx in range(num_envs)),
    )

    _assert_export_contains_env_roots_and_children(exported, range(num_envs))


@pytest.mark.parametrize("num_envs", [1, 4])
def test_export_stage_without_clone_destinations(num_envs):
    """Single environments and global-only scenes export without losing content."""
    stage = _make_multi_env_stage(1) if num_envs == 1 else Usd.Stage.CreateInMemory()
    UsdGeom.Sphere.Define(stage, "/World/Shared")
    env_paths = tuple(f"/World/envs/env_{i}" for i in range(num_envs))

    exported = export_stage_to_string(
        stage,
        tuple(f"/World/envs/env_{i}" for i in range(num_envs)),
        source_paths=("/World/envs/env_0",) if num_envs == 1 else ("/World/Shared",),
        env_paths=env_paths,
    )

    if num_envs == 1:
        _assert_export_contains_env_roots_and_children(exported, range(1))
    result = Usd.Stage.CreateInMemory()
    result.GetRootLayer().ImportFromString(exported)
    assert result.GetPrimAtPath("/World/Shared").IsA(UsdGeom.Sphere)
    assert all(result.GetPrimAtPath(path).IsDefined() for path in env_paths)
    assert all(result.GetPrimAtPath(path).IsA(UsdGeom.Xform) for path in env_paths)


def test_export_stage_homogeneous_keeps_only_env0_prototype():
    """Homogeneous cloning exports only the env_0 prototype subtree."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        tuple(f"/World/envs/env_{i}" for i in range(num_envs)),
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
        tuple(f"/World/envs/env_{i}" for i in range(num_envs)),
        source_paths=("/World/envs/env_0",),
        keep_env_roots=False,
    )

    _assert_export_contains_env_roots_and_children(exported, [0])
    for env_idx in range(1, num_envs):
        assert f'def Xform "env_{env_idx}"' not in exported
        assert f'def Xform "Object_env{env_idx}_only"' not in exported


def test_export_stage_keeps_file_references_without_exporting_unrouted_copies(tmp_path):
    """The native export retains a clean asset reference and excludes clones it will recreate."""
    asset_path = tmp_path / "asset.usda"
    asset = Usd.Stage.CreateNew(str(asset_path))
    root = UsdGeom.Xform.Define(asset, "/Asset")
    UsdGeom.Mesh.Define(asset, "/Asset/Geometry")
    asset.SetDefaultPrim(root.GetPrim())
    asset.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    UsdGeom.Xform.Define(stage, "/World/envs")
    for env_idx in range(2):
        env = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_idx}")
        env.GetPrim().GetReferences().AddReference("./asset.usda")
    stage.GetRootLayer().Save()

    exported = export_stage_to_string(
        stage,
        tuple(f"/World/envs/env_{i}" for i in range(2)),
        source_paths=("/World/envs/env_0",),
        keep_env_roots=False,
        preserve_references=True,
    )
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(exported)
    result = Usd.Stage.Open(layer)
    assert result.GetPrimAtPath("/World/envs/env_0/Geometry").IsA(UsdGeom.Mesh)
    assert result.GetPrimAtPath("/World/envs/env_0").HasAuthoredReferences()
    assert not result.GetPrimAtPath("/World/envs/env_1")
    assert layer.GetPrimAtPath("/World/envs/env_0/Geometry") is None

    asset.GetPrimAtPath("/Asset/Geometry").CreateAttribute("unsavedValue", Sdf.ValueTypeNames.Int).Set(7)
    dirty_layer = Sdf.Layer.CreateAnonymous(".usda")
    assert dirty_layer.ImportFromString(
        export_stage_to_string(
            stage,
            tuple(f"/World/envs/env_{i}" for i in range(2)),
            source_paths=("/World/envs/env_0",),
            keep_env_roots=False,
            preserve_references=True,
        )
    )
    dirty_result = Usd.Stage.Open(dirty_layer)
    geometry = dirty_result.GetPrimAtPath("/World/envs/env_0/Geometry")
    assert geometry.GetAttribute("unsavedValue").Get() == 7
    assert not dirty_result.GetPrimAtPath("/World/envs/env_0").HasAuthoredReferences()


def test_export_stage_heterogeneous_keeps_multiple_sources():
    """Heterogeneous source paths export only prototype env subtrees."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)

    exported = export_stage_to_string(
        stage,
        tuple(f"/World/envs/env_{i}" for i in range(num_envs)),
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


@pytest.mark.parametrize("settings_path", [None, "/Render/AuthoredSettings"])
def test_export_stage_preserves_authored_scene(settings_path):
    """Renderer settings and clone trimming are confined to the exported scene."""
    num_envs = 4
    stage = _make_multi_env_stage(num_envs)
    if settings_path is not None:
        settings = UsdRender.Settings.Define(stage, settings_path)
        settings.CreateResolutionAttr(Gf.Vec2i(1280, 720))
        stage.SetMetadata("renderSettingsPrimPath", settings_path)
    before = stage.ExportToString()
    layer = Sdf.Layer.CreateAnonymous()
    layer.ImportFromString(
        export_stage_to_string(
            stage, tuple(f"/World/envs/env_{i}" for i in range(num_envs)), source_paths=("/World/envs/env_0",)
        )
    )
    exported = Usd.Stage.Open(layer)
    assert exported.HasAuthoredMetadata("renderSettingsPrimPath")
    settings = exported.GetPrimAtPath(exported.GetMetadata("renderSettingsPrimPath"))
    show_all = settings.GetAttribute("omni:rtx:scenePartitioning:showAllPartitionsByDefault")
    assert show_all and show_all.Get() is True
    if settings_path is not None:
        assert settings.GetPath() == settings_path
        assert settings.GetAttribute("resolution").Get() == (1280, 720)
    assert stage.ExportToString() == before
    assert not exported.GetPrimAtPath("/World/envs/env_1/Robot")


@pytest.mark.parametrize(
    ("data_types", "shading_mode", "native_mode"),
    [
        (["rgb"], RtxMinimalShadingMode.DIFFUSE_MDL, 2),
        (["depth"], None, 3),
        (["rgb", "simple_shading_diffuse_mdl"], RtxMinimalShadingMode.FULL_MDL, 2),
    ],
)
def test_explicit_minimal_is_independent_of_outputs_and_other_products(
    camera_spec, render_data, data_types, shading_mode, native_mode
):
    """Ordinary outputs retain settings, shortcuts override shading, and another product stays isolated."""
    camera_spec.cfg.data_types = data_types
    renderer_before = camera_spec.cfg.renderer_cfg.to_dict()
    camera_spec.cfg.render_settings = {"omni:rtx:quality": ("Int", 0)}
    if shading_mode is not None:
        apply_minimal_shading_mode(camera_spec.cfg, shading_mode)
    assert "omni:rtx:rendermode" not in camera_spec.cfg.render_settings
    apply_render_mode(camera_spec.cfg, RtxRenderMode.MINIMAL)
    assert camera_spec.cfg.renderer_cfg.to_dict() == renderer_before
    minimal = build_render_product_as_string(camera_spec, render_data, device_id=0)
    camera_spec.cfg.data_types = ["rgb"]
    render_data.render_scope_name = "ColorCamera"
    render_data.render_product_path = "/ColorCamera/RenderProduct"
    apply_render_mode(camera_spec.cfg, RtxRenderMode.REAL_TIME_PATH_TRACING)
    color = build_render_product_as_string(camera_spec, render_data, device_id=0)
    stage = Usd.Stage.CreateInMemory()
    layers = []
    for usd in (minimal, color):
        layer = Sdf.Layer.CreateAnonymous(".usda")
        assert layer.ImportFromString(usd)
        stage.GetRootLayer().subLayerPaths.append(layer.identifier)
        layers.append(layer)  # Retain anonymous layers through composition.
    product = stage.GetPrimAtPath("/RenderCamera_0/RenderProduct")
    assert product.GetAttribute("omni:rtx:rendermode").Get() == "Minimal"
    assert product.GetAttribute("omni:rtx:minimal:mode").Get() == native_mode
    assert product.GetAttribute("omni:rtx:quality").Get() == 0
    color_product = stage.GetPrimAtPath("/ColorCamera/RenderProduct")
    assert color_product.GetAttribute("omni:rtx:rendermode").Get() == "RealTimePathTracing"
    assert not color_product.HasAttribute("omni:rtx:minimal:mode")


@pytest.mark.parametrize(
    ("helper", "value"),
    [
        (apply_render_mode, "Minimal"),
        (apply_minimal_shading_mode, 2),
    ],
)
def test_mode_helpers_require_enum_members(camera_spec, helper, value):
    """Typed helpers reject raw inputs before modifying camera settings."""
    before = camera_spec.cfg.render_settings.copy()
    with pytest.raises(ValueError, match="must be a"):
        helper(camera_spec.cfg, value)
    assert camera_spec.cfg.render_settings == before


@pytest.mark.parametrize(
    ("name", "setting", "message"),
    [
        ("omni:rtx:rendermode", ("String", "Minimal"), "requires USD type 'Token'"),
        ("omni:rtx:minimal:mode", ("Float", 2), "requires USD type 'Int'"),
    ],
)
def test_raw_mode_settings_require_native_usd_types(camera_spec, render_data, name, setting, message):
    camera_spec.cfg.render_settings[name] = setting
    with pytest.raises(ValueError, match=message):
        build_render_product_as_string(camera_spec, render_data, device_id=0)


def test_explicit_minimal_rejects_implicit_isp_hdr(camera_spec, render_data):
    """An ISP request must be validated against its implicit HDR input, even with RGB-only outputs."""
    camera_spec.cfg.isp_cfg = object()
    apply_render_mode(camera_spec.cfg, RtxRenderMode.MINIMAL)
    with pytest.raises(ValueError, match="ISP, which requires HDR"):
        build_render_product_as_string(camera_spec, render_data, device_id=0)
