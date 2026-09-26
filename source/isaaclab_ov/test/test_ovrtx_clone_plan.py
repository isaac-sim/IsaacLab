# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX clone-plan consumption and OVRTX-side cloning."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ClonePlan, PrototypeWorldTopology, UsdReplicateContext, make_clone_plan
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg, SpawnerCfg

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXCameraRenderData, OVRTXRenderer  # noqa: E402

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402
else:
    OVRTXRenderer = None
    ovrtx_renderer_module = None
    OVRTXRendererCfg = None
    Gf = None
    Sdf = None
    Usd = None
    UsdGeom = None
    UsdShade = None


_PRE_OVRTX_STAGE_FILE = "pre_ovrtx_renderer_stage.usda"
_OVRTX_STAGE_FILE = "ovrtx_renderer_stage.usda"


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


def _patch_simulation_context(monkeypatch: pytest.MonkeyPatch, usd: UsdReplicateContext) -> None:
    mock_ctx = SimpleNamespace(clone_contexts={UsdReplicateContext: usd})
    monkeypatch.setattr(
        "isaaclab_ov.renderers.ovrtx_renderer.SimulationContext",
        SimpleNamespace(instance=lambda: mock_ctx),
    )


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer.backend = SimpleNamespace()
    renderer.backend.renderer = SimpleNamespace(
        add_usd_reference_from_string=lambda *args, **kwargs: 1,
        remove_usd=lambda reference: None,
        clone_usd=lambda *args, **kwargs: None,
        write_array_attribute=lambda *args, **kwargs: None,
        write_attribute=lambda *args, **kwargs: None,
    )
    renderer._usd = None
    renderer._device = "cuda:0"  # __init__'s default, replaced by create_render_data(spec)
    # create_render_data resolves this from the spec; tests that bypass it get the default.
    renderer._warp_device = SimpleNamespace(ordinal=0)
    renderer._camera_prim_path = "/World/envs/env_0/Camera"
    renderer._render_product_paths = []
    renderer._camera_render_data = []
    renderer._next_camera_id = 0
    renderer._exported_usd_string = None
    renderer._initialized_scene = False
    renderer._use_ovstage = False
    renderer._sdp = SimpleNamespace(backend=SimpleNamespace(transform_paths=[]), get_geometry_points=lambda: {})
    renderer._object_scales = None
    renderer._object_scales_by_path = {}
    return renderer


def _make_camera_render_spec(num_envs: int = 1, device: str = "cpu") -> CameraRenderSpec:
    spawn = PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    )
    cfg = CameraCfg(
        height=8,
        width=16,
        prim_path="/World/envs/env_0/Camera",
        spawn=spawn,
        data_types=["rgb"],
    )
    camera_paths = tuple(f"/World/envs/env_{env_idx}/Camera" for env_idx in range(num_envs))
    return CameraRenderSpec(
        cfg=cfg,
        device=device,
        num_instances=num_envs,
        camera_prim_paths=camera_paths,
        view_count=num_envs,
    )


def test_clone_sources_in_ovrtx_uses_world_compositions():
    """Each asset clones directly to the worlds that contain it, excluding its authored source."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._usd = UsdReplicateContext(
        None,
        make_clone_plan(
            tuple(
                AssetBaseCfg(prim_path="/World/envs/env_[^/]+/" + name)
                for name in ("Robot", "Object", "Light", "Robot/Camera")
            ),
            ((3, 0, 2), (3, 0, 1)),
            4,
            weights=(1, 3),
            positions=np.zeros((4, 3), dtype=np.float32),
        ),
    )
    clone_calls: list[tuple[str, list[str]]] = []

    def _clone_usd(source: str, target_paths: list[str]) -> None:
        clone_calls.append((source, target_paths))

    renderer.backend.renderer.clone_usd = _clone_usd

    renderer._clone_sources_in_ovrtx()

    assert clone_calls == [
        (
            "/World/envs/env_0/Robot",
            ["/World/envs/env_1/Robot", "/World/envs/env_2/Robot", "/World/envs/env_3/Robot"],
        ),
        ("/World/envs/env_1/Object", ["/World/envs/env_2/Object", "/World/envs/env_3/Object"]),
    ]


def test_clone_sources_in_ovrtx_writes_plan_positions_after_cloning():
    """Legacy OVRTX cloning writes the authored world origins after copying assets."""
    renderer = _make_ovrtx_renderer_without_backend()
    positions = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5], [-3.0, 4.0, 1.5]], dtype=np.float32)
    renderer._usd = UsdReplicateContext(
        None,
        make_clone_plan((AssetBaseCfg(prim_path="/World/envs/env_[^/]+"),), ((0,),), 3, positions=positions),
    )
    call_order: list[str] = []
    clone_calls: list[tuple[str, list[str]]] = []
    write_calls: list[dict] = []

    def _clone_usd(source: str, target_paths: list[str]) -> None:
        call_order.append("clone")
        clone_calls.append((source, target_paths))

    renderer.backend.renderer.clone_usd = _clone_usd

    def _write_attribute(**kwargs):
        call_order.append("write")
        write_calls.append(kwargs)

    renderer.backend.renderer.write_attribute = _write_attribute

    renderer._clone_sources_in_ovrtx()

    expected = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    expected[:, 3, :3] = positions
    assert call_order == ["clone", "write"]
    assert clone_calls == [("/World/envs/env_0", ["/World/envs/env_1", "/World/envs/env_2"])]
    assert len(write_calls) == 1
    assert write_calls[0]["prim_paths"] == ["/World/envs/env_0", "/World/envs/env_1", "/World/envs/env_2"]
    assert write_calls[0]["attribute_name"] == "omni:xform"
    np.testing.assert_array_equal(write_calls[0]["tensor"], expected)


@pytest.mark.skipif(importlib.util.find_spec("ovstage") is None, reason="requires optional module: ovstage")
def test_clone_sources_ovstage_writes_plan_positions_after_cloning(monkeypatch: pytest.MonkeyPatch):
    """Ovstage copies each active asset and then applies the authored world origins."""
    renderer = _make_ovrtx_renderer_without_backend()
    positions = np.array([[0.0, 0.0, 0.0], [1.5, -2.0, 0.25], [3.0, 4.0, 0.5]], dtype=np.float32)
    renderer._usd = UsdReplicateContext(
        None,
        make_clone_plan(
            tuple(
                AssetBaseCfg(prim_path="/World/envs/env_[^/]+/" + name) for name in ("Robot", "Object", "Object/Camera")
            ),
            ((0,), (2, 1)),
            3,
            weights=(1, 2),
            positions=positions,
        ),
    )
    events: list[tuple[str, str, object]] = []
    xforms: list[np.ndarray] = []
    completion = SimpleNamespace(wait=lambda: None)

    def _clone(source: str, target_paths: list[str], **_kwargs):
        events.append(("clone", source, target_paths))

    def _query(path_list: str) -> str:
        events.append(("query", "envs", path_list))
        return "env_query"

    def _write(_query, attribute_name: str, **kwargs):
        events.append(("write", attribute_name, kwargs["tensors"]))
        return completion

    def _create_paths(paths: list[str]) -> str:
        events.append(("paths", "envs", paths))
        return "env_paths"

    renderer.backend.stage = SimpleNamespace(
        query_from_path_list=_query,
        clone=_clone,
        write_attribute=_write,
        release_query=lambda _query: completion,
    )
    renderer.backend.paths = SimpleNamespace(
        create_path_list_from_strings=_create_paths,
        destroy_path_list=lambda _paths: None,
    )
    renderer._current_ordinal = 3

    def _record_xforms(value: np.ndarray) -> str:
        xforms.append(value.copy())
        return "root_xforms"

    monkeypatch.setattr("isaaclab_ov.renderers.ovrtx_renderer.xform_tensor_from_numpy", _record_xforms)

    renderer._clone_sources_ovstage()

    expected = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    expected[:, 3, :3] = positions
    assert events == [
        ("clone", "/World/envs/env_1/Object", ["/World/envs/env_2/Object"]),
        ("paths", "envs", ["/World/envs/env_0", "/World/envs/env_1", "/World/envs/env_2"]),
        ("query", "envs", "env_paths"),
        ("write", "omni:xform", "root_xforms"),
    ]
    assert len(xforms) == 1
    np.testing.assert_array_equal(xforms[0], expected)


@pytest.mark.parametrize("env_template", ["/World/envs/env_{}", "/World/Instances/World_{}"])
def test_capture_object_scales_populates_source_and_destination_scale_array(env_template):
    """Only declared prototype and shared scales reach the body array, independent of namespace."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, f"{env_template.format(0)}/Object").AddScaleOp().Set(Gf.Vec3d(1, 1, 8))
    UsdGeom.Xform.Define(stage, f"{env_template.format(1)}/Object").AddScaleOp().Set(Gf.Vec3d(1, 1, 4))
    UsdGeom.Xform.Define(stage, "/World/Shared").AddScaleOp().Set(Gf.Vec3d(2, 3, 4))
    UsdGeom.Xform.Define(stage, "/World/envs/Unplanned").AddScaleOp().Set(Gf.Vec3d(5, 6, 7))
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    renderer._usd = UsdReplicateContext(
        stage,
        ClonePlan(
            PrototypeWorldTopology(
                tuple(
                    AssetBaseCfg(
                        prim_path=env_template.format("[^/]+") + "/Object",
                        spawn=SpawnerCfg(spawn_path=env_template.format(i) + "/Object"),
                    )
                    for i in range(2)
                )
                + (AssetBaseCfg(prim_path="/World/Shared"),),
                np.asarray([2, 0, 0, 1]),
                np.asarray([0, 1, 3, 4]),
                np.asarray([0, 1, 0]),
            ),
        ),
        env_template=env_template,
    )

    renderer._capture_object_scales(stage)
    scales = renderer._create_object_scale_array(
        [f"{env_template.format(index)}/Object" for index in range(3)]
        + [f"{env_template.format(index)}/Object_1" for index in (0, 2)]
        + ["/World/Shared"]
    )

    np.testing.assert_allclose(scales.numpy(), [[1, 1, 8], [1, 1, 4], [1, 1, 8], [1, 1, 8], [1, 1, 8], [2, 3, 4]])
    assert "/World/envs/Unplanned" not in renderer._object_scales_by_path


@pytest.mark.parametrize("dump_enabled", [False, True])
def test_prepare_stage_writes_debug_dump_only_when_requested(tmp_path, monkeypatch, dump_enabled):
    """The optional dump preserves the raw stage; default preparation performs no file writes."""
    _patch_simulation_context(
        monkeypatch,
        UsdReplicateContext(
            None,
            make_clone_plan(
                (AssetBaseCfg(prim_path="/World/envs/env_[^/]+"),),
                ((0,),),
                2,
                positions=np.zeros((2, 3), dtype=np.float32),
            ),
        ),
    )

    stage = _make_multi_env_stage(2)
    renderer = _make_ovrtx_renderer_without_backend()
    output_dir = tmp_path / "nested" / "usd"
    renderer.cfg.temp_usd_dir = str(output_dir) if dump_enabled else None
    if not dump_enabled:
        monkeypatch.setattr(ovrtx_renderer_module, "_write_file", Mock(side_effect=AssertionError("Unexpected dump")))
    expected_pre_export = stage.ExportToString()

    renderer.prepare_stage(stage, 2)

    assert output_dir.exists() is dump_enabled
    if dump_enabled:
        assert (output_dir / _PRE_OVRTX_STAGE_FILE).read_text(encoding="utf-8") == expected_pre_export
    assert (output_dir / _OVRTX_STAGE_FILE).exists() is False


def test_initialize_camera_render_data_from_spec_writes_combined_stage_dump(tmp_path: Path):
    """_initialize_camera_render_data_from_spec writes the combined stage when temp_usd_dir is set."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.temp_usd_dir = str(tmp_path)
    scene = _make_multi_env_stage(1)
    scene.SetDefaultPrim(scene.GetPrimAtPath("/World"))
    scene.SetMetadata("metersPerUnit", 1.0)
    scene_usd = scene.GetRootLayer().ExportToString()
    renderer._exported_usd_string = scene_usd

    open_calls: list[str] = []
    renderer.backend.renderer.open_usd_from_string = lambda usd_string: open_calls.append(usd_string)
    reference_calls = []
    renderer.backend.renderer.add_usd_reference_from_string = lambda usd, path: reference_calls.append((usd, path))
    renderer.backend.renderer.bind_attribute = lambda **kwargs: SimpleNamespace(unbind=lambda: None)
    renderer.backend.renderer.write_attribute = lambda **kwargs: None

    spec = _make_camera_render_spec(num_envs=1)
    render_data = OVRTXCameraRenderData(spec, "cpu", render_scope_name="RenderCamera_0")
    renderer._initialize_camera_render_data_from_spec(spec, render_data)

    combined_path = tmp_path / _OVRTX_STAGE_FILE
    combined_text = combined_path.read_text(encoding="utf-8")
    combined_layer = Sdf.Layer.CreateAnonymous("combined.usda")
    assert combined_layer.ImportFromString(combined_text)
    assert combined_layer.defaultPrim == "World"
    assert combined_layer.pseudoRoot.GetInfo("metersPerUnit") == 1.0
    assert combined_layer.GetPrimAtPath(spec.camera_prim_paths[0])
    assert combined_layer.GetPrimAtPath(render_data.render_product_path).typeName == "RenderProduct"
    assert open_calls == [scene_usd]
    reference_text, reference_path = reference_calls[0]
    assert reference_path == "/RenderCamera_0"
    reference_layer = Sdf.Layer.CreateAnonymous("reference.usda")
    assert reference_layer.ImportFromString(reference_text)
    assert reference_layer.defaultPrim == render_data.render_scope_name
    assert reference_layer.GetPrimAtPath(render_data.render_product_path).typeName == "RenderProduct"
    assert renderer._exported_usd_string is None


def test_create_render_data_pins_the_render_product_to_the_spec_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The render product is pinned to the CUDA device whose Warp kernels read its render vars.

    Without this, OVRTX picks the device itself and hands back buffers on ``cuda:0`` while the tile
    extraction kernels launch on the simulation device.
    """
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.temp_usd_dir = str(tmp_path)
    renderer._exported_usd_string = "#usda 1.0\n"

    renderer.backend.renderer.open_usd_from_string = lambda _usd_string: None
    renderer.backend.renderer.bind_attribute = lambda **kwargs: SimpleNamespace(unbind=lambda: None)
    renderer.backend.renderer.write_attribute = lambda **kwargs: None

    class _FakeWarpDevice:
        ordinal = 1

        def __str__(self) -> str:
            return "cuda:1"

    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_device", lambda device: _FakeWarpDevice())
    renderer.create_render_data(_make_camera_render_spec(num_envs=1, device="cuda:1"))

    combined_text = (tmp_path / _OVRTX_STAGE_FILE).read_text(encoding="utf-8")
    assert "uint[] deviceIds = [1]" in combined_text


def test_initialize_camera_render_data_from_spec_refreshes_camera_relationship_after_cloning():
    """Multi-environment initialization rewrites the RenderProduct cameras after cloning."""
    num_envs = 4
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._exported_usd_string = "#usda 1.0\n"

    call_order: list[str] = []
    write_array_calls: list[tuple[list[str], str, list[list[str]]]] = []

    renderer.backend.renderer.open_usd_from_string = lambda _usd_string: call_order.append("open")
    renderer._clone_sources_in_ovrtx = lambda: call_order.append("clone")
    renderer._update_scene_partitions_after_clone = lambda _num_envs: call_order.append("partitions")

    def _write_array_attribute(prim_paths: list[str], attribute_name: str, tensors: list[list[str]]) -> None:
        call_order.append("rewrite_cameras")
        write_array_calls.append((prim_paths, attribute_name, tensors))

    renderer.backend.renderer.write_array_attribute = _write_array_attribute
    renderer.backend.renderer.bind_attribute = lambda **_kwargs: object()
    renderer.backend.renderer.write_attribute = lambda **_kwargs: None
    renderer._setup_xform_bindings_legacy = lambda: None
    renderer._setup_geometry_bindings_legacy = lambda: None

    spec = _make_camera_render_spec(num_envs=num_envs)
    render_data = OVRTXCameraRenderData(spec, "cpu", render_scope_name="RenderCamera_0")
    renderer._initialize_camera_render_data_from_spec(spec, render_data)

    assert call_order == ["open", "clone", "partitions", "rewrite_cameras"]
    assert write_array_calls == [
        (
            [render_data.render_product_path],
            "camera",
            [[f"/World/envs/env_{env_id}/Camera" for env_id in range(num_envs)]],
        )
    ]


@pytest.mark.parametrize("suffix", ["", "/Robot"])
def test_prepare_stage_exports_only_clone_sources_and_their_materials(monkeypatch, suffix):
    """Keep prototype contents and material bindings, retaining only needed destination ancestors."""
    stage = _make_multi_env_stage(3)
    source = f"/World/envs/env_0{suffix}"
    material = UsdShade.Material.Define(stage, f"{source}/warm")
    body = UsdGeom.Xform.Define(stage, f"{source}/Body").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(body)
    UsdShade.MaterialBindingAPI(body).Bind(material)
    plan = make_clone_plan(
        (AssetBaseCfg(prim_path=f"/World/envs/env_{{}}{suffix}".format("[^/]+"), spawn=SpawnerCfg(spawn_path=source)),),
        ((0,),),
        3,
        positions=np.zeros((3, 3), dtype=np.float32),
    )
    _patch_simulation_context(monkeypatch, UsdReplicateContext(stage, plan))
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.prepare_stage(stage, 3)
    exported = Usd.Stage.CreateInMemory()
    assert exported.GetRootLayer().ImportFromString(renderer._exported_usd_string)
    binding = UsdShade.MaterialBindingAPI(exported.GetPrimAtPath(f"{source}/Body")).GetDirectBindingRel()
    assert binding.GetTargets() == [Sdf.Path(f"{source}/warm")]
    assert exported.GetPrimAtPath(f"{source}/warm")
    assert exported.GetPrimAtPath("/World/envs/env_0/Robot")
    assert bool(exported.GetPrimAtPath("/World/envs/env_0/Camera")) is (not suffix)
    for env_id in (1, 2):
        root = f"/World/envs/env_{env_id}"
        assert bool(exported.GetPrimAtPath(root)) is bool(suffix)
        assert not exported.GetPrimAtPath(f"{root}/Robot")
        assert not exported.GetPrimAtPath(f"{root}/Object_env{env_id}_only")
