# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX clone-plan consumption and OVRTX-side cloning."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ClonePlan, PrototypeWorldTopology, make_clone_plan
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
    from isaaclab_ov.cloner import (  # noqa: E402
        OvrtxReplicateContext,
        OvstageReplicateContext,
        ovrtx_replicate,
        ovstage_replicate,
    )
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer  # noqa: E402

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


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer.backend = SimpleNamespace(
        clone_copies=[], clone_env_paths=[], population_env_paths=[], clone_positions=None
    )
    renderer.scene = renderer.backend
    renderer.backend.renderer = SimpleNamespace(
        add_usd_reference_from_string=lambda *args, **kwargs: 1,
        remove_usd=lambda reference: None,
        clone_usd=lambda *args, **kwargs: None,
        write_array_attribute=lambda *args, **kwargs: None,
        write_attribute=lambda *args, **kwargs: None,
    )
    renderer._device = "cuda:0"  # __init__'s default, replaced by create_render_data(spec)
    # create_render_data resolves this from the spec; tests that bypass it get the default.
    renderer._warp_device = SimpleNamespace(ordinal=0)
    renderer._camera_render_data = []
    renderer.scene.next_camera_id = 0
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
        cfg=cfg, device=device, num_instances=num_envs, camera_prim_paths=camera_paths, view_count=num_envs
    )


def _prepare_clones(renderer, plan, routed=None):
    from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXBackendCfg

    from isaaclab.utils import string_to_callable

    cfg = OVRTXBackendCfg(scene_key=renderer.cfg, use_ovstage=renderer._use_ovstage, read_gpu_transforms=True)
    if renderer._use_ovstage:
        from isaaclab_ov.stage import OvstageBackendCfg

        cfg = OvstageBackendCfg(scene_key=cfg)
    sim = SimpleNamespace(_backend_registry=[(cfg, renderer.scene)])
    routed = tuple(range(len(plan.asset_cfgs))) if routed is None else routed
    for context in renderer.cfg.cloning_contexts:
        string_to_callable(context)(sim).replicate(plan, routed)


@pytest.mark.parametrize(
    "use_ovstage, names, worlds, weights, copies",
    [
        (False, ("",), ((0,),), (1,), [(0, "", [1, 2, 3])]),
        (
            True,
            ("/Robot", "/Object", "/Light", "/Robot/Camera"),
            ((3, 0, 2), (3, 0, 1)),
            (1, 3),
            [(0, "/Robot", [1, 2, 3]), (1, "/Object", [2, 3])],
        ),
    ],
    ids=["environment-roots", "nested-assets"],
)
def test_native_replication_applies_compositions_then_positions(
    monkeypatch, use_ovstage, names, worlds, weights, copies
):
    """Both native APIs consume the same prepared copies without consulting the clone plan."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = use_ovstage
    positions = np.array([[0, 0, 0], [2, -1, 0.5], [-3, 4, 1.5], [5, 2, 0]], dtype=np.float32)
    assets = tuple(AssetBaseCfg(prim_path="/World/envs/env_[^/]+" + name) for name in names)
    plan = make_clone_plan(assets, worlds, 4, weights=weights, positions=positions)
    _prepare_clones(renderer, plan)
    native = Mock()
    backend = renderer.backend
    if use_ovstage:
        paths = Mock()
        paths.create_path_list_from_strings.return_value = "env_paths"
        native.query_from_path_list.return_value = contextlib.nullcontext("env_query")
        monkeypatch.setattr("isaaclab_ov.stage.xform_tensor_from_numpy", lambda value: value)
        ovstage_replicate(
            native, paths, backend.clone_copies, backend.clone_env_paths, backend.clone_positions, ordinal=3
        )
        paths.create_path_list_from_strings.assert_called_once_with(backend.clone_env_paths)
        paths.destroy_path_list.assert_called_once_with("env_paths")
        actual = native.write_attribute.call_args.kwargs["tensors"]
        assert native.write_attribute.call_args.args == ("env_query", "omni:xform")
        assert native.write_attribute.call_args.kwargs["ordinal"] == 3
        clone_calls = native.clone.call_args_list
    else:
        ovrtx_replicate(native, backend.clone_copies, backend.clone_env_paths, backend.clone_positions)
        actual_paths, attribute, actual = native.write_attribute.call_args.args
        assert actual_paths == backend.clone_env_paths
        assert attribute == "omni:xform"
        clone_calls = native.clone_usd.call_args_list
    assert clone_calls == [
        call(
            f"/World/envs/env_{world}{suffix}",
            [f"/World/envs/env_{target}{suffix}" for target in targets],
            **({"ordinal": 3} if use_ovstage else {}),
        )
        for world, suffix, targets in copies
    ]
    calls = [method[0] for method in native.method_calls if method[0] in {"clone", "clone_usd", "write_attribute"}]
    assert calls == ["clone" if use_ovstage else "clone_usd"] * len(copies) + ["write_attribute"]
    assert backend.clone_env_paths == [f"/World/envs/env_{world}" for world in range(4)]
    expected = np.tile(np.eye(4, dtype=np.float64), (4, 1, 1))
    expected[:, 3, :3] = positions
    np.testing.assert_array_equal(actual, expected)


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
    plan = ClonePlan(
        PrototypeWorldTopology(3, np.asarray([2, 0, 0, 1]), np.asarray([0, 1, 3, 4]), np.asarray([0, 1, 0])),
        asset_cfgs=tuple(
            AssetBaseCfg(
                prim_path=env_template.format("[^/]+") + "/Object",
                spawn=SpawnerCfg(spawn_path=env_template.format(i) + "/Object"),
            )
            for i in range(2)
        )
        + (AssetBaseCfg(prim_path="/World/Shared"),),
        env_template=env_template,
    )

    _prepare_clones(renderer, plan)
    renderer._capture_object_scales(stage)
    renderer._sdp.backend.transform_paths = (
        [f"{env_template.format(index)}/Object" for index in range(3)]
        + [f"{env_template.format(index)}/Object_1" for index in (0, 2)]
        + ["/World/Shared"]
    )
    renderer.backend.renderer.bind_attribute = Mock()
    renderer._setup_xform_bindings()

    np.testing.assert_allclose(
        renderer._object_scales.numpy(), [[1, 1, 8], [1, 1, 4], [1, 1, 8], [1, 1, 8], [1, 1, 8], [2, 3, 4]]
    )
    assert "/World/envs/Unplanned" not in renderer._object_scales_by_path


@pytest.mark.parametrize("dump_enabled", [False, True])
def test_prepare_stage_writes_debug_dump_only_when_requested(tmp_path, monkeypatch, dump_enabled):
    """The optional dump preserves the raw stage; default preparation performs no file writes."""
    assets = (AssetBaseCfg(prim_path="/World/envs/env_[^/]+"),)
    plan = make_clone_plan(assets, ((0,),), 2, positions=np.zeros((2, 3), dtype=np.float32))

    stage = _make_multi_env_stage(2)
    renderer = _make_ovrtx_renderer_without_backend()
    output_dir = tmp_path / "nested" / "usd"
    renderer.cfg.temp_usd_dir = str(output_dir) if dump_enabled else None
    if not dump_enabled:
        monkeypatch.setattr(ovrtx_renderer_module, "_write_file", Mock(side_effect=AssertionError("Unexpected dump")))
    expected_pre_export = stage.ExportToString()

    _prepare_clones(renderer, plan)
    renderer.prepare_stage(stage, 2)

    assert output_dir.exists() is dump_enabled
    if dump_enabled:
        assert (output_dir / _PRE_OVRTX_STAGE_FILE).read_text(encoding="utf-8") == expected_pre_export
    assert (output_dir / _OVRTX_STAGE_FILE).exists() is False


def test_create_render_data_writes_combined_stage_dump(tmp_path: Path, monkeypatch):
    """Camera registration preserves scene metadata and includes its scoped product in the debug dump."""
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
    renderer.backend.renderer.write_attribute = lambda *args, **kwargs: None

    spec = _make_camera_render_spec(num_envs=1)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_device", lambda _: SimpleNamespace(ordinal=0))
    monkeypatch.setattr(ovrtx_renderer_module.wp, "empty", Mock())
    render_data = renderer.create_render_data(spec)

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
    renderer.backend.renderer.write_attribute = lambda *args, **kwargs: None

    class _FakeWarpDevice:
        ordinal = 1

        def __str__(self) -> str:
            return "cuda:1"

    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_device", lambda device: _FakeWarpDevice())
    monkeypatch.setattr(ovrtx_renderer_module.wp, "empty", Mock())
    renderer.create_render_data(_make_camera_render_spec(num_envs=1, device="cuda:1"))

    combined_text = (tmp_path / _OVRTX_STAGE_FILE).read_text(encoding="utf-8")
    assert "uint[] deviceIds = [1]" in combined_text


@pytest.mark.parametrize("suffix", ["", "/Robot"])
def test_prepare_stage_exports_only_clone_sources_and_their_materials(monkeypatch, suffix):
    """Export routed prototypes and materials, excluding unrouted sources and cloned descendants."""
    stage = _make_multi_env_stage(3)
    stage.GetPrimAtPath("/World/envs/env_0").ClearTypeName()
    stage.RemovePrim("/World/envs/env_2")
    source = f"/World/envs/env_0{suffix}"
    material = UsdShade.Material.Define(stage, f"{source}/warm")
    body = UsdGeom.Xform.Define(stage, f"{source}/Body").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(body)
    UsdShade.MaterialBindingAPI(body).Bind(material)
    asset = AssetBaseCfg(prim_path=f"/World/envs/env_[^/]+{suffix}", spawn=SpawnerCfg(spawn_path=source))
    excluded_path = "/World/envs/env_1/Excluded"
    UsdGeom.Xform.Define(stage, excluded_path)
    excluded = AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Excluded", spawn=SpawnerCfg(spawn_path=excluded_path))
    plan = make_clone_plan((asset, excluded), ((0, 1),), 3, positions=np.zeros((3, 3), dtype=np.float32))
    renderer = _make_ovrtx_renderer_without_backend()
    _prepare_clones(renderer, plan, routed=(0,))
    renderer.prepare_stage(stage, 3)
    if suffix:
        import ovstage
        from isaaclab_ov.stage import OvstageBackend, OvstageBackendCfg

        native, paths = Mock(), Mock()
        native.query_from_path_list.return_value = contextlib.nullcontext("env_query")
        monkeypatch.setattr("isaaclab_ov.stage.create_ovstage", lambda _: contextlib.nullcontext(native))
        monkeypatch.setattr(ovstage, "PathDictionary", lambda _: contextlib.nullcontext(paths))
        imported = Mock()
        monkeypatch.setattr(ovstage.population, "open_usd_from_string", imported)
        cfg = OvstageBackendCfg(scene_key=renderer.cfg, population_domains=ovstage.PopulationDomain.ALL)
        backend = OvstageBackend(cfg)
        OvstageReplicateContext(SimpleNamespace(_backend_registry=[(cfg, backend)])).replicate(plan, (0,))
        backend.populate(renderer._exported_usd_string)
        assert imported.call_args.args == (native, renderer._exported_usd_string)
        assert imported.call_args.kwargs["domains"] == ovstage.PopulationDomain.ALL
        native.clone.assert_called_once_with(source, [f"/World/envs/env_{i}{suffix}" for i in (1, 2)], ordinal=1)
        native.advance_write_floor.assert_called_once_with(ordinal=1)
        backend.close()
    exported = Usd.Stage.CreateInMemory()
    assert exported.GetRootLayer().ImportFromString(renderer._exported_usd_string)
    binding = UsdShade.MaterialBindingAPI(exported.GetPrimAtPath(f"{source}/Body")).GetDirectBindingRel()
    assert binding.GetTargets() == [Sdf.Path(f"{source}/warm")]
    assert exported.GetPrimAtPath(f"{source}/warm")
    assert exported.GetPrimAtPath("/World/envs/env_0").IsA(UsdGeom.Xform)
    assert not exported.GetPrimAtPath(excluded_path)
    assert exported.GetPrimAtPath("/World/envs/env_0/Robot")
    assert bool(exported.GetPrimAtPath("/World/envs/env_0/Camera")) is (not suffix)
    for env_id in (1, 2):
        root = f"/World/envs/env_{env_id}"
        assert bool(exported.GetPrimAtPath(root)) is bool(suffix)
        if suffix:
            assert exported.GetPrimAtPath(root).IsA(UsdGeom.Xform)
        assert not exported.GetPrimAtPath(f"{root}/Robot")
        assert not exported.GetPrimAtPath(f"{root}/Object_env{env_id}_only")


def test_native_cloners_keep_plan_interpretation_in_the_context():
    """Contexts own preparation; consumers and native execution never reinterpret plans or routing ids."""
    from isaaclab_ov.cloner import replicate as replication

    for name in ("_iter_clone_copies", "_OvRenderReplicateContext", "OvRenderReplicateContext"):
        assert not hasattr(replication, name)
    assert not hasattr(OVRTXRenderer, "_clone_sources")
    for name in ("_initialize_camera_render_data_from_spec", "_init_fields_legacy", "_init_fields_ovstage"):
        assert not hasattr(OVRTXRenderer, name)
    for name in (
        "_create_object_scale_array",
        "_update_camera_legacy",
        "_update_camera_ovstage",
        "_render_legacy",
        "_render_ovstage",
    ):
        assert not hasattr(OVRTXRenderer, name)
    renderer_tree = ast.parse(Path(ovrtx_renderer_module.__file__).read_text())
    assert not any(isinstance(node, ast.Name) and node.id == "ClonePlan" for node in ast.walk(renderer_tree))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "_render_product_paths" for node in ast.walk(renderer_tree)
    )
    tree = ast.parse(Path(replication.__file__).read_text())
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name in {"ovrtx_replicate", "ovstage_replicate"}:
            assert not any(isinstance(node, ast.Name) and node.id in {"plan", "cloner"} for node in ast.walk(function))
        if isinstance(function, ast.ClassDef) and function.name == "OvstageReplicateContext":
            assert not any(
                isinstance(node, ast.ImportFrom) and node.module.startswith("isaaclab_ov.renderers")
                for node in ast.walk(function)
            )


@pytest.mark.parametrize(
    ("child_source", "routed", "expected"),
    [
        (
            "/Sources/Robot/Camera",
            (0, 1),
            [
                ("/Sources/Robot", ["/World/envs/env_0/Robot"]),
                ("/Sources/Robot/Camera", ["/World/envs/env_1/Robot/Camera"]),
            ],
        ),
        (
            "/Sources/Camera",
            (0, 1),
            [
                ("/Sources/Robot", ["/World/envs/env_0/Robot"]),
                ("/Sources/Camera", ["/World/envs/env_0/Robot/Camera", "/World/envs/env_1/Robot/Camera"]),
            ],
        ),
        (
            "/Sources/Robot/Camera",
            (0,),
            [("/Sources/Robot/Camera", ["/World/envs/env_0/Robot/Camera", "/World/envs/env_1/Robot/Camera"])],
        ),
        ("/Sources/Robot/Camera", (), []),
    ],
    ids=["child covered by its parent", "independent child", "parent not routed", "nothing routed"],
)
def test_clone_context_omits_covered_children_and_honors_routing(child_source, routed, expected):
    """Each context prepares only its own backend; routed parents carry their covered children."""
    from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXBackendCfg
    from isaaclab_ov.stage import OvstageBackendCfg

    cfgs = (
        AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot/Camera", spawn=SpawnerCfg(spawn_path=child_source)),
        AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot", spawn=SpawnerCfg(spawn_path="/Sources/Robot")),
    )
    plan = make_clone_plan(cfgs, ((0, 1), (0,)), 2, positions=np.zeros((2, 3), dtype=np.float32))

    renderer_cfg = OVRTXRendererCfg()
    configs = (
        OVRTXBackendCfg(scene_key=renderer_cfg, use_ovstage=False, read_gpu_transforms=True),
        OvstageBackendCfg(scene_key=renderer_cfg),
        OVRTXBackendCfg(scene_key=renderer_cfg, use_ovstage=True, read_gpu_transforms=True),
    )
    for selected, context in enumerate((OvrtxReplicateContext, OvstageReplicateContext)):
        backends = [SimpleNamespace() for _ in configs]
        sim = SimpleNamespace(_backend_registry=list(zip(configs, backends)), physics_manager=SimpleNamespace())
        context(sim).replicate(plan, routed)
        assert backends[selected].clone_copies == expected
        assert not vars(sim.physics_manager)
        assert all(not vars(backend) for index, backend in enumerate(backends) if index != selected)
