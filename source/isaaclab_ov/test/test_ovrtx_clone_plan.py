# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX clone-plan consumption and OVRTX-side cloning."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg

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
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer, _write_file  # noqa: E402

    from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: E402
else:
    OVRTXRenderer = None
    ovrtx_renderer_module = None
    OVRTXRendererCfg = None
    Sdf = None
    Usd = None
    UsdGeom = None
    UsdShade = None
    _write_file = None


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


def _assert_export_contains_env_roots_and_children(exported: str, env_indices: range | list[int]) -> None:
    """Listed environment roots appear in the stage export."""
    for env_idx in env_indices:
        assert f'def Xform "env_{env_idx}"' in exported
        assert f'def Xform "Object_env{env_idx}_only"' in exported

    assert exported.count('def Xform "Robot"') == len(env_indices)
    assert exported.count('def Camera "Camera"') == len(env_indices)


def _assert_export_contains_empty_env_roots(exported: str, env_indices: range | list[int]) -> None:
    """Listed environment roots remain while their non-source children are omitted."""
    for env_idx in env_indices:
        assert f'def Xform "env_{env_idx}"' in exported
        assert f'def Xform "Object_env{env_idx}_only"' not in exported


def _patch_simulation_context(monkeypatch: pytest.MonkeyPatch, clone_plan: ClonePlan | None) -> None:
    mock_ctx = SimpleNamespace(get_clone_plan=lambda: clone_plan)
    monkeypatch.setattr(
        "isaaclab_ov.renderers.ovrtx_renderer.SimulationContext",
        SimpleNamespace(instance=lambda: mock_ctx),
    )


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer._renderer = SimpleNamespace(
        clone_usd=lambda *args, **kwargs: None,
        write_array_attribute=lambda *args, **kwargs: None,
        write_attribute=lambda *args, **kwargs: None,
    )
    renderer._clone_plan = None
    renderer._device = "cuda:0"  # __init__'s default, replaced by create_render_data(spec)
    # create_render_data resolves this from the spec; tests that bypass it get the default.
    renderer._warp_device = SimpleNamespace(ordinal=0)
    renderer._env_paths = ()
    renderer._camera_paths = ()
    renderer._camera_env_paths = ()
    renderer._source_camera_path = None
    renderer._render_product_paths = []
    renderer._exported_usd_string = None
    renderer._initialized_scene = False
    renderer._use_ovstage = False
    renderer._object_scales = None
    renderer._object_scales_by_path = {}
    return renderer


def _make_camera_render_spec(
    num_envs: int = 1, device: str = "cpu", camera_paths: tuple[str, ...] | None = None
) -> CameraRenderSpec:
    spawn = PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    )
    camera_paths = camera_paths or tuple(f"/World/envs/env_{env_idx}/Camera" for env_idx in range(num_envs))
    cfg = CameraCfg(
        height=8,
        width=16,
        prim_path=camera_paths[0],
        spawn=spawn,
        data_types=["rgb"],
    )
    return CameraRenderSpec(
        cfg=cfg,
        device=device,
        num_instances=num_envs,
        camera_prim_paths=camera_paths,
        view_count=num_envs,
    )


def _prepare_stage(renderer: OVRTXRenderer, stage: Usd.Stage, num_envs: int) -> None:
    """Run the renderer's declared camera-then-stage preparation lifecycle."""
    renderer.prepare_cameras(stage, _make_camera_render_spec(num_envs))
    renderer.prepare_stage(stage, num_envs)


def _attach_camera_plan(renderer: OVRTXRenderer, spec: CameraRenderSpec) -> None:
    """Attach the plan-derived path state normally produced by stage preparation."""
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, spec.num_instances), dtype=torch.bool),
        env_ids=torch.arange(spec.num_instances),
        positions=torch.zeros((spec.num_instances, 3)),
    )
    renderer._env_paths = tuple(f"/World/envs/env_{env_id}" for env_id in range(spec.num_instances))
    renderer._camera_paths = spec.camera_prim_paths
    renderer._camera_env_paths = renderer._env_paths
    renderer._source_camera_path = spec.camera_prim_paths[0]


def test_clone_sources_in_ovrtx_uses_active_plan_rows():
    """Each plan row clones directly to its active destinations other than its source."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Object", "/World/envs/env_0/Light"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object", "/World/envs/env_{}/Light"),
        clone_mask=torch.tensor(
            [
                [True, True, True, True],
                [False, True, True, True],
                [True, False, False, False],
            ],
            dtype=torch.bool,
        ),
        env_ids=torch.arange(4),
        positions=torch.zeros((4, 3)),
    )
    clone_calls: list[tuple[str, list[str]]] = []

    def _clone_usd(source: str, target_paths: list[str]) -> None:
        clone_calls.append((source, target_paths))

    renderer._renderer.clone_usd = _clone_usd

    renderer._clone_sources_in_ovrtx()

    assert clone_calls == [
        (
            "/World/envs/env_0/Robot",
            ["/World/envs/env_1/Robot", "/World/envs/env_2/Robot", "/World/envs/env_3/Robot"],
        ),
        ("/World/envs/env_1/Object", ["/World/envs/env_2/Object", "/World/envs/env_3/Object"]),
    ]


def test_clone_sources_in_ovrtx_raises_on_clone_failure():
    """clone_usd failures surface as RuntimeError with the row index."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.arange(2),
        positions=torch.zeros((2, 3)),
    )

    def _clone_usd(source: str, target_paths: list[str]) -> None:
        raise OSError("clone failed")

    renderer._renderer.clone_usd = _clone_usd

    with pytest.raises(RuntimeError, match="Failed to clone row 0 from /World/envs/env_0"):
        renderer._clone_sources_in_ovrtx()


def test_clone_sources_in_ovrtx_writes_plan_positions_after_cloning():
    """Legacy OVRTX cloning writes translated identity root transforms from the plan."""
    renderer = _make_ovrtx_renderer_without_backend()
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5], [-3.0, 4.0, 1.5]])
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_5",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 3), dtype=torch.bool),
        env_ids=torch.tensor([5, 11, 3]),
        positions=positions,
    )
    call_order: list[str] = []
    clone_calls: list[tuple[str, list[str]]] = []
    write_calls: list[dict] = []

    def _clone_usd(source: str, target_paths: list[str]) -> None:
        call_order.append("clone")
        clone_calls.append((source, target_paths))

    renderer._renderer.clone_usd = _clone_usd

    def _write_attribute(**kwargs):
        call_order.append("write")
        write_calls.append(kwargs)

    renderer._renderer.write_attribute = _write_attribute

    renderer._clone_sources_in_ovrtx()

    expected = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    expected[:, 3, :3] = positions.numpy()
    assert call_order == ["clone", "write"]
    assert clone_calls == [("/World/envs/env_5", ["/World/envs/env_11", "/World/envs/env_3"])]
    assert len(write_calls) == 1
    assert write_calls[0]["prim_paths"] == ["/World/envs/env_5", "/World/envs/env_11", "/World/envs/env_3"]
    assert write_calls[0]["attribute_name"] == "omni:xform"
    np.testing.assert_array_equal(write_calls[0]["tensor"], expected)


@pytest.mark.skipif(importlib.util.find_spec("ovstage") is None, reason="requires optional module: ovstage")
def test_clone_sources_ovstage_writes_plan_positions_after_cloning(monkeypatch: pytest.MonkeyPatch):
    """Ovstage skips inactive rows and writes translated identity root transforms from the plan."""
    renderer = _make_ovrtx_renderer_without_backend()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.5, -2.0, 0.25], [3.0, 4.0, 0.5]])
    renderer._clone_plan = ClonePlan(
        sources=("/World/scenes/scene_7/Robot", "/World/scenes/scene_3/Object"),
        destinations=("/World/scenes/scene_{}/Robot", "/World/scenes/scene_{}/Object"),
        clone_mask=torch.tensor([[True, False, False], [False, True, True]]),
        env_ids=torch.tensor([7, 3, 12]),
        positions=positions,
        env_template="/World/scenes/scene_{}",
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

    renderer._stage = SimpleNamespace(
        query_from_path_list=_query,
        clone=_clone,
        write_attribute=_write,
        release_query=lambda _query: completion,
    )
    renderer._stage_paths = SimpleNamespace(
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
    expected[:, 3, :3] = positions.numpy()
    assert events == [
        ("clone", "/World/scenes/scene_3/Object", ["/World/scenes/scene_12/Object"]),
        (
            "paths",
            "envs",
            ["/World/scenes/scene_7", "/World/scenes/scene_3", "/World/scenes/scene_12"],
        ),
        ("query", "envs", "env_paths"),
        ("write", "omni:xform", "root_xforms"),
    ]
    assert len(xforms) == 1
    np.testing.assert_array_equal(xforms[0], expected)


def test_write_file_creates_parent_directory_and_writes_utf8(tmp_path: Path):
    """_write_file creates nested directories and writes UTF-8 content."""
    output_dir = tmp_path / "nested" / "usd"

    _write_file(output_dir, "stage.usda", "#usda 1.0\n")

    output_path = output_dir / "stage.usda"
    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8") == "#usda 1.0\n"


def test_prepare_stage_requires_published_plan(monkeypatch: pytest.MonkeyPatch):
    """OVRTX stage preparation rejects an absent plan."""
    _patch_simulation_context(monkeypatch, None)
    stage = _make_multi_env_stage(2)
    renderer = _make_ovrtx_renderer_without_backend()

    with pytest.raises(RuntimeError, match="Clone plan is required"):
        _prepare_stage(renderer, stage, 2)


def test_custom_environment_template_drives_ovrtx_stage_and_clone_paths(monkeypatch: pytest.MonkeyPatch):
    """OVRTX derives non-contiguous roots, nested cameras, partitions, scales, and cloning from the plan."""
    env_ids = torch.tensor([2, 5])
    env_template = "/World/scenes/scene_{}"
    env_paths = tuple(env_template.format(env_id) for env_id in env_ids.tolist())
    camera_paths = tuple(f"{env_path}/Robot/link/Camera" for env_path in reversed(env_paths))
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/scenes")
    for env_path in env_paths:
        UsdGeom.Xform.Define(stage, env_path)
        robot = UsdGeom.Xform.Define(stage, f"{env_path}/Robot")
        robot.AddScaleOp().Set((2.0, 1.0, 1.0))
        UsdGeom.Xform.Define(stage, f"{env_path}/Robot/link")
    for camera_path in camera_paths:
        UsdGeom.Camera.Define(stage, camera_path)

    plan = ClonePlan(
        sources=(f"{env_paths[1]}/Robot",),
        destinations=(f"{env_template}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=env_ids,
        positions=torch.zeros((2, 3)),
        env_template=env_template,
    )
    _patch_simulation_context(monkeypatch, plan)
    renderer = _make_ovrtx_renderer_without_backend()
    spec = _make_camera_render_spec(2, camera_paths=camera_paths)
    renderer.prepare_cameras(stage, spec)
    renderer.prepare_stage(stage, 2)

    assert renderer._env_paths == env_paths
    assert renderer._camera_paths == camera_paths
    assert renderer._camera_env_paths == tuple(reversed(env_paths))
    assert f"{env_paths[1]}/Robot" in renderer._object_scales_by_path

    root_layer = stage.GetRootLayer()
    for env_path in env_paths:
        camera_path = f"{env_path}/Robot/link/Camera"
        token = env_path.rsplit("/", 1)[-1]
        assert (
            root_layer.GetAttributeAtPath(Sdf.Path(env_path).AppendProperty("primvars:omni:scenePartition")).default
            == token
        )
        assert (
            root_layer.GetAttributeAtPath(Sdf.Path(camera_path).AppendProperty("omni:scenePartition")).default == token
        )

    renderer._initialize_from_spec_legacy = lambda _spec: None
    renderer._initialize_from_spec(spec)
    assert renderer._source_camera_path == camera_paths[0]

    clone_calls: list[tuple[str, list[str]]] = []
    write_calls: list[tuple[tuple, dict]] = []
    renderer._renderer.clone_usd = lambda source, destinations: clone_calls.append((source, destinations))
    renderer._renderer.write_attribute = lambda *args, **kwargs: write_calls.append((args, kwargs))
    renderer._clone_sources_in_ovrtx()
    renderer._update_scene_partitions_after_clone()

    assert clone_calls == [(f"{env_paths[1]}/Robot", [f"{env_paths[0]}/Robot"])]
    assert write_calls[0][1]["prim_paths"] == list(env_paths)
    assert write_calls[1][0][:3] == (env_paths, "primvars:omni:scenePartition", ["scene_2", "scene_5"])
    assert write_calls[2][0][:3] == (camera_paths, "omni:scenePartition", ["scene_5", "scene_2"])


def test_custom_environment_membership_excludes_nested_camera_subtree(monkeypatch: pytest.MonkeyPatch):
    """Newton bindings use exact environment and camera subtrees instead of default-path substrings."""
    env_paths = ("/World/scenes/scene_2", "/World/scenes/scene_5")
    camera_paths = tuple(f"{env_path}/Robot/link/Camera" for env_path in env_paths)
    model = SimpleNamespace(
        body_label=(
            f"{env_paths[0]}/Robot/base",
            camera_paths[0],
            f"{camera_paths[1]}/housing",
            "/World/envs/env_0/Robot/base",
        )
    )
    monkeypatch.setattr("isaaclab_newton.physics.NewtonManager.get_model", lambda: model)
    monkeypatch.setattr(ovrtx_renderer_module.SimulationContext, "instance", lambda: object())
    monkeypatch.setattr(ovrtx_renderer_module.wp, "array", lambda values, **_kwargs: values)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "zeros", lambda *_args, **_kwargs: [])

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._env_paths = env_paths
    renderer._camera_paths = camera_paths
    renderer._camera_env_paths = env_paths
    renderer._clone_plan = ClonePlan(
        sources=(f"{env_paths[0]}/Robot",),
        destinations=("/World/scenes/scene_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.tensor([2, 5]),
        positions=torch.zeros((2, 3)),
        env_template="/World/scenes/scene_{}",
    )
    bound_paths: list[str] = []
    renderer._renderer.bind_attribute = lambda prim_paths, **_kwargs: bound_paths.extend(prim_paths) or object()
    renderer._renderer.write_attribute = lambda **_kwargs: None
    renderer._create_object_scale_array = lambda paths: paths

    renderer._setup_xform_bindings_legacy()

    assert bound_paths == [f"{env_paths[0]}/Robot/base"]


def test_prepare_stage_keeps_material_binding_inside_clone_source(monkeypatch: pytest.MonkeyPatch):
    """A row export keeps its bound material beneath the root cloned by OVRTX."""
    num_envs = 3
    stage = _make_multi_env_stage(num_envs)
    source = "/World/envs/env_0/Robot"
    material = UsdShade.Material.Define(stage, f"{source}/warm")
    body = UsdGeom.Xform.Define(stage, f"{source}/Body").GetPrim()
    UsdShade.MaterialBindingAPI.Apply(body)
    UsdShade.MaterialBindingAPI(body).Bind(material)
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, num_envs), dtype=torch.bool),
        env_ids=torch.arange(num_envs),
        positions=torch.zeros((num_envs, 3)),
    )
    _patch_simulation_context(monkeypatch, plan)
    renderer = _make_ovrtx_renderer_without_backend()

    _prepare_stage(renderer, stage, num_envs)

    exported_layer = Sdf.Layer.CreateAnonymous(".usda")
    assert exported_layer.ImportFromString(renderer._exported_usd_string)
    exported_stage = Usd.Stage.Open(exported_layer)
    binding = UsdShade.MaterialBindingAPI(exported_stage.GetPrimAtPath(f"{source}/Body")).GetDirectBindingRel()
    assert binding.GetTargets() == [Sdf.Path(f"{source}/warm")]
    assert exported_stage.GetPrimAtPath(f"{source}/warm")
    assert not exported_stage.GetPrimAtPath("/World/envs/env_1/Robot")


def test_prepare_stage_writes_pre_ovrtx_stage_dump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """prepare_stage writes the raw stage before OVRTX-specific preparation."""
    _patch_simulation_context(
        monkeypatch,
        ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 2), dtype=torch.bool),
            env_ids=torch.arange(2),
            positions=torch.zeros((2, 3)),
        ),
    )

    stage = _make_multi_env_stage(2)
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.temp_usd_dir = str(tmp_path)
    expected_pre_export = stage.ExportToString()

    _prepare_stage(renderer, stage, 2)

    pre_stage_path = tmp_path / _PRE_OVRTX_STAGE_FILE
    assert pre_stage_path.is_file()
    assert pre_stage_path.read_text(encoding="utf-8") == expected_pre_export
    assert (tmp_path / _OVRTX_STAGE_FILE).exists() is False


def test_prepare_stage_skips_temp_usd_write_when_temp_usd_dir_unset(monkeypatch: pytest.MonkeyPatch):
    """prepare_stage does not write debug dumps when temp_usd_dir is None."""
    _patch_simulation_context(
        monkeypatch,
        ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 2), dtype=torch.bool),
            env_ids=torch.arange(2),
            positions=torch.zeros((2, 3)),
        ),
    )
    write_calls: list[tuple[Path, str, str]] = []

    def _record_write(output_dir: Path, file_name: str, content: str) -> None:
        write_calls.append((output_dir, file_name, content))

    monkeypatch.setattr("isaaclab_ov.renderers.ovrtx_renderer._write_file", _record_write)

    stage = _make_multi_env_stage(2)
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.temp_usd_dir = None

    _prepare_stage(renderer, stage, 2)

    assert write_calls == []


def test_initialize_from_spec_writes_combined_stage_dump(tmp_path: Path):
    """_initialize_from_spec writes the combined stage when temp_usd_dir is set."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.temp_usd_dir = str(tmp_path)
    renderer._exported_usd_string = "#usda 1.0\n"

    open_calls: list[str] = []
    renderer._renderer.open_usd_from_string = lambda usd_string: open_calls.append(usd_string)
    renderer._renderer.bind_attribute = lambda **kwargs: object()
    renderer._renderer.write_attribute = lambda **kwargs: None

    spec = _make_camera_render_spec(num_envs=1)
    _attach_camera_plan(renderer, spec)
    renderer._initialize_from_spec(spec)

    combined_path = tmp_path / _OVRTX_STAGE_FILE
    combined_text = combined_path.read_text(encoding="utf-8")
    assert combined_text.startswith("#usda 1.0")
    assert 'def RenderProduct "RenderProduct"' in combined_text
    assert open_calls == [combined_text]
    assert renderer._exported_usd_string is None


def test_create_render_data_pins_the_render_product_to_the_spec_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The render product is pinned to the CUDA device whose Warp kernels read its render vars.

    Without this, OVRTX picks the device itself and hands back buffers on ``cuda:0`` while the tile
    extraction kernels launch on the simulation device.
    """
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.temp_usd_dir = str(tmp_path)
    renderer._exported_usd_string = "#usda 1.0\n"

    renderer._renderer.open_usd_from_string = lambda _usd_string: None
    renderer._renderer.bind_attribute = lambda **kwargs: object()
    renderer._renderer.write_attribute = lambda **kwargs: None

    class _FakeWarpDevice:
        ordinal = 1

        def __str__(self) -> str:
            return "cuda:1"

    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_device", lambda device: _FakeWarpDevice())
    spec = _make_camera_render_spec(num_envs=1, device="cuda:1")
    _attach_camera_plan(renderer, spec)
    renderer.create_render_data(spec)

    combined_text = (tmp_path / _OVRTX_STAGE_FILE).read_text(encoding="utf-8")
    assert "uint[] deviceIds = [1]" in combined_text


def test_initialize_from_spec_refreshes_camera_relationship_after_cloning():
    """Multi-environment initialization rewrites the RenderProduct cameras after cloning."""
    num_envs = 4
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._exported_usd_string = "#usda 1.0\n"

    call_order: list[str] = []
    write_array_calls: list[tuple[list[str], str, list[list[str]]]] = []

    renderer._renderer.open_usd_from_string = lambda _usd_string: call_order.append("open")
    renderer._clone_sources_in_ovrtx = lambda: call_order.append("clone")
    renderer._update_scene_partitions_after_clone = lambda: call_order.append("partitions")

    def _write_array_attribute(prim_paths: list[str], attribute_name: str, tensors: list[list[str]]) -> None:
        call_order.append("rewrite_cameras")
        write_array_calls.append((prim_paths, attribute_name, tensors))

    renderer._renderer.write_array_attribute = _write_array_attribute
    renderer._renderer.bind_attribute = lambda **_kwargs: object()
    renderer._renderer.write_attribute = lambda **_kwargs: None
    renderer._setup_xform_bindings = lambda: None
    renderer._setup_deformable_bindings = lambda _num_envs: None

    spec = _make_camera_render_spec(num_envs=num_envs)
    _attach_camera_plan(renderer, spec)
    renderer._initialize_from_spec(spec)

    assert call_order == ["open", "clone", "partitions", "rewrite_cameras"]
    assert write_array_calls == [
        (
            ["/Render/RenderProduct"],
            "camera",
            [[f"/World/envs/env_{env_id}/Camera" for env_id in range(num_envs)]],
        )
    ]


def test_prepare_stage_stores_clone_plan_and_exports(monkeypatch: pytest.MonkeyPatch):
    """prepare_stage stores the clone plan and exports only its source-row content."""
    num_envs = 4

    published = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, num_envs), dtype=torch.bool),
        env_ids=torch.arange(num_envs),
        positions=torch.zeros((num_envs, 3)),
    )
    _patch_simulation_context(monkeypatch, published)

    stage = _make_multi_env_stage(num_envs)
    renderer = _make_ovrtx_renderer_without_backend()

    _prepare_stage(renderer, stage, 4)

    assert renderer._clone_plan is published

    # Only the env_0 source subtree keeps content; legacy OVRTX still needs every root for xform writes.
    _assert_export_contains_env_roots_and_children(renderer._exported_usd_string, [0])
    _assert_export_contains_empty_env_roots(renderer._exported_usd_string, [1, 2, 3])
