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
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer, _write_file  # noqa: E402

    from pxr import Usd, UsdGeom  # noqa: E402
else:
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    Usd = None
    UsdGeom = None
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


def _assert_export_contains_env_roots_but_omits_children(exported: str, env_indices: range | list[int]) -> None:
    """Listed environments and their unique children are omitted from the stage export."""
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
    renderer._camera_rel_path = "Camera"
    renderer._render_product_paths = []
    renderer._exported_usd_string = None
    renderer._initialized_scene = False
    renderer._use_ovstage = False
    renderer._object_scales = None
    renderer._object_scales_by_path = {}
    return renderer


def _make_camera_render_spec(num_envs: int = 1) -> CameraRenderSpec:
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
        device="cpu",
        num_instances=num_envs,
        camera_prim_paths=camera_paths,
        view_count=num_envs,
        camera_path_relative_to_env_0="Camera",
    )


def test_clone_sources_in_ovrtx_heterogeneous_rows():
    """Each active row clones its targets while an inactive row is skipped."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Object", "/World/envs/env_0/Light"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object", "/World/envs/env_{}/Light"),
        clone_mask=torch.tensor(
            [
                [True, True, True, True],
                [False, False, True, True],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        ),
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
            [
                "/World/envs/env_1/Robot",
                "/World/envs/env_2/Robot",
                "/World/envs/env_3/Robot",
            ],
        ),
        (
            "/World/envs/env_1/Object",
            ["/World/envs/env_2/Object", "/World/envs/env_3/Object"],
        ),
    ]


def test_clone_sources_in_ovrtx_raises_on_clone_failure():
    """clone_usd failures surface as RuntimeError with the row index."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        positions=torch.zeros((2, 3)),
    )

    def _clone_usd(source: str, target_paths: list[str]) -> None:
        raise OSError("clone failed")

    renderer._renderer.clone_usd = _clone_usd

    with pytest.raises(RuntimeError, match="Failed to clone row 0"):
        renderer._clone_sources_in_ovrtx()


def test_clone_sources_in_ovrtx_writes_plan_positions_after_cloning():
    """Legacy OVRTX cloning writes translated identity root transforms from the plan."""
    renderer = _make_ovrtx_renderer_without_backend()
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5], [-3.0, 4.0, 1.5]])
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 3), dtype=torch.bool),
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
    assert clone_calls == [("/World/envs/env_0", ["/World/envs/env_1", "/World/envs/env_2"])]
    assert len(write_calls) == 1
    assert write_calls[0]["attribute_name"] == "omni:xform"
    np.testing.assert_array_equal(write_calls[0]["tensor"], expected)


@pytest.mark.skipif(importlib.util.find_spec("ovstage") is None, reason="requires optional module: ovstage")
def test_clone_sources_ovstage_writes_plan_positions_after_cloning(monkeypatch: pytest.MonkeyPatch):
    """Ovstage skips inactive rows and writes translated identity root transforms from the plan."""
    renderer = _make_ovrtx_renderer_without_backend()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.5, -2.0, 0.25], [3.0, 4.0, 0.5]])
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[False, False, False], [False, True, True]]),
        positions=positions,
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

    monkeypatch.setattr("isaaclab_ov.renderers.ovrtx_renderer._xform_tensor_from_numpy", _record_xforms)

    renderer._clone_sources_ovstage()

    expected = np.tile(np.eye(4, dtype=np.float64), (3, 1, 1))
    expected[:, 3, :3] = positions.numpy()
    assert events == [
        ("clone", "/World/envs/env_1/Object", ["/World/envs/env_2/Object"]),
        ("paths", "envs", ["/World/envs/env_0", "/World/envs/env_1", "/World/envs/env_2"]),
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


@pytest.mark.parametrize(
    "clone_plan",
    [
        None,
        ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 2), dtype=torch.bool),
        ),
    ],
)
def test_prepare_stage_requires_published_plan_positions(monkeypatch: pytest.MonkeyPatch, clone_plan: ClonePlan | None):
    """OVRTX stage preparation rejects an absent plan and a plan without positions."""
    _patch_simulation_context(monkeypatch, clone_plan)

    with pytest.raises(RuntimeError, match="Clone plan with environment positions is required"):
        _make_ovrtx_renderer_without_backend().prepare_stage(_make_multi_env_stage(2), 2)


def test_prepare_stage_writes_pre_ovrtx_stage_dump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """prepare_stage writes the raw stage before OVRTX-specific preparation."""
    _patch_simulation_context(
        monkeypatch,
        ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 2), dtype=torch.bool),
            positions=torch.zeros((2, 3)),
        ),
    )

    stage = _make_multi_env_stage(2)
    renderer = _make_ovrtx_renderer_without_backend()
    renderer.cfg.temp_usd_dir = str(tmp_path)
    expected_pre_export = stage.ExportToString()

    renderer.prepare_stage(stage, 2)

    pre_stage_path = tmp_path / _PRE_OVRTX_STAGE_FILE
    assert pre_stage_path.is_file()
    assert pre_stage_path.read_text(encoding="utf-8") == expected_pre_export
    assert pre_stage_path.read_text(encoding="utf-8") != stage.ExportToString()
    assert (tmp_path / _OVRTX_STAGE_FILE).exists() is False


def test_prepare_stage_skips_temp_usd_write_when_temp_usd_dir_unset(monkeypatch: pytest.MonkeyPatch):
    """prepare_stage does not write debug dumps when temp_usd_dir is None."""
    _patch_simulation_context(
        monkeypatch,
        ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 2), dtype=torch.bool),
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

    renderer.prepare_stage(stage, 2)

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

    renderer._initialize_from_spec(_make_camera_render_spec(num_envs=1))

    combined_path = tmp_path / _OVRTX_STAGE_FILE
    combined_text = combined_path.read_text(encoding="utf-8")
    assert combined_text.startswith("#usda 1.0")
    assert 'def RenderProduct "RenderProduct"' in combined_text
    assert open_calls == [combined_text]
    assert renderer._exported_usd_string is None


def test_initialize_from_spec_refreshes_camera_relationship_after_cloning():
    """Multi-environment initialization rewrites the RenderProduct cameras after cloning."""
    num_envs = 4
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._exported_usd_string = "#usda 1.0\n"

    call_order: list[str] = []
    write_array_calls: list[tuple[list[str], str, list[list[str]]]] = []

    renderer._renderer.open_usd_from_string = lambda _usd_string: call_order.append("open")
    renderer._clone_sources_in_ovrtx = lambda: call_order.append("clone")
    renderer._update_scene_partitions_after_clone = lambda _num_envs: call_order.append("partitions")

    def _write_array_attribute(prim_paths: list[str], attribute_name: str, tensors: list[list[str]]) -> None:
        call_order.append("rewrite_cameras")
        write_array_calls.append((prim_paths, attribute_name, tensors))

    renderer._renderer.write_array_attribute = _write_array_attribute
    renderer._renderer.bind_attribute = lambda **_kwargs: object()
    renderer._renderer.write_attribute = lambda **_kwargs: None
    renderer._setup_xform_bindings = lambda: None
    renderer._setup_deformable_bindings = lambda _num_envs: None

    spec = _make_camera_render_spec(num_envs=num_envs)
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
    """prepare_stage stores the published clone plan and exports a trimmed prototype stage."""
    num_envs = 4

    published = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, num_envs), dtype=torch.bool),
        positions=torch.zeros((num_envs, 3)),
    )
    _patch_simulation_context(monkeypatch, published)

    stage = _make_multi_env_stage(num_envs)
    renderer = _make_ovrtx_renderer_without_backend()

    renderer.prepare_stage(stage, 4)

    assert renderer._clone_plan is published

    # Only the env_0 prototype subtree is exported.
    _assert_export_contains_env_roots_and_children(renderer._exported_usd_string, [0])
    _assert_export_contains_env_roots_but_omits_children(renderer._exported_usd_string, [1, 2, 3])
