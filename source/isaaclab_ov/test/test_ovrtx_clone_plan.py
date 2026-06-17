# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX clone-plan resolution and OVRTX-side cloning."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
    from isaaclab_ov.renderers.ovrtx_renderer import (  # noqa: E402
        OVRTXRenderer,
        _create_homogeneous_clone_plan,
        _resolve_clone_plan,
        _write_file,
    )

    from pxr import Usd, UsdGeom  # noqa: E402
else:
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    Usd = None
    UsdGeom = None
    _create_homogeneous_clone_plan = None
    _resolve_clone_plan = None
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
    renderer._renderer = SimpleNamespace(clone_usd=lambda *args, **kwargs: None)
    renderer._clone_plan = None
    renderer._camera_rel_path = "Camera"
    renderer._render_product_paths = []
    renderer._exported_usd_string = None
    renderer._initialized_scene = False
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


def test_resolve_clone_plan_returns_homogeneous_when_unpublished(monkeypatch: pytest.MonkeyPatch):
    """Missing published plan falls back to env_0 replication."""
    _patch_simulation_context(monkeypatch, None)

    num_envs = 4
    resolved = _resolve_clone_plan(num_envs)
    expected = _create_homogeneous_clone_plan(num_envs)

    assert resolved.sources == expected.sources
    assert resolved.destinations == expected.destinations
    assert torch.equal(resolved.clone_mask, expected.clone_mask)


def test_resolve_clone_plan_returns_homogeneous_when_no_active_rows(monkeypatch: pytest.MonkeyPatch):
    """Plan with no active rows falls back to homogeneous replication."""
    published = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    _patch_simulation_context(monkeypatch, published)

    num_envs = 4
    resolved = _resolve_clone_plan(num_envs)
    expected = _create_homogeneous_clone_plan(num_envs)

    assert resolved.sources == expected.sources
    assert resolved.destinations == expected.destinations
    assert torch.equal(resolved.clone_mask, expected.clone_mask)


def test_resolve_clone_plan_filters_inactive_rows(monkeypatch: pytest.MonkeyPatch):
    """Inactive clone-plan rows are removed before OVRTX uses the plan."""
    published = ClonePlan(
        sources=(
            "/World/envs/env_0/Robot",
            "/World/envs/env_1/Object",
            "/World/envs/env_0/table",
        ),
        destinations=(
            "/World/envs/env_{}/Robot",
            "/World/envs/env_{}/Object",
            "/World/envs/env_{}/table",
        ),
        clone_mask=torch.tensor(
            [
                [True, True, True, True],
                [False, False, False, False],
                [True, True, True, True],
            ],
            dtype=torch.bool,
        ),
    )
    _patch_simulation_context(monkeypatch, published)

    resolved = _resolve_clone_plan(4)

    assert resolved.sources == (published.sources[0], published.sources[2])
    assert resolved.destinations == (published.destinations[0], published.destinations[2])
    assert torch.equal(resolved.clone_mask, published.clone_mask[[0, 2]])


def test_resolve_clone_plan_returns_published_plan_when_all_active(monkeypatch: pytest.MonkeyPatch):
    """Fully active published plans are reused without copying."""
    published = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 3), dtype=torch.bool),
    )
    _patch_simulation_context(monkeypatch, published)

    resolved = _resolve_clone_plan(3)

    assert resolved is published


def test_clone_sources_in_ovrtx_homogeneous_row():
    """Homogeneous env_0 row clones only env_1..env_{N-1} (env_0 is the source)."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._clone_plan = _create_homogeneous_clone_plan(4)

    clone_calls: list[tuple[str, list[str]]] = []

    def _clone_usd(source: str, target_paths: list[str]) -> None:
        clone_calls.append((source, target_paths))

    renderer._renderer.clone_usd = _clone_usd

    renderer._clone_sources_in_ovrtx()

    assert clone_calls == [
        (
            "/World/envs/env_0",
            ["/World/envs/env_1", "/World/envs/env_2", "/World/envs/env_3"],
        )
    ]


def test_clone_sources_in_ovrtx_heterogeneous_rows():
    """Each active clone-plan row issues its own clone_usd call."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor(
            [
                [True, True, True, True],
                [False, False, True, True],
            ],
            dtype=torch.bool,
        ),
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


def test_clone_sources_in_ovrtx_skips_empty_target_rows():
    """Rows with no clone targets do not call clone_usd."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_7/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor(
            [
                [False, False, False, False],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        ),
    )

    clone_calls: list[tuple[str, list[str]]] = []

    def _clone_usd(source: str, target_paths: list[str]) -> None:
        clone_calls.append((source, target_paths))

    renderer._renderer.clone_usd = _clone_usd

    renderer._clone_sources_in_ovrtx()

    assert clone_calls == []


def test_clone_sources_in_ovrtx_raises_on_clone_failure():
    """clone_usd failures surface as RuntimeError with the row index."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._clone_plan = _create_homogeneous_clone_plan(2)

    def _clone_usd(source: str, target_paths: list[str]) -> None:
        raise OSError("clone failed")

    renderer._renderer.clone_usd = _clone_usd

    with pytest.raises(RuntimeError, match="Failed to clone row 0"):
        renderer._clone_sources_in_ovrtx()


def test_write_file_creates_parent_directory_and_writes_utf8(tmp_path: Path):
    """_write_file creates nested directories and writes UTF-8 content."""
    output_dir = tmp_path / "nested" / "usd"

    _write_file(output_dir, "stage.usda", "#usda 1.0\n")

    output_path = output_dir / "stage.usda"
    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8") == "#usda 1.0\n"


def test_prepare_stage_writes_pre_ovrtx_stage_dump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """prepare_stage writes the raw stage before OVRTX-specific preparation."""
    _patch_simulation_context(monkeypatch, None)

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
    _patch_simulation_context(monkeypatch, None)
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


def test_prepare_stage_stores_clone_plan_and_exports(monkeypatch: pytest.MonkeyPatch):
    """prepare_stage resolves the clone plan and exports a trimmed prototype stage."""
    num_envs = 4

    published = _create_homogeneous_clone_plan(num_envs)
    _patch_simulation_context(monkeypatch, published)

    stage = _make_multi_env_stage(num_envs)
    renderer = _make_ovrtx_renderer_without_backend()

    renderer.prepare_stage(stage, 4)

    assert renderer._clone_plan is not None
    assert renderer._clone_plan.sources == published.sources

    # Only the env_0 prototype subtree is exported.
    _assert_export_contains_env_roots_and_children(renderer._exported_usd_string, [0])
    _assert_export_contains_env_roots_but_omits_children(renderer._exported_usd_string, [1, 2, 3])
