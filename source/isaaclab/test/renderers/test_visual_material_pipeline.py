# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior and architecture gates for the renderer-owned visual-material pipeline."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.assets.visual_material.visual_material import VisualMaterial, _channel_specs
from isaaclab.cloner import ClonePlan
from isaaclab.renderers.render_context import RenderContext

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = _SOURCE_ROOT / "isaaclab" / "isaaclab"


class _Material:
    def __init__(self, name: str, channel: str, values: torch.Tensor, *, is_per_env: bool = True):
        self.channels = (channel,)
        self.is_per_env = is_per_env
        self.num_instances = len(values)
        self._channel = channel
        self._paths = tuple(f"/World/Looks/{name}_{row}" for row in range(len(values)))
        self._shader_paths = tuple(f"{path}/Shader" for path in self._paths)
        self._material_paths = self._paths
        self._input_names = {channel: channel}
        self._offsets = {channel: -1}
        self._values = {channel: values}

    @property
    def values(self) -> torch.Tensor:
        return self._values[self._channel]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise the GPU scatter path.")
@pytest.mark.parametrize(
    ("channel", "trailing_shape"),
    [
        pytest.param("roughness", (), id="scalar"),
        pytest.param("uv_scale", (2,), id="float2"),
        pytest.param("color", (3,), id="float3"),
    ],
)
def test_partial_gpu_write_updates_flat_material_rows(channel: str, trailing_shape: tuple[int, ...]) -> None:
    def initial_values(offset: float) -> torch.Tensor:
        count = 4 * math.prod(trailing_shape)
        shape = (4, *trailing_shape)
        return torch.arange(count, dtype=torch.float32, device="cuda:0").reshape(shape) + offset

    context = RenderContext()
    first = _Material("first", channel, initial_values(0.0))
    second = _Material("second", channel, initial_values(20.0))
    context.register_visual_material(first)
    context.register_visual_material(second)
    factory = _WriterFactory()
    context.finalize_consumers([SimpleNamespace(visual_material_writer=factory)])

    assert first.values.untyped_storage().data_ptr() == second.values.untyped_storage().data_ptr()
    first_before = first.values.clone()
    expected_second = second.values.clone()
    env_ids = torch.tensor([3, 1], dtype=torch.int32, device="cuda:0")
    update_shape = (1, len(env_ids), *trailing_shape)
    updates = torch.arange(math.prod(update_shape), dtype=torch.float32, device="cuda:0").reshape(update_shape) + 100
    expected_second[env_ids] = updates[0]

    context.write_visual_materials([second], {channel: updates}, env_ids)
    torch.cuda.synchronize()

    torch.testing.assert_close(first.values, first_before)
    torch.testing.assert_close(second.values, expected_second)
    offsets, selected_env_ids = factory.writers[0].writes[-1]
    torch.testing.assert_close(wp.to_torch(offsets[channel]), torch.tensor([4], dtype=torch.int32, device="cuda:0"))
    torch.testing.assert_close(wp.to_torch(selected_env_ids), env_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise stream ordering.")
def test_write_runs_core_and_backend_on_the_calling_torch_stream() -> None:
    streams = []

    class Writer(_Writer):
        def __call__(self, material_offsets=None, env_ids=None) -> None:
            streams.append(wp.get_stream("cuda:0").cuda_stream)
            super().__call__(material_offsets, env_ids)

    context = RenderContext()
    material = _Material("material", "roughness", torch.zeros(2, device="cuda:0"))
    context.register_visual_material(material)
    context.finalize_consumers([SimpleNamespace(visual_material_writer=lambda _batches: Writer())])
    streams.clear()

    stream = torch.cuda.Stream()
    updates = torch.empty((1, 2), device="cuda:0")
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        updates.fill_(0.75)
        context.write_visual_materials([material], {"roughness": updates})
    torch.cuda.current_stream().wait_stream(stream)

    torch.testing.assert_close(material.values, torch.full((2,), 0.75, device="cuda:0"))
    assert streams == [stream.cuda_stream]


class _Writer:
    def __init__(self):
        self.closed = False
        self.calls = 0
        self.writes = []

    def __call__(self, material_offsets=None, env_ids=None) -> None:
        self.calls += 1
        self.writes.append((material_offsets, env_ids))

    def close(self) -> None:
        self.closed = True


class _WriterFactory:
    def __init__(self):
        self.batches = []
        self.writers: list[_Writer] = []

    def __call__(self, batches):
        self.batches.append(batches)
        writer = _Writer()
        self.writers.append(writer)
        return writer


def test_global_material_write_uses_its_single_flat_row() -> None:
    context = RenderContext()
    material = _Material("global", "roughness", torch.zeros(1), is_per_env=False)
    context.register_visual_material(material)
    factory = _WriterFactory()
    context.finalize_consumers([SimpleNamespace(visual_material_writer=factory)])

    context.write_visual_materials([material], {"roughness": torch.tensor([0.75])})

    torch.testing.assert_close(material.values, torch.tensor([0.75]))
    offsets, env_ids = factory.writers[0].writes[-1]
    torch.testing.assert_close(wp.to_torch(offsets["roughness"]), torch.zeros(1, dtype=torch.int32))
    torch.testing.assert_close(wp.to_torch(env_ids), torch.zeros(1, dtype=torch.int32))


def test_finalize_deduplicates_and_rebuilds_shared_writer() -> None:
    context = RenderContext()
    material = _Material("material", "roughness", torch.zeros(2))
    context.register_visual_material(material)
    factory = _WriterFactory()
    consumer = SimpleNamespace(visual_material_writer=factory)
    context._renderer_entries = [(object(), consumer)]  # type: ignore[assignment]  # noqa: SLF001
    visualizers = [consumer, SimpleNamespace(visual_material_writer=factory)]

    context.finalize_consumers(visualizers)
    context.finalize_consumers(visualizers)
    assert len(factory.writers) == 1
    assert not factory.writers[0].closed
    assert factory.writers[0].calls == 1

    material._values["roughness"] = torch.ones(2)  # noqa: SLF001 - model an asset lifecycle reinitialization
    context.finalize_consumers(visualizers, rebuild=True)
    assert len(factory.writers) == 2
    assert factory.writers[0].closed
    assert not factory.writers[1].closed
    assert factory.writers[1].calls == 1
    assert factory.batches[0] is not factory.batches[1]
    assert material.values.untyped_storage().data_ptr() == factory.batches[1][0].values.untyped_storage().data_ptr()
    torch.testing.assert_close(factory.batches[1][0].values, torch.ones(2))


def test_material_registration_is_idempotent_after_lifecycle_stop() -> None:
    context = RenderContext()
    material = _Material("material", "roughness", torch.zeros(2))
    context.register_visual_material(material)
    context.finalize_consumers([])

    context.register_visual_material(material)

    assert context._visual_materials == [material]  # noqa: SLF001

    with pytest.raises(RuntimeError, match="before rendering consumers"):
        context.register_visual_material(_Material("late", "roughness", torch.zeros(2)))


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _runtime_tokens(method: ast.FunctionDef) -> set[str]:
    has_docstring = (
        method.body
        and isinstance(method.body[0], ast.Expr)
        and isinstance(method.body[0].value, ast.Constant)
        and isinstance(method.body[0].value.value, str)
    )
    body = method.body[1:] if has_docstring else method.body
    tokens: set[str] = set()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Name):
            tokens.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr.lower())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            tokens.add(node.value.lower())
    return tokens


def test_runtime_material_writes_have_no_host_or_usd_path() -> None:
    methods = (
        _class_method(_PACKAGE_ROOT / "renderers" / "render_context.py", "RenderContext", "write_visual_materials"),
        _class_method(
            _PACKAGE_ROOT / "assets" / "visual_material" / "visual_material.py", "VisualMaterial", "write_channels"
        ),
    )
    forbidden = ("cpu", "tolist", "numpy", "usd", "sdf")
    for method in methods:
        hits = {token for token in _runtime_tokens(method) if any(name in token for name in forbidden)}
        assert not hits, f"{method.name} contains forbidden runtime tokens: {sorted(hits)}"


def _identifiers(tree: ast.AST) -> set[str]:
    return {
        identifier
        for node in ast.walk(tree)
        for identifier in (
            (node.id,) if isinstance(node, ast.Name) else (node.attr,) if isinstance(node, ast.Attribute) else ()
        )
    }


def test_material_initialization_orders_paths_by_plan_column(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = ClonePlan(
        sources=("/World/envs/env_42/Robot", "/World/envs/env_7/Robot"),
        destinations=("/World/envs/env_{}/Robot",) * 2,
        env_ids=np.asarray([19, 42, 7], dtype=np.int64),
        clone_mask=np.asarray([[True, True, False], [False, False, True]], dtype=np.bool_),
    )
    simulation = SimpleNamespace(get_clone_plan=lambda: plan)
    monkeypatch.setattr(
        "isaaclab.assets.visual_material.visual_material.SimulationContext",
        SimpleNamespace(instance=lambda: simulation),
    )
    registered = []
    material = SimpleNamespace(
        _is_per_env=True,
        _source_material_path="/World/envs/env_42/Robot/Looks/test",
        _source_shader_path="/World/envs/env_42/Robot/Looks/test/Shader",
        cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Robot/Looks/test"),
        _material_paths=(),
        _shader_paths=(),
        _values={},
        _initial_values={"color": torch.tensor([0.1, 0.2, 0.3])},
        device="cpu",
        _render_context=SimpleNamespace(register_visual_material=registered.append),
    )

    VisualMaterial._initialize_impl(material)

    assert material._material_paths == (
        "/World/envs/env_19/Robot/Looks/test",
        "/World/envs/env_42/Robot/Looks/test",
        "/World/envs/env_7/Robot/Looks/test",
    )
    assert material._shader_paths == tuple(path + "/Shader" for path in material._material_paths)
    assert registered == [material]


def test_preview_surface_channels_follow_shader_id() -> None:
    shader = SimpleNamespace(GetShaderId=lambda: "UsdPreviewSurface")

    assert _channel_specs(shader)["color"] == ("diffuseColor", (0.18, 0.18, 0.18))


def test_material_initialization_rejects_partial_owner_row(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        env_ids=np.arange(4, dtype=np.int64),
        clone_mask=np.asarray([[True, True, False, False]], dtype=np.bool_),
    )
    simulation = SimpleNamespace(get_clone_plan=lambda: plan)
    monkeypatch.setattr(
        "isaaclab.assets.visual_material.visual_material.SimulationContext",
        SimpleNamespace(instance=lambda: simulation),
    )
    material = SimpleNamespace(
        _is_per_env=True,
        _source_material_path="/World/envs/env_0/Robot/Looks/test",
        cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Robot/Looks/test"),
    )

    with pytest.raises(ValueError, match="must populate every environment"):
        VisualMaterial._initialize_impl(material)


def test_rejected_visual_material_ownership_does_not_return() -> None:
    material_path = _PACKAGE_ROOT / "assets" / "visual_material" / "visual_material.py"
    cfg_path = material_path.with_name("visual_material_cfg.py")
    scene_path = _PACKAGE_ROOT / "scene" / "interactive_scene.py"
    for path in (material_path, cfg_path, scene_path):
        assert not {"owner_asset", "target_prim_paths"} & _identifiers(ast.parse(path.read_text()))

    material_tree = ast.parse(material_path.read_text())
    material_class = next(
        node for node in material_tree.body if isinstance(node, ast.ClassDef) and node.name == "VisualMaterial"
    )
    assert "FactoryBase" not in {ast.unparse(base) for base in material_class.bases}
    assert not material_path.with_name("base_visual_material.py").exists()
    for package in ("isaaclab_newton", "isaaclab_physx", "isaaclab_ov"):
        backend_assets = _SOURCE_ROOT / package / package / "assets" / "visual_material"
        assert not tuple(backend_assets.glob("*.py"))

    renderer_tree = ast.parse((_PACKAGE_ROOT / "renderers" / "base_renderer.py").read_text())
    renderer_names = {
        node.name
        for node in renderer_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    renderer_names.update(
        node.target.id
        for node in renderer_tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    )
    assert not {"VisualMaterialWriterFactory", "VisualMaterialBackendFactory"} & renderer_names

    simulation_tree = ast.parse((_PACKAGE_ROOT / "sim" / "simulation_context.py").read_text())
    simulation_class = next(
        node for node in simulation_tree.body if isinstance(node, ast.ClassDef) and node.name == "SimulationContext"
    )
    simulation_api = {
        node.name
        for node in simulation_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "visual_material" in node.name
    }
    assert not simulation_api

    owners = []
    for path in (_PACKAGE_ROOT / "envs" / "mdp").glob("*.py"):
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "randomize_visual_color"
            for node in tree.body
        ):
            owners.append(path.name)
    assert owners == ["events.py"]
    assert "randomize_visual_color" not in _identifiers(
        ast.parse((_PACKAGE_ROOT / "envs" / "mdp" / "visual_events.py").read_text())
    )
