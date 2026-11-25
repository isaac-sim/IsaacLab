# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for typed visual-material writes into OVRTX-owned scenes."""

import contextlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"),
    pytest.mark.ov,
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer
    from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXRendererCfg

    from isaaclab.renderers.base_renderer import VisualMaterialBatch


class _Completion:
    def __init__(self, callback=None):
        self.wait_count = 0
        self._callback = callback

    def wait(self):
        self.wait_count += 1
        if self._callback is not None:
            self._callback()


class _NativeRecorder:
    def __init__(self, events: list[str]):
        self.events = events

    def step(self, **kwargs):
        self.events.append("step")
        return {path: SimpleNamespace(frames=[object()]) for path in kwargs["render_products"]}


class _OvstageRecorder:
    def __init__(self, events: list[str]):
        self.events = events
        self.writes = []
        self.completions = []
        self.released = []

    @contextlib.contextmanager
    def query_from_path_list(self, path_list):
        query = f"query:{path_list}"
        try:
            yield query
        finally:
            self.release_query(query).wait()

    def write_attribute(self, query, attribute_name, **kwargs):
        self.events.append(f"write:{attribute_name}")
        completion = _Completion(lambda: self.events.append(f"wait:{attribute_name}"))
        self.writes.append((query, attribute_name, kwargs, completion))
        self.completions.append(completion)
        return completion

    def release_query(self, query):
        self.released.append(query)
        return _Completion()

    def advance_write_floor(self, **kwargs):
        self.events.append("floor")
        return _Completion(lambda: self.events.append("wait:floor"))


class _PathRecorder:
    def __init__(self):
        self.created = []
        self.destroyed = []

    def create_path_list_from_strings(self, paths):
        path_list = tuple(paths)
        self.created.append(path_list)
        return path_list

    def destroy_path_list(self, path_list):
        self.destroyed.append(path_list)


def _renderer():
    events: list[str] = []
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.backend = SimpleNamespace(render_products=set(), unrendered={})
    renderer.scene = renderer.backend
    renderer._initialized_scene = True
    renderer.backend.renderer = _NativeRecorder(events)
    renderer._visual_material_writer_ref = None
    renderer.cfg = OVRTXRendererCfg()
    renderer._camera_render_data = []
    from isaaclab_ov.stage import OvstageBackend

    renderer.scene = OvstageBackend.__new__(OvstageBackend)
    renderer.scene.stage = _OvstageRecorder(events)
    renderer.scene.paths = _PathRecorder()
    renderer.scene.ordinal = 7
    return renderer, events


def _batch(channel: str, input_names: tuple[str, ...], values: torch.Tensor) -> VisualMaterialBatch:
    material_paths = tuple(f"/Looks/{channel}_{index}" for index in range(len(values)))
    return VisualMaterialBatch(
        channel,
        material_paths,
        tuple(f"{path}/Shader" for path in material_paths),
        input_names,
        values,
    )


@pytest.mark.skipif(importlib.util.find_spec("ovstage") is None, reason="requires optional module: ovstage")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ovstage_compiles_queries_and_publishes_selected_channel_zero_copy():
    renderer, _events = _renderer()
    roughness = torch.tensor([0.2, 0.8], device="cuda")
    color = torch.zeros((3, 3), device="cuda")
    texture_scale = torch.ones((2, 2), device="cuda")
    writer = renderer.visual_material_writer(
        (
            _batch("roughness", ("roughness", "roughness"), roughness),
            _batch("color", ("diffuseColor", "diffuseColor", "diffuse_color_constant"), color),
            _batch("texture_scale", ("texture_scale", "texture_scale"), texture_scale),
        )
    )
    assert len(renderer.scene.paths.created) == 4
    writer(
        {"roughness": torch.tensor([0], dtype=torch.int32, device="cuda")},
        torch.tensor([1], dtype=torch.int32, device="cuda"),
    )

    assert renderer.scene.stage.writes == []
    writer.publish()

    assert len(renderer.scene.stage.writes) == 1
    query, attribute_name, kwargs, completion = renderer.scene.stage.writes[0]
    assert query == f"query:{renderer.scene.paths.created[0]}"
    assert attribute_name == "inputs:roughness"
    assert kwargs["tensors"].untyped_storage().data_ptr() == roughness.untyped_storage().data_ptr()
    assert kwargs["ordinal"] == 7
    assert kwargs["is_array"] is False
    assert kwargs["cuda_event"] == writer._event.cuda_event
    writer.drain()
    assert completion.wait_count == 1


def test_render_publishes_and_drains_material_writes_at_backend_boundary():
    renderer, events = _renderer()
    paths = ("/Render/Product0", "/Render/Product1")
    renderer._camera_render_data = [
        SimpleNamespace(render_product_paths=(path,), warp_buffers={}, first_frame_error=None) for path in paths
    ]
    renderer.backend.render_products = set(paths)
    renderer._process_render_frames = lambda *args: None

    class Writer:
        def publish(self):
            events.append("publish")

        def drain(self):
            events.append("drain")

    writer = Writer()
    renderer._visual_material_writer_ref = lambda: writer
    renderer.render_batch(renderer._camera_render_data)

    assert events == ["publish", "floor", "wait:floor", "drain", "step"]


@pytest.mark.parametrize(
    ("failure", "expected_events"),
    (("publish", ["publish", "drain"]), ("floor", ["publish", "floor", "drain"])),
)
def test_ovstage_drain_does_not_mask_publish_or_floor_failure(failure, expected_events):
    renderer, events = _renderer()
    renderer._camera_render_data = [
        SimpleNamespace(render_product_paths=("/RenderCamera_0/Product",), first_frame_error=None)
    ]

    class Writer:
        def publish(self):
            events.append("publish")
            if failure == "publish":
                raise ValueError("publish failed")

        def drain(self):
            events.append("drain")
            raise RuntimeError("drain failed")

    writer = Writer()
    renderer._visual_material_writer_ref = lambda: writer

    def advance_write_floor(**_kwargs):
        events.append("floor")
        raise ValueError("floor failed")

    renderer.scene.stage.advance_write_floor = advance_write_floor
    with pytest.raises(ValueError, match=failure):
        renderer.render(renderer._camera_render_data[0])

    assert events == expected_events


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_writer_close_drains_and_releases_compiled_addresses():
    renderer, _events = _renderer()
    writer = renderer.visual_material_writer((_batch("roughness", ("roughness",), torch.zeros(1, device="cuda")),))
    writer(None)
    writer.publish()
    writer.close()

    writer.close()
    assert renderer.scene.stage.released == [f"query:{paths}" for paths in renderer.scene.paths.created]
    assert renderer.scene.paths.destroyed == renderer.scene.paths.created
    assert all(op.wait_count == 1 for op in renderer.scene.stage.completions)


@pytest.mark.parametrize("values", [torch.zeros(1, dtype=torch.float64), torch.zeros(1, 4)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compilation_rejects_unsupported_device_buffers_before_backend_mutation(values):
    renderer, _events = _renderer()
    valid = _batch("color", ("diffuseColor",), torch.zeros(1, 3, device="cuda"))
    invalid = _batch("unsupported", ("value",), values.to(device="cuda"))

    with pytest.raises(TypeError, match="float, float2, or float3"):
        renderer.visual_material_writer((valid, invalid))

    assert renderer._visual_material_writer_ref is None
    assert renderer.scene.paths.created == []


def test_compilation_rejects_host_buffers_without_fallback():
    renderer, _events = _renderer()
    with pytest.raises(RuntimeError, match="one CUDA device"):
        renderer.visual_material_writer((_batch("color", ("diffuseColor",), torch.zeros(1, 3)),))

    assert renderer._visual_material_writer_ref is None
    assert renderer.scene.paths.created == []


def test_writer_factory_requires_ingested_detached_scene():
    renderer, _events = _renderer()
    renderer._initialized_scene = False
    assert renderer.visual_material_writer is None
    renderer._initialized_scene = True
    assert callable(renderer.visual_material_writer)
