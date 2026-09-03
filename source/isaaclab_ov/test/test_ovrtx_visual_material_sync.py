# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for typed visual-material writes into OVRTX-owned scenes."""

import importlib.util
import inspect
from types import SimpleNamespace

import pytest
import torch

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = pytest.mark.skipif(
    bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"
)

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer
    from isaaclab_ov.renderers.visual_materials import OVRTXVisualMaterialWriter
    from ovrtx import DataAccess

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
        self.bindings = []
        self.writes = []

    def bind_attribute(self, prim_paths, attribute_name, **kwargs):
        binding = _NativeBinding(self, tuple(prim_paths), attribute_name, kwargs)
        self.bindings.append(binding)
        return binding

    def step(self, **kwargs):
        self.events.append("step")
        return {}


class _NativeBinding:
    def __init__(self, recorder, prim_paths, attribute_name, kwargs):
        self.recorder = recorder
        self.prim_paths = prim_paths
        self.attribute_name = attribute_name
        self.kwargs = kwargs
        self.unbound = False

    def write_async(self, tensor, **kwargs):
        self.recorder.events.append(f"write:{self.attribute_name}")
        completion = _Completion(lambda: self.recorder.events.append(f"wait:{self.attribute_name}"))
        self.recorder.writes.append((self, tensor, kwargs, completion))
        return completion

    def unbind(self):
        self.unbound = True


class _OvstageRecorder:
    def __init__(self, events: list[str]):
        self.events = events
        self.writes = []
        self.completions = []
        self.released = []

    def query_from_path_list(self, path_list):
        return f"query:{path_list}"

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


def _renderer(*, use_ovstage: bool = False):
    events: list[str] = []
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer._initialized_scene = True
    renderer._use_ovstage = use_ovstage
    renderer._renderer = _NativeRecorder(events)
    renderer._visual_material_writer_ref = None
    if use_ovstage:
        renderer._stage = _OvstageRecorder(events)
        renderer._stage_paths = _PathRecorder()
        renderer._current_ordinal = 7
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_legacy_compiles_typed_bindings_and_publishes_selected_channels_zero_copy():
    renderer, _events = _renderer()
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device="cuda")
    roughness = torch.tensor([0.1, 0.9], device="cuda")
    texture_scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    writer = renderer.visual_material_writer(
        (
            _batch("color", ("diffuse_color_constant", "diffuse_color_constant", "diffuseColor"), colors),
            _batch("roughness", ("roughness", "roughness"), roughness),
            _batch("texture_scale", ("texture_scale", "texture_scale"), texture_scale),
        )
    )

    writer(
        {"texture_scale": torch.tensor([0], dtype=torch.int32, device="cuda")},
        torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    assert renderer._renderer.writes == []
    assert len(renderer._renderer.bindings) == 4
    writer.publish()

    writes = {write[0].attribute_name: write for write in renderer._renderer.writes}
    assert set(writes) == {"inputs:texture_scale"}
    write = writes["inputs:texture_scale"]
    assert write[1].untyped_storage().data_ptr() == texture_scale.untyped_storage().data_ptr()
    assert write[2]["data_access"] is DataAccess.ASYNC
    assert write[2]["cuda_event"] == writer._event.cuda_event
    writer.drain()
    assert all(write[3].wait_count == 1 for write in renderer._renderer.writes)


@pytest.mark.skipif(importlib.util.find_spec("ovstage") is None, reason="requires optional module: ovstage")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ovstage_compiles_queries_and_publishes_selected_channel_zero_copy():
    renderer, _events = _renderer(use_ovstage=True)
    roughness = torch.tensor([0.2, 0.8], device="cuda")
    writer = renderer.visual_material_writer((_batch("roughness", ("roughness", "roughness"), roughness),))
    writer(
        {"roughness": torch.tensor([0], dtype=torch.int32, device="cuda")},
        torch.tensor([1], dtype=torch.int32, device="cuda"),
    )

    assert renderer._stage.writes == []
    writer.publish()

    assert len(renderer._stage.writes) == 1
    query, attribute_name, kwargs, completion = renderer._stage.writes[0]
    assert query == f"query:{renderer._stage_paths.created[0]}"
    assert attribute_name == "inputs:roughness"
    assert kwargs["tensors"].untyped_storage().data_ptr() == roughness.untyped_storage().data_ptr()
    assert kwargs["ordinal"] == 7
    assert kwargs["is_array"] is False
    assert kwargs["cuda_event"] == writer._event.cuda_event
    writer.drain()
    assert completion.wait_count == 1


@pytest.mark.parametrize(
    ("use_ovstage", "expected_events"),
    (
        (False, ["publish", "step", "drain"]),
        (True, ["publish", "floor", "wait:floor", "drain", "step"]),
    ),
)
def test_render_publishes_and_drains_material_writes_at_backend_boundary(use_ovstage, expected_events):
    renderer, events = _renderer(use_ovstage=use_ovstage)
    renderer._render_product_paths = ["/Render/Product"]

    class Writer:
        def publish(self):
            events.append("publish")

        def drain(self):
            events.append("drain")

    writer = Writer()
    renderer._visual_material_writer_ref = lambda: writer
    render = renderer._render_ovstage if use_ovstage else renderer._render_legacy
    render(SimpleNamespace(ppisp_pipeline=None))

    assert events == expected_events


@pytest.mark.parametrize(
    ("failure", "expected_events"),
    (("publish", ["publish", "drain"]), ("floor", ["publish", "floor", "drain"])),
)
def test_ovstage_drain_does_not_mask_publish_or_floor_failure(failure, expected_events):
    renderer, events = _renderer(use_ovstage=True)
    renderer._render_product_paths = ["/Render/Product"]

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

    renderer._stage.advance_write_floor = advance_write_floor
    with pytest.raises(ValueError, match=failure):
        renderer._render_ovstage(SimpleNamespace(ppisp_pipeline=None))

    assert events == expected_events


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_writer_close_drains_and_unbinds_compiled_legacy_addresses():
    renderer, _events = _renderer()
    writer = renderer.visual_material_writer((_batch("roughness", ("roughness",), torch.zeros(1, device="cuda")),))
    writer(None)
    writer.publish()
    writer.close()

    assert all(binding.unbound for binding in renderer._renderer.bindings)
    assert all(write[3].wait_count == 1 for write in renderer._renderer.writes)
    assert writer._addresses == []
    assert writer._buffers == {}


@pytest.mark.parametrize("values", [torch.zeros(1, dtype=torch.float64), torch.zeros(1, 4)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compilation_rejects_unsupported_device_buffers_before_backend_mutation(values):
    renderer, _events = _renderer()
    valid = _batch("color", ("diffuseColor",), torch.zeros(1, 3, device="cuda"))
    invalid = _batch("unsupported", ("value",), values.to(device="cuda"))

    with pytest.raises(TypeError, match="float, float2, or float3"):
        renderer.visual_material_writer((valid, invalid))

    assert renderer._visual_material_writer_ref is None
    assert renderer._renderer.bindings == []


def test_compilation_rejects_host_buffers_without_fallback():
    renderer, _events = _renderer()
    with pytest.raises(RuntimeError, match="one CUDA device"):
        renderer.visual_material_writer((_batch("color", ("diffuseColor",), torch.zeros(1, 3)),))

    assert renderer._visual_material_writer_ref is None
    assert renderer._renderer.bindings == []


def test_writer_factory_requires_ingested_detached_scene():
    renderer, _events = _renderer()
    renderer._initialized_scene = False
    with pytest.raises(RuntimeError, match="ingest its detached scene"):
        renderer.visual_material_writer((_batch("color", ("diffuseColor",), torch.zeros(1, 3)),))


def test_material_runtime_has_no_host_or_usd_path():
    source = inspect.getsource(OVRTXVisualMaterialWriter)
    for forbidden in (".cpu(", ".numpy(", ".tolist(", "pxr", "Usd", "Sdf"):
        assert forbidden not in source
