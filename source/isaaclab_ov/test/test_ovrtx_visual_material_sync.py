# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for typed visual-material writes into OVRTX-owned scenes."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch
import warp as wp

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = pytest.mark.skipif(
    bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"
)

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer
    from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXRendererCfg
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
        return {path: SimpleNamespace(frames=[object()]) for path in kwargs["render_products"]}


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


def _renderer():
    events: list[str] = []
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.backend = SimpleNamespace(render_products=set())
    renderer._initialized_scene = True
    renderer.backend.renderer = _NativeRecorder(events)
    renderer._visual_material_writer_ref = None
    renderer.cfg = OVRTXRendererCfg()
    renderer._camera_render_data = []
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
    assert renderer.backend.renderer.writes == []
    assert len(renderer.backend.renderer.bindings) == 4
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.default_stream())):
        writer.publish()

    writes = {write[0].attribute_name: write for write in renderer.backend.renderer.writes}
    assert set(writes) == {"inputs:texture_scale"}
    write = writes["inputs:texture_scale"]
    assert write[1].untyped_storage().data_ptr() == texture_scale.untyped_storage().data_ptr()
    assert write[2]["data_access"] is DataAccess.ASYNC
    assert write[2]["cuda_event"] == writer._event.cuda_event
    assert write[2]["cuda_stream"] == 1
    writer.drain()
    assert all(write[3].wait_count == 1 for write in renderer.backend.renderer.writes)


def test_render_publishes_and_drains_material_writes_at_backend_boundary():
    renderer, events = _renderer()
    renderer._camera_render_data = [
        SimpleNamespace(render_product_path=path, ppisp_pipeline=None, warp_buffers={})
        for path in ("/Render/Product0", "/Render/Product1")
    ]
    renderer.backend.render_products = {camera.render_product_path for camera in renderer._camera_render_data}
    renderer._process_render_frame = lambda *args: None

    class Writer:
        def publish(self):
            events.append("publish")

        def drain(self):
            events.append("drain")

    writer = Writer()
    renderer._visual_material_writer_ref = lambda: writer
    renderer.render_batch(renderer._camera_render_data)

    assert events == ["publish", "step", "drain"]


def test_drain_does_not_mask_publish_failure():
    renderer, events = _renderer()
    renderer._camera_render_data = [SimpleNamespace(render_product_path="/RenderCamera_0/Product", ppisp_pipeline=None)]

    class Writer:
        def publish(self):
            events.append("publish")
            raise ValueError("publish failed")

        def drain(self):
            events.append("drain")
            raise RuntimeError("drain failed")

    writer = Writer()
    renderer._visual_material_writer_ref = lambda: writer

    with pytest.raises(ValueError, match="publish failed"):
        renderer.render(renderer._camera_render_data[0])

    assert events == ["publish", "drain"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_writer_close_drains_and_releases_compiled_addresses():
    renderer, _events = _renderer()
    writer = renderer.visual_material_writer((_batch("roughness", ("roughness",), torch.zeros(1, device="cuda")),))
    writer(None)
    writer.publish()
    writer.close()

    writer.close()
    assert all(binding.unbound for binding in renderer.backend.renderer.bindings)
    assert all(write[3].wait_count == 1 for write in renderer.backend.renderer.writes)


@pytest.mark.parametrize("values", [torch.zeros(1, dtype=torch.float64), torch.zeros(1, 4)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compilation_rejects_unsupported_device_buffers_before_backend_mutation(values):
    renderer, _events = _renderer()
    valid = _batch("color", ("diffuseColor",), torch.zeros(1, 3, device="cuda"))
    invalid = _batch("unsupported", ("value",), values.to(device="cuda"))

    with pytest.raises(TypeError, match="float, float2, or float3"):
        renderer.visual_material_writer((valid, invalid))

    assert renderer._visual_material_writer_ref is None
    assert renderer.backend.renderer.bindings == []


def test_compilation_rejects_host_buffers_without_fallback():
    renderer, _events = _renderer()
    with pytest.raises(RuntimeError, match="one CUDA device"):
        renderer.visual_material_writer((_batch("color", ("diffuseColor",), torch.zeros(1, 3)),))

    assert renderer._visual_material_writer_ref is None
    assert renderer.backend.renderer.bindings == []


def test_writer_factory_requires_ingested_detached_scene():
    renderer, _events = _renderer()
    renderer._initialized_scene = False
    assert renderer.visual_material_writer is None
    renderer._initialized_scene = True
    assert callable(renderer.visual_material_writer)
