# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX deformable mesh point bindings."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "pxr", "isaaclab_newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    import isaaclab_ov.renderers.ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_newton.physics import NewtonManager  # noqa: E402
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer  # noqa: E402
    from ovrtx import DataAccess  # noqa: E402
else:
    NewtonManager = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None


class _FakeAttributeMapping:
    """Minimal OVRTX attribute mapping context manager."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class _FakePointsBinding:
    """Capture array writes made through an OVRTX array attribute binding."""

    def __init__(self, attribute_name: str):
        self.attribute_name = attribute_name
        self.written = None
        self.write_kwargs: dict | None = None
        self.unbound = False

    def write(self, data, **kwargs):
        self.written = data
        self.write_kwargs = kwargs

    def map(self, device=None, device_id=0):  # noqa: ARG002
        raise RuntimeError("bind_array_attribute bindings do not expose mapped point buffers")

    def unbind(self):
        self.unbound = True


class _FakeExtentBinding:
    """Scalar OVRTX binding that exposes mapped extent memory."""

    def __init__(self, attribute_name: str, mapped_extents: wp.array):
        self.attribute_name = attribute_name
        self.mapped_extents = mapped_extents
        self.written = None
        self.map_calls: list[tuple] = []
        self.unbound = False

    def write(self, data, **kwargs):  # noqa: ARG002
        self.written = data

    def map(self, device=None, device_id=0):
        self.map_calls.append((device, device_id))
        return _FakeAttributeMapping(self.mapped_extents.__dlpack__())

    def unbind(self):
        self.unbound = True


class _FakeOVRTXBackend:
    """Minimal OVRTX backend stub for deformable binding setup."""

    def __init__(self):
        self.bindings = {}
        self.calls = []
        self.writes = []

    def bind_array_attribute(self, **kwargs):
        self.calls.append(kwargs)
        binding = _FakePointsBinding(kwargs["attribute_name"])
        self.bindings[kwargs["attribute_name"]] = binding
        return binding

    def bind_attribute(self, **kwargs):
        self.calls.append(kwargs)
        binding = _FakePointsBinding(kwargs["attribute_name"])
        self.bindings[kwargs["attribute_name"]] = binding
        return binding

    def query_prims(self, **kwargs):  # noqa: ARG002
        return {
            "/World/envs/env_0/Deformable/mesh": {},
            "/World/envs/env_0/Deformable/geometry/mesh": {},
        }

    def write_attribute(self, **kwargs):
        self.writes.append(kwargs)


def _make_renderer_without_backend(device: str = "cpu") -> tuple[OVRTXRenderer, _FakeOVRTXBackend]:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    renderer._device = device
    renderer._camera_rel_path = "Camera"
    renderer._renderer = _FakeOVRTXBackend()
    renderer._deformable_points_binding = None
    renderer._deformable_extent_binding = None
    renderer._deformable_vis_mesh_prim_paths = []
    renderer._deformable_points_buffers = []
    renderer._deformable_particle_offsets = []
    renderer._deformable_particles_per_body = []
    return renderer, renderer._renderer


def test_points_array_binding_uses_write_not_map():
    """OVRTX array bindings accept ``List[DLTensor]`` via ``write()``, not mapped tensors."""
    binding = _FakePointsBinding("points")
    with pytest.raises(RuntimeError, match="do not expose mapped point buffers"):
        binding.map()


def test_setup_deformable_bindings_binds_surface_mesh_points(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries create OVRTX ``points`` array bindings."""
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[7],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings()

    assert len(backend.calls) == 2
    assert backend.calls[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.calls[0]["attribute_name"] == "points"
    assert backend.calls[0]["dtype"] is np.float32
    assert backend.calls[0]["shape"] == (3,)
    assert backend.calls[1]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.calls[1]["attribute_name"] == "extent"
    assert backend.calls[1]["dtype"] is np.float64
    assert backend.calls[1]["shape"] == (2, 3)
    assert renderer._deformable_points_binding is backend.bindings["points"]
    assert renderer._deformable_extent_binding is backend.bindings["extent"]
    assert len(backend.writes) == 2
    assert backend.writes[0]["attribute_name"] == "omni:resetXformStack"
    assert backend.writes[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.writes[1]["attribute_name"] == "omni:xform"
    assert backend.writes[1]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert len(renderer._deformable_points_buffers) == 1
    assert renderer._deformable_points_buffers[0].shape == (3,)
    assert renderer._deformable_particle_offsets == [7]
    assert renderer._deformable_particles_per_body == [3]


def test_setup_deformable_bindings_binds_volume_mesh_points(monkeypatch: pytest.MonkeyPatch):
    """Volume deformable registry entries create OVRTX ``points`` bindings."""
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="volume",
        particle_offsets=[7],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings()

    assert len(backend.calls) == 2
    assert backend.calls[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.calls[0]["attribute_name"] == "points"
    assert renderer._deformable_points_binding is backend.bindings["points"]
    assert renderer._deformable_particle_offsets == [7]


def test_setup_deformable_bindings_binds_mixed_surface_and_volume_entries(monkeypatch: pytest.MonkeyPatch):
    """Surface and volume deformable registry entries bind together with distinct offsets."""
    renderer, backend = _make_renderer_without_backend()
    surface_entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/DeformableSurface",
        vis_mesh_prim_path="/World/envs/env_.*/DeformableSurface/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3],
        particles_per_body=3,
    )
    volume_entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/DeformableVolume",
        vis_mesh_prim_path="/World/envs/env_.*/DeformableVolume/mesh",
        deformable_type="volume",
        particle_offsets=[6, 9],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [surface_entry, volume_entry])

    renderer._setup_deformable_bindings()

    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/DeformableSurface/mesh",
        "/World/envs/env_1/DeformableSurface/mesh",
        "/World/envs/env_0/DeformableVolume/mesh",
        "/World/envs/env_1/DeformableVolume/mesh",
    ]
    assert renderer._deformable_particle_offsets == [0, 3, 6, 9]
    assert len(renderer._deformable_points_buffers) == 4


def test_setup_deformable_bindings_works_without_stage(monkeypatch: pytest.MonkeyPatch):
    """Deformable bindings are created from registry metadata without a USD stage."""
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0],
        particles_per_body=3,
    )

    monkeypatch.setattr("isaaclab.sim.utils.stage.get_current_stage", lambda: None)
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings()

    assert len(backend.calls) == 2
    assert backend.calls[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert renderer._deformable_points_binding is backend.bindings["points"]


def test_setup_deformable_bindings_binds_all_surface_mesh_instances(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries bind every cloned visual mesh instance."""
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3, 6, 9],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_bindings()

    expected_paths = [f"/World/envs/env_{i}/Deformable/mesh" for i in range(4)]
    assert backend.calls[0]["prim_paths"] == expected_paths
    assert renderer._deformable_particle_offsets == [0, 3, 6, 9]
    assert len(renderer._deformable_points_buffers) == 4


def test_update_deformable_points_writes_world_particle_positions(monkeypatch: pytest.MonkeyPatch):
    """Newton ``particle_q`` is copied into OVRTX point buffers and extents are mapped in place."""
    renderer, _backend = _make_renderer_without_backend()
    mapped_extents = wp.zeros((1, 2), dtype=wp.vec3d, device="cpu")
    renderer._deformable_points_binding = _FakePointsBinding("points")
    renderer._deformable_extent_binding = _FakeExtentBinding("extent", mapped_extents)
    renderer._deformable_points_buffers = [wp.empty(3, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_particle_offsets = [1]
    renderer._deformable_particles_per_body = [3]
    particle_q = wp.array(
        [
            wp.vec3f(-1.0, -1.0, -1.0),
            wp.vec3f(1.0, 2.0, 3.0),
            wp.vec3f(4.0, 5.0, 6.0),
            wp.vec3f(7.0, 8.0, 9.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    NewtonManager._particles_dirty = True
    monkeypatch.setattr(NewtonManager, "particles_dirty", classmethod(lambda cls: cls._particles_dirty))

    def _clear_particles_dirty(cls) -> None:
        cls._particles_dirty = False

    monkeypatch.setattr(NewtonManager, "clear_particles_dirty", classmethod(_clear_particles_dirty))

    launch_calls: list[tuple] = []

    def _fake_launch(kernel, dim, inputs, device):  # noqa: ARG001
        launch_calls.append((kernel, dim, inputs))
        if getattr(kernel, "__name__", "") == "compute_deformable_mesh_extent_kernel":
            point_buffer, extents, mesh_index = inputs
            points = point_buffer.numpy()
            extents.numpy()[mesh_index, 0] = points.min(axis=0)
            extents.numpy()[mesh_index, 1] = points.max(axis=0)

    class _FakeStream:
        cuda_stream = 42

    monkeypatch.setattr(ovrtx_renderer_module.wp, "launch", _fake_launch)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_stream", lambda device: _FakeStream())  # noqa: ARG005

    renderer._update_deformable_points()

    assert [getattr(call[0], "__name__", "") for call in launch_calls] == ["compute_deformable_mesh_extent_kernel"]
    assert NewtonManager._particles_dirty is False
    assert renderer._deformable_extent_binding.map_calls
    assert renderer._deformable_extent_binding.written is None
    assert mapped_extents.numpy().tolist() == [[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]]
    assert renderer._deformable_points_binding.written is renderer._deformable_points_buffers
    assert renderer._deformable_points_binding.write_kwargs is not None
    assert renderer._deformable_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._deformable_points_binding.write_kwargs["cuda_stream"] == 42
    assert renderer._deformable_points_buffers[0].numpy().tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]


def test_update_deformable_points_skips_when_particles_clean(monkeypatch: pytest.MonkeyPatch):
    """Dirty-gated sync skips data copies and kernel launches when particle state is clean."""
    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakePointsBinding("points")
    renderer._deformable_extent_binding = _FakeExtentBinding("extent", wp.zeros((1, 2), dtype=wp.vec3d, device="cpu"))
    renderer._deformable_points_buffers = [wp.empty(3, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particles_per_body = [3]

    NewtonManager._particles_dirty = False
    monkeypatch.setattr(NewtonManager, "particles_dirty", classmethod(lambda cls: cls._particles_dirty))
    launch_calls: list[tuple] = []
    monkeypatch.setattr(
        ovrtx_renderer_module.wp,
        "launch",
        lambda kernel, dim, inputs, device: launch_calls.append((kernel, dim, inputs)),  # noqa: ARG005
    )

    renderer._update_deformable_points()

    assert launch_calls == []
    assert renderer._deformable_points_binding.written is None
