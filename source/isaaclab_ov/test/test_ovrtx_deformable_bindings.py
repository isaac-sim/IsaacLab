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
else:
    NewtonManager = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None


class _FakeArrayBinding:
    """Capture array writes made through an OVRTX binding."""

    def __init__(self, attribute_name: str):
        self.attribute_name = attribute_name
        self.written = None
        self.unbound = False

    def write(self, data, **kwargs):  # noqa: ARG002
        self.written = data

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
        binding = _FakeArrayBinding(kwargs["attribute_name"])
        self.bindings[kwargs["attribute_name"]] = binding
        return binding

    def bind_attribute(self, **kwargs):
        self.calls.append(kwargs)
        binding = _FakeArrayBinding(kwargs["attribute_name"])
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
    renderer._deformable_extent_buffer = None
    renderer._deformable_particle_offsets = None
    renderer._deformable_particles_per_body = None
    renderer._deformable_max_particles_per_body = 0
    renderer._deformable_extent_mins = []
    renderer._deformable_extent_maxs = []
    return renderer, renderer._renderer


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
    assert renderer._deformable_points_buffers == []
    assert renderer._deformable_extent_buffer is None
    assert renderer._deformable_vis_mesh_prim_paths == ["/World/envs/env_0/Deformable/mesh"]
    assert renderer._deformable_particle_offsets.numpy().tolist() == [7]
    assert renderer._deformable_particles_per_body.numpy().tolist() == [3]


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
    assert renderer._deformable_particle_offsets.numpy().tolist() == [7]


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
    assert renderer._deformable_particle_offsets.numpy().tolist() == [0, 3, 6, 9]
    assert renderer._deformable_points_buffers == []


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
    assert renderer._deformable_vis_mesh_prim_paths == expected_paths
    assert renderer._deformable_particle_offsets.numpy().tolist() == [0, 3, 6, 9]
    assert renderer._deformable_points_buffers == []


def test_ensure_deformable_update_buffers_allocates_once():
    """Update buffers are allocated lazily from setup metadata and only once."""
    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_vis_mesh_prim_paths = ["/World/envs/env_0/Deformable/mesh"]
    renderer._deformable_particle_offsets = wp.array([7], dtype=wp.int32, device="cpu")
    renderer._deformable_particles_per_body = wp.array([3], dtype=wp.int32, device="cpu")

    renderer._ensure_deformable_update_buffers()
    first_point_buffers = renderer._deformable_points_buffers
    first_extent_buffer = renderer._deformable_extent_buffer

    assert len(first_point_buffers) == 1
    assert first_point_buffers[0].shape == (3,)
    assert first_extent_buffer.shape == (1, 2, 3)
    assert renderer._deformable_particle_offsets.numpy().tolist() == [7]
    assert renderer._deformable_particles_per_body.numpy().tolist() == [3]
    assert renderer._deformable_max_particles_per_body == 3
    assert len(renderer._deformable_extent_mins) == 1
    assert len(renderer._deformable_extent_maxs) == 1

    renderer._ensure_deformable_update_buffers()
    assert renderer._deformable_points_buffers is first_point_buffers
    assert renderer._deformable_extent_buffer is first_extent_buffer


def test_update_deformable_points_allocates_buffers_on_first_update(monkeypatch: pytest.MonkeyPatch):
    """First dirty update allocates sync buffers from setup metadata."""
    renderer, backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakeArrayBinding("points")
    renderer._deformable_extent_binding = _FakeArrayBinding("extent")
    renderer._deformable_vis_mesh_prim_paths = ["/World/envs/env_0/Deformable/mesh"]
    renderer._deformable_particle_offsets = wp.array([0], dtype=wp.int32, device="cpu")
    renderer._deformable_particles_per_body = wp.array([3], dtype=wp.int32, device="cpu")

    particle_q = wp.array(
        [wp.vec3f(0.0, 0.0, 0.0), wp.vec3f(1.0, 0.0, 0.0), wp.vec3f(2.0, 0.0, 0.0)],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    NewtonManager._particles_dirty = True
    monkeypatch.setattr(NewtonManager, "particles_dirty", classmethod(lambda cls: cls._particles_dirty))
    monkeypatch.setattr(
        NewtonManager, "clear_particles_dirty", classmethod(lambda cls: setattr(cls, "_particles_dirty", False))
    )

    launch_calls: list[tuple] = []

    def _fake_launch(kernel, dim, inputs, device):  # noqa: ARG001
        launch_calls.append((kernel, dim, inputs))
        if getattr(kernel, "__name__", "") == "sync_newton_deformable_points_batched_kernel":
            stacked_points, particle_positions, particle_offsets, particles_per_mesh = inputs
            mesh_index = 0
            offset = int(particle_offsets.numpy()[mesh_index])
            count = int(particles_per_mesh.numpy()[mesh_index])
            stacked_points.numpy()[mesh_index, :count] = particle_positions.numpy()[offset : offset + count]
            return
        if getattr(kernel, "__name__", "") == "compute_deformable_mesh_extent_kernel":
            point_buffer, extent_min, extent_max = inputs
            points = point_buffer.numpy()
            extent_min.numpy()[0] = points.min(axis=0)
            extent_max.numpy()[0] = points.max(axis=0)

    monkeypatch.setattr(ovrtx_renderer_module.wp, "launch", _fake_launch)

    renderer._update_deformable_points()

    assert len(renderer._deformable_points_buffers) == 1
    assert backend.writes == []
    assert any(
        getattr(call[0], "__name__", "") == "sync_newton_deformable_points_batched_kernel" for call in launch_calls
    )


def test_update_deformable_points_writes_world_particle_positions(monkeypatch: pytest.MonkeyPatch):
    """Newton ``particle_q`` is copied into OVRTX point buffers before binding writes."""
    renderer, backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakeArrayBinding("points")
    renderer._deformable_extent_binding = _FakeArrayBinding("extent")
    renderer._deformable_points_buffers = [wp.empty(3, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_buffer = np.zeros((1, 2, 3), dtype=np.float64)
    renderer._deformable_particle_offsets = wp.array([1], dtype=wp.int32, device="cpu")
    renderer._deformable_particles_per_body = wp.array([3], dtype=wp.int32, device="cpu")
    renderer._deformable_max_particles_per_body = 3
    renderer._deformable_extent_mins = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_maxs = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
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
        if getattr(kernel, "__name__", "") == "sync_newton_deformable_points_batched_kernel":
            stacked_points, particle_positions, particle_offsets, particles_per_mesh = inputs
            mesh_index = 0
            offset = int(particle_offsets.numpy()[mesh_index])
            count = int(particles_per_mesh.numpy()[mesh_index])
            stacked_points.numpy()[mesh_index, :count] = particle_positions.numpy()[offset : offset + count]
            return
        if getattr(kernel, "__name__", "") == "compute_deformable_mesh_extent_kernel":
            point_buffer, extent_min, extent_max = inputs
            points = point_buffer.numpy()
            extent_min.numpy()[0] = points.min(axis=0)
            extent_max.numpy()[0] = points.max(axis=0)

    monkeypatch.setattr(ovrtx_renderer_module.wp, "launch", _fake_launch)

    renderer._update_deformable_points()

    assert any(
        getattr(call[0], "__name__", "") == "sync_newton_deformable_points_batched_kernel" for call in launch_calls
    )
    assert NewtonManager._particles_dirty is False

    assert renderer._deformable_extent_binding.written is renderer._deformable_extent_buffer
    assert renderer._deformable_extent_buffer[0].tolist() == [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]
    assert renderer._deformable_points_binding.written is renderer._deformable_points_buffers
    assert renderer._deformable_points_buffers[0].numpy().tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]


def test_update_deformable_points_skips_when_particles_clean(monkeypatch: pytest.MonkeyPatch):
    """Dirty-gated sync skips kernel launches when particle state is clean."""
    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakeArrayBinding("points")
    renderer._deformable_extent_binding = _FakeArrayBinding("extent")
    renderer._deformable_points_buffers = [wp.empty(3, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_buffer = np.zeros((1, 2, 3), dtype=np.float64)
    renderer._deformable_particle_offsets = wp.array([0], dtype=wp.int32, device="cpu")
    renderer._deformable_particles_per_body = wp.array([3], dtype=wp.int32, device="cpu")
    renderer._deformable_max_particles_per_body = 3
    renderer._deformable_extent_mins = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_maxs = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]

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
