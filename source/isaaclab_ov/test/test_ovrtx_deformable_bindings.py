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
    from ovrtx import BindingFlag, DataAccess  # noqa: E402
else:
    NewtonManager = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None
    BindingFlag = None
    DataAccess = None


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
    renderer._deformable_particle_offsets = []
    renderer._deformable_particle_counts = []
    renderer._particle_points_binding = None
    renderer._particle_visual_offsets = []
    renderer._particle_visual_counts = []
    renderer._particle_workaround_applied = False
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

    renderer._setup_deformable_bindings(num_envs=1)

    assert len(backend.calls) == 1
    assert backend.calls[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.calls[0]["attribute_name"] == "points"
    assert backend.calls[0]["dtype"] is np.float32
    assert backend.calls[0]["shape"] == (3,)
    assert renderer._deformable_points_binding is backend.bindings["points"]
    assert len(backend.writes) == 2
    assert backend.writes[0]["attribute_name"] == "omni:resetXformStack"
    assert backend.writes[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.writes[1]["attribute_name"] == "omni:xform"
    assert backend.writes[1]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert len(renderer._deformable_particle_counts) == 1
    assert renderer._deformable_particle_counts[0] == 3
    assert renderer._deformable_particle_offsets == [7]


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

    renderer._setup_deformable_bindings(num_envs=1)

    assert len(backend.calls) == 1
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

    renderer._setup_deformable_bindings(num_envs=2)

    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/DeformableSurface/mesh",
        "/World/envs/env_1/DeformableSurface/mesh",
        "/World/envs/env_0/DeformableVolume/mesh",
        "/World/envs/env_1/DeformableVolume/mesh",
    ]
    assert renderer._deformable_particle_offsets == [0, 3, 6, 9]
    assert renderer._deformable_particle_counts == [3, 3, 3, 3]


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

    renderer._setup_deformable_bindings(num_envs=1)

    assert len(backend.calls) == 1
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

    renderer._setup_deformable_bindings(num_envs=4)

    expected_paths = [f"/World/envs/env_{i}/Deformable/mesh" for i in range(4)]
    assert backend.calls[0]["prim_paths"] == expected_paths
    assert renderer._deformable_particle_offsets == [0, 3, 6, 9]
    assert renderer._deformable_particle_counts == [3, 3, 3, 3]


def test_update_deformable_points_writes_world_particle_positions(monkeypatch: pytest.MonkeyPatch):
    """Newton ``particle_q`` slices are handed to OVRTX through :meth:`OVRTXRenderer.update_geometries`."""
    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakePointsBinding("points")
    renderer._deformable_particle_offsets = [1]
    renderer._deformable_particle_counts = [3]
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

    class _FakeStream:
        cuda_stream = 42

    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_stream", lambda device: _FakeStream())  # noqa: ARG005

    renderer.update_geometries()

    written = renderer._deformable_points_binding.written
    assert written is not None
    assert len(written) == 1
    assert written[0].ptr == particle_q[1:4].ptr
    assert renderer._deformable_points_binding.write_kwargs is not None
    assert renderer._deformable_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._deformable_points_binding.write_kwargs["cuda_stream"] == 42
    assert written[0].numpy().tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]


def test_setup_deformable_bindings_rejects_offset_count_mismatch(monkeypatch: pytest.MonkeyPatch):
    """Registry entries must provide one particle offset per environment, listing every bad entry."""
    renderer, _backend = _make_renderer_without_backend()
    bad_entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0],
        particles_per_body=3,
    )
    other_bad_entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/DeformableOther",
        vis_mesh_prim_path="/World/envs/env_.*/DeformableOther/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3, 6],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [bad_entry, other_bad_entry])

    with pytest.raises(RuntimeError, match="one particle offset per environment") as excinfo:
        renderer._setup_deformable_bindings(num_envs=2)

    message = str(excinfo.value)
    assert bad_entry.prim_path in message
    assert other_bad_entry.prim_path in message


def test_update_geometries_rejects_inconsistent_deformable_mapping(monkeypatch: pytest.MonkeyPatch):
    """Geometry sync fails fast when offset and count metadata drift out of alignment."""
    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakePointsBinding("points")
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [3, 3]
    particle_q = wp.array(
        [wp.vec3f(float(i), 0.0, 0.0) for i in range(4)],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))

    with pytest.raises(ValueError, match="zip"):
        renderer.update_geometries()


def test_setup_particle_points_bindings_binds_mpm_visual_prims(monkeypatch: pytest.MonkeyPatch):
    """MPM particle visual prims create an OPTIMIZE ``points`` array binding."""
    renderer, backend = _make_renderer_without_backend()
    particle_visual_prims = {
        "/World/envs/env_0/Media/Particles": SimpleNamespace(offset=10, count=5),
        "/World/envs/env_1/Media/Particles": SimpleNamespace(offset=15, count=5),
    }

    monkeypatch.setattr(NewtonManager, "_particle_visual_prims", particle_visual_prims)

    renderer._setup_particle_bindings()

    assert len(backend.calls) == 1
    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/Media/Particles",
        "/World/envs/env_1/Media/Particles",
    ]
    assert backend.calls[0]["attribute_name"] == "points"
    assert backend.calls[0]["flags"] is BindingFlag.OPTIMIZE
    assert renderer._particle_points_binding is backend.bindings["points"]
    assert renderer._particle_workaround_applied is False
    assert renderer._particle_visual_offsets == [10, 15]
    assert renderer._particle_visual_counts == [5, 5]
    assert len(backend.writes) == 2
    assert backend.writes[0]["attribute_name"] == "omni:resetXformStack"
    assert backend.writes[1]["attribute_name"] == "omni:xform"


def test_setup_particle_points_bindings_binds_multiple_mpm_assets(monkeypatch: pytest.MonkeyPatch):
    """Multiple MPM assets bind as ``num_assets * num_envs`` points prims, like deformables."""
    renderer, backend = _make_renderer_without_backend()
    particle_visual_prims = {
        "/World/envs/env_0/Media/Particles": SimpleNamespace(offset=0, count=5),
        "/World/envs/env_1/Media/Particles": SimpleNamespace(offset=5, count=5),
        "/World/envs/env_0/Foam/Particles": SimpleNamespace(offset=10, count=3),
        "/World/envs/env_1/Foam/Particles": SimpleNamespace(offset=13, count=3),
    }

    monkeypatch.setattr(NewtonManager, "_particle_visual_prims", particle_visual_prims)

    renderer._setup_particle_bindings()

    # Binding order follows dict insertion order (no path sort).
    assert backend.calls[0]["prim_paths"] == [
        "/World/envs/env_0/Media/Particles",
        "/World/envs/env_1/Media/Particles",
        "/World/envs/env_0/Foam/Particles",
        "/World/envs/env_1/Foam/Particles",
    ]
    assert renderer._particle_visual_offsets == [0, 5, 10, 13]
    assert renderer._particle_visual_counts == [5, 5, 3, 3]


def test_update_particle_points_primes_with_host_sync_then_gpu_async(monkeypatch: pytest.MonkeyPatch):
    """First MPM ``points`` update host-SYNC primes via binding; later frames use GPU ASYNC."""
    renderer, backend = _make_renderer_without_backend()
    renderer._particle_points_binding = _FakePointsBinding("points")
    renderer._particle_visual_offsets = [2]
    renderer._particle_visual_counts = [2]
    renderer._particle_workaround_applied = False
    particle_q = wp.array(
        [
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(2.0, 3.0, 4.0),
            wp.vec3f(5.0, 6.0, 7.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))

    class _FakeStream:
        cuda_stream = 42

    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_stream", lambda device: _FakeStream())  # noqa: ARG005

    renderer.update_geometries()

    assert renderer._particle_workaround_applied is True
    assert len(backend.writes) == 0
    written = renderer._particle_points_binding.written
    assert written is not None
    assert len(written) == 1
    assert isinstance(written[0], np.ndarray)
    assert written[0].tolist() == [
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
    ]
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.SYNC
    assert "cuda_stream" not in renderer._particle_points_binding.write_kwargs

    renderer.update_geometries()

    written = renderer._particle_points_binding.written
    assert written is not None
    assert len(written) == 1
    assert written[0].ptr == particle_q[2:4].ptr
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert renderer._particle_points_binding.write_kwargs["cuda_stream"] == 42
    assert len(backend.writes) == 0


def test_update_geometries_writes_deformable_and_mpm_bindings(monkeypatch: pytest.MonkeyPatch):
    """Deformable mesh stays GPU ASYNC; MPM primes once with host SYNC then switches to ASYNC."""
    renderer, backend = _make_renderer_without_backend()
    renderer._deformable_points_binding = _FakePointsBinding("deformable_points")
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [2]
    renderer._particle_points_binding = _FakePointsBinding("points")
    renderer._particle_visual_offsets = [2]
    renderer._particle_visual_counts = [2]
    renderer._particle_workaround_applied = False
    particle_q = wp.array(
        [
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(2.0, 3.0, 4.0),
            wp.vec3f(5.0, 6.0, 7.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))

    class _FakeStream:
        cuda_stream = 42

    monkeypatch.setattr(ovrtx_renderer_module.wp, "get_stream", lambda device: _FakeStream())  # noqa: ARG005

    renderer.update_geometries()

    deformable_written = renderer._deformable_points_binding.written
    assert deformable_written is not None
    assert len(deformable_written) == 1
    assert deformable_written[0].ptr == particle_q[0:2].ptr
    assert renderer._deformable_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC

    assert renderer._particle_workaround_applied is True
    assert len(backend.writes) == 0
    mpm_written = renderer._particle_points_binding.written
    assert mpm_written is not None
    assert isinstance(mpm_written[0], np.ndarray)
    assert mpm_written[0].tolist() == [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.SYNC

    renderer.update_geometries()

    mpm_written = renderer._particle_points_binding.written
    assert mpm_written is not None
    assert len(mpm_written) == 1
    assert mpm_written[0].ptr == particle_q[2:4].ptr
    assert renderer._particle_points_binding.write_kwargs["data_access"] is DataAccess.ASYNC
    assert len(backend.writes) == 0
