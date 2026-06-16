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

    from pxr import Gf, Usd, UsdGeom, Vt  # noqa: E402
else:
    Gf = None
    NewtonManager = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None
    Usd = None
    UsdGeom = None
    Vt = None


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
    renderer._deformable_point_binding = None
    renderer._deformable_extent_binding = None
    renderer._deformable_visual_mesh_paths = []
    renderer._deformable_point_buffers = []
    renderer._deformable_extent_buffer = None
    renderer._deformable_particle_offsets = None
    renderer._deformable_inverse_world_matrices = None
    renderer._deformable_particles_per_mesh = None
    renderer._deformable_max_particles_per_mesh = 0
    renderer._deformable_extent_mins = []
    renderer._deformable_extent_maxs = []
    renderer._deformable_bindings_ready = False
    renderer._deformable_bindings_warned = False
    return renderer, renderer._renderer


def _make_stage_with_surface_mesh(num_points: int = 3) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Deformable/mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(i), 0.0, 0.0) for i in range(num_points)]))
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    return stage


def _make_stage_with_surface_mesh_envs(num_envs: int, num_points: int = 3) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    for env_idx in range(num_envs):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_idx}")
        mesh = UsdGeom.Mesh.Define(stage, f"/World/envs/env_{env_idx}/Deformable/mesh")
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(i), 0.0, 0.0) for i in range(num_points)]))
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    return stage


def test_setup_deformable_mesh_bindings_binds_surface_mesh_points(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries create OVRTX ``points`` array bindings."""
    stage = _make_stage_with_surface_mesh()
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[7],
        particles_per_body=3,
    )

    monkeypatch.setattr("isaaclab.sim.utils.stage.get_current_stage", lambda: stage)
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_mesh_bindings()

    assert len(backend.calls) == 2
    assert backend.calls[0]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.calls[0]["attribute_name"] == "points"
    assert backend.calls[0]["dtype"] is np.float32
    assert backend.calls[0]["shape"] == (3,)
    assert backend.calls[1]["prim_paths"] == ["/World/envs/env_0/Deformable/mesh"]
    assert backend.calls[1]["attribute_name"] == "extent"
    assert backend.calls[1]["dtype"] is np.float64
    assert backend.calls[1]["shape"] == (2, 3)
    assert renderer._deformable_point_binding is backend.bindings["points"]
    assert renderer._deformable_extent_binding is backend.bindings["extent"]
    assert backend.writes == []
    assert renderer._deformable_point_buffers[0].shape == (3,)
    assert renderer._deformable_extent_buffer.tolist() == [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    assert renderer._deformable_visual_mesh_paths == ["/World/envs/env_0/Deformable/mesh"]
    assert renderer._deformable_particle_offsets.numpy().tolist() == [7]
    assert renderer._deformable_bindings_ready is True
    assert NewtonManager.particles_dirty() is True


def test_resolve_deformable_instance_path_env_regex():
    """Registry env regex paths resolve to concrete instance paths."""
    from isaaclab_ov.renderers.ovrtx_renderer import _resolve_deformable_instance_path

    assert (
        _resolve_deformable_instance_path("/World/envs/env_.*/Deformable/mesh", 2)
        == "/World/envs/env_2/Deformable/mesh"
    )


def test_resolve_deformable_instance_path_plain_regex():
    """Plain ``.*`` segments resolve to the instance index."""
    from isaaclab_ov.renderers.ovrtx_renderer import _resolve_deformable_instance_path

    assert _resolve_deformable_instance_path("/World/envs/env_0/item_.*", 4) == "/World/envs/env_0/item_4"


def test_setup_skips_volume_deformable_with_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """Volume-only deformable registry entries are skipped with a warning."""
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="volume",
        particle_offsets=[0],
        particles_per_body=3,
    )

    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    with caplog.at_level("WARNING"):
        renderer._setup_deformable_mesh_bindings()

    assert backend.calls == []
    assert "surface deformables only" in caplog.text


def test_setup_warns_when_registry_nonempty_but_stage_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Surface deformables without a USD stage emit a warning instead of binding."""
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

    with caplog.at_level("WARNING"):
        renderer._setup_deformable_mesh_bindings()

    assert backend.calls == []
    assert "skipping OVRTX deformable mesh bindings" in caplog.text


def test_setup_deformable_mesh_bindings_binds_all_surface_mesh_instances(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries bind every cloned visual mesh instance."""
    stage = _make_stage_with_surface_mesh_envs(num_envs=4)
    renderer, backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0, 3, 6, 9],
        particles_per_body=3,
    )

    monkeypatch.setattr("isaaclab.sim.utils.stage.get_current_stage", lambda: stage)
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer._setup_deformable_mesh_bindings()

    expected_paths = [f"/World/envs/env_{i}/Deformable/mesh" for i in range(4)]
    assert backend.calls[0]["prim_paths"] == expected_paths
    assert renderer._deformable_visual_mesh_paths == expected_paths
    assert renderer._deformable_particle_offsets.numpy().tolist() == [0, 3, 6, 9]
    assert len(renderer._deformable_point_buffers) == 4


def test_setup_deformable_mesh_bindings_raises_on_point_count_mismatch(monkeypatch: pytest.MonkeyPatch):
    """Surface deformable registry entries must match visual mesh point counts."""
    stage = _make_stage_with_surface_mesh(num_points=2)
    renderer, _backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/mesh",
        deformable_type="surface",
        particle_offsets=[0],
        particles_per_body=3,
    )

    monkeypatch.setattr("isaaclab.sim.utils.stage.get_current_stage", lambda: stage)
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    with pytest.raises(RuntimeError, match="mesh has 2 points but Newton has 3 particles"):
        renderer._setup_deformable_mesh_bindings()


def test_update_deformable_points_writes_local_particle_positions(monkeypatch: pytest.MonkeyPatch):
    """Newton ``particle_q`` is copied into OVRTX point buffers before binding writes."""
    renderer, backend = _make_renderer_without_backend()
    renderer._deformable_bindings_ready = True
    renderer._deformable_point_binding = _FakeArrayBinding("points")
    renderer._deformable_extent_binding = _FakeArrayBinding("extent")
    renderer._deformable_point_buffers = [wp.empty(3, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_buffer = np.zeros((1, 2, 3), dtype=np.float64)
    renderer._deformable_particle_offsets = wp.array([1], dtype=wp.int32, device="cpu")
    renderer._deformable_particles_per_mesh = wp.array([3], dtype=wp.int32, device="cpu")
    renderer._deformable_max_particles_per_mesh = 3
    renderer._deformable_extent_mins = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_maxs = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
    identity = wp.mat44f(
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    renderer._deformable_inverse_world_matrices = wp.array([identity], dtype=wp.mat44f, device="cpu")
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
    monkeypatch.setattr(NewtonManager, "transforms_dirty", classmethod(lambda cls: False))

    def _clear_particles_dirty(cls) -> None:
        cls._particles_dirty = False

    monkeypatch.setattr(NewtonManager, "clear_particles_dirty", classmethod(_clear_particles_dirty))

    launch_calls: list[tuple] = []

    def _fake_launch(kernel, dim, inputs, device):  # noqa: ARG001
        launch_calls.append((kernel, dim, inputs))
        if getattr(kernel, "__name__", "") == "sync_newton_deformable_points_batched_kernel":
            stacked_points, particle_positions, particle_offsets, _inverse_world_matrices, particles_per_mesh = inputs
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
    assert renderer._deformable_point_binding.written is renderer._deformable_point_buffers
    assert renderer._deformable_point_buffers[0].numpy().tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]


def test_update_deformable_points_refreshes_visual_mesh_world_transform(monkeypatch: pytest.MonkeyPatch):
    """OVRTX point sync uses the visual mesh's current world transform."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    deformable_xform = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Deformable")
    deformable_xform.AddTranslateOp().Set(Gf.Vec3d(10.0, 20.0, 30.0))
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Deformable/mesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]))
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])

    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_bindings_ready = True
    renderer._deformable_point_binding = _FakeArrayBinding("points")
    renderer._deformable_extent_binding = _FakeArrayBinding("extent")
    renderer._deformable_visual_mesh_paths = ["/World/envs/env_0/Deformable/mesh"]
    renderer._deformable_point_buffers = [wp.empty(3, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_buffer = np.zeros((1, 2, 3), dtype=np.float64)
    renderer._deformable_particle_offsets = wp.array([1], dtype=wp.int32, device="cpu")
    renderer._deformable_particles_per_mesh = wp.array([3], dtype=wp.int32, device="cpu")
    renderer._deformable_max_particles_per_mesh = 3
    renderer._deformable_extent_mins = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_maxs = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
    identity = wp.mat44f(
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    renderer._deformable_inverse_world_matrices = wp.array([identity], dtype=wp.mat44f, device="cpu")
    particle_q = wp.array(
        [
            wp.vec3f(-1.0, -1.0, -1.0),
            wp.vec3f(11.0, 22.0, 33.0),
            wp.vec3f(14.0, 25.0, 36.0),
            wp.vec3f(17.0, 28.0, 39.0),
        ],
        dtype=wp.vec3f,
        device="cpu",
    )

    monkeypatch.setattr("isaaclab.sim.utils.stage.get_current_stage", lambda: stage)
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    NewtonManager._particles_dirty = True
    monkeypatch.setattr(NewtonManager, "particles_dirty", classmethod(lambda cls: cls._particles_dirty))
    monkeypatch.setattr(NewtonManager, "transforms_dirty", classmethod(lambda cls: True))
    monkeypatch.setattr(
        NewtonManager, "clear_particles_dirty", classmethod(lambda cls: setattr(cls, "_particles_dirty", False))
    )

    def _fake_launch(kernel, dim, inputs, device):  # noqa: ARG001
        if getattr(kernel, "__name__", "") == "sync_newton_deformable_points_batched_kernel":
            stacked_points, particle_positions, particle_offsets, inverse_world_matrices, particles_per_mesh = inputs
            mesh_index = 0
            offset = int(particle_offsets.numpy()[mesh_index])
            count = int(particles_per_mesh.numpy()[mesh_index])
            points = particle_positions.numpy()[offset : offset + count]
            inverse_matrix = inverse_world_matrices.numpy()[mesh_index]
            translation = inverse_matrix[:3, 3]
            stacked_points.numpy()[mesh_index, :count] = points + translation
            return
        if getattr(kernel, "__name__", "") == "compute_deformable_mesh_extent_kernel":
            point_buffer, extent_min, extent_max = inputs
            points = point_buffer.numpy()
            extent_min.numpy()[0] = points.min(axis=0)
            extent_max.numpy()[0] = points.max(axis=0)

    monkeypatch.setattr(ovrtx_renderer_module.wp, "launch", _fake_launch)

    renderer._update_deformable_points()

    np.testing.assert_allclose(
        renderer._deformable_point_buffers[0].numpy(),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    assert renderer._deformable_extent_buffer[0].tolist() == [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]


def test_update_deformable_points_skips_when_particles_clean(monkeypatch: pytest.MonkeyPatch):
    """Dirty-gated sync skips kernel launches when particle state is clean."""
    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_bindings_ready = True
    renderer._deformable_point_binding = _FakeArrayBinding("points")
    renderer._deformable_extent_binding = _FakeArrayBinding("extent")
    renderer._deformable_point_buffers = [wp.empty(3, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_buffer = np.zeros((1, 2, 3), dtype=np.float64)
    renderer._deformable_particle_offsets = wp.array([0], dtype=wp.int32, device="cpu")
    renderer._deformable_particles_per_mesh = wp.array([3], dtype=wp.int32, device="cpu")
    renderer._deformable_max_particles_per_mesh = 3
    renderer._deformable_extent_mins = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_maxs = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_inverse_world_matrices = wp.array(
        [wp.mat44f(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        dtype=wp.mat44f,
        device="cpu",
    )

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
    assert renderer._deformable_point_binding.written is None


def test_update_deformable_points_propagates_refresh_failure(monkeypatch: pytest.MonkeyPatch):
    """Disappeared visual mesh prims raise instead of being swallowed."""
    renderer, _backend = _make_renderer_without_backend()
    renderer._deformable_bindings_ready = True
    renderer._deformable_point_binding = _FakeArrayBinding("points")
    renderer._deformable_extent_binding = _FakeArrayBinding("extent")
    renderer._deformable_visual_mesh_paths = ["/World/envs/env_0/Deformable/mesh"]
    renderer._deformable_point_buffers = [wp.empty(3, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_buffer = np.zeros((1, 2, 3), dtype=np.float64)
    renderer._deformable_particle_offsets = wp.array([0], dtype=wp.int32, device="cpu")
    renderer._deformable_particles_per_mesh = wp.array([3], dtype=wp.int32, device="cpu")
    renderer._deformable_max_particles_per_mesh = 3
    renderer._deformable_extent_mins = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_maxs = [wp.zeros(1, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_inverse_world_matrices = wp.array(
        [wp.mat44f(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        dtype=wp.mat44f,
        device="cpu",
    )
    particle_q = wp.array([wp.vec3f(0.0, 0.0, 0.0)], dtype=wp.vec3f, device="cpu")

    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr("isaaclab.sim.utils.stage.get_current_stage", lambda: stage)
    monkeypatch.setattr(NewtonManager, "get_state", classmethod(lambda cls: SimpleNamespace(particle_q=particle_q)))
    NewtonManager._particles_dirty = True
    monkeypatch.setattr(NewtonManager, "particles_dirty", classmethod(lambda cls: cls._particles_dirty))
    monkeypatch.setattr(NewtonManager, "transforms_dirty", classmethod(lambda cls: True))

    with pytest.raises(RuntimeError, match="visual mesh disappeared"):
        renderer._update_deformable_points()
