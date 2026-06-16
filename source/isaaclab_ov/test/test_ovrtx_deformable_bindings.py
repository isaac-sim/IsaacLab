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
    renderer._deformable_original_extents = []
    renderer._deformable_particle_offsets = None
    renderer._deformable_inverse_world_matrices = None
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


def test_prepare_stage_deactivates_deformable_sim_meshes_for_export(monkeypatch: pytest.MonkeyPatch):
    """prepare_stage deactivates Newton deformable sim meshes only in the OVRTX USD export."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Deformable")
    UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Deformable/sim_mesh")
    UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Deformable/visual_mesh")

    renderer, _backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        sim_mesh_prim_path="/World/envs/env_.*/Deformable/sim_mesh",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/visual_mesh",
    )
    mock_ctx = SimpleNamespace(get_clone_plan=lambda: None)
    monkeypatch.setattr(
        "isaaclab_ov.renderers.ovrtx_renderer.SimulationContext",
        SimpleNamespace(instance=lambda: mock_ctx),
    )
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer.prepare_stage(stage, 1)

    exported_stage = Usd.Stage.CreateInMemory()
    exported_stage.GetRootLayer().ImportFromString(renderer._exported_usd_string)
    assert not exported_stage.GetPrimAtPath("/World/envs/env_0/Deformable/sim_mesh").IsValid()
    assert exported_stage.GetPrimAtPath("/World/envs/env_0/Deformable/visual_mesh").IsActive() is True
    assert stage.GetPrimAtPath("/World/envs/env_0/Deformable/sim_mesh").IsActive() is True


def test_prepare_stage_keeps_sim_mesh_when_registry_paths_match(monkeypatch: pytest.MonkeyPatch):
    """prepare_stage skips sim mesh deactivation when registry sim and visual paths match."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Deformable")
    UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Deformable/visual_mesh")
    UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Deformable/sim_mesh")

    renderer, _backend = _make_renderer_without_backend()
    entry = SimpleNamespace(
        prim_path="/World/envs/env_.*/Deformable",
        sim_mesh_prim_path="/World/envs/env_.*/Deformable/visual_mesh",
        vis_mesh_prim_path="/World/envs/env_.*/Deformable/visual_mesh",
    )
    mock_ctx = SimpleNamespace(get_clone_plan=lambda: None)
    monkeypatch.setattr(
        "isaaclab_ov.renderers.ovrtx_renderer.SimulationContext",
        SimpleNamespace(instance=lambda: mock_ctx),
    )
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])

    renderer.prepare_stage(stage, 1)

    exported_stage = Usd.Stage.CreateInMemory()
    exported_stage.GetRootLayer().ImportFromString(renderer._exported_usd_string)
    assert exported_stage.GetPrimAtPath("/World/envs/env_0/Deformable/sim_mesh").IsActive() is True
    assert exported_stage.GetPrimAtPath("/World/envs/env_0/Deformable/visual_mesh").IsActive() is True
    assert stage.GetPrimAtPath("/World/envs/env_0/Deformable/sim_mesh").IsActive() is True


def test_update_deformable_points_writes_local_particle_positions(monkeypatch: pytest.MonkeyPatch):
    """Newton ``particle_q`` is copied into OVRTX point buffers before binding writes."""
    renderer, backend = _make_renderer_without_backend()
    renderer._deformable_point_binding = _FakeArrayBinding("points")
    renderer._deformable_extent_binding = _FakeArrayBinding("extent")
    renderer._deformable_point_buffers = [wp.empty(3, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_buffer = np.zeros((1, 2, 3), dtype=np.float32)
    renderer._deformable_original_extents = [([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])]
    renderer._deformable_particle_offsets = wp.array([1], dtype=wp.int32, device="cpu")
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

    def _fake_launch(kernel, dim, inputs, device):  # noqa: ARG001
        point_buffer, particle_positions, particle_offsets, _inverse_world_matrices, mesh_index = inputs
        offset = int(particle_offsets.numpy()[mesh_index])
        point_buffer.numpy()[:] = particle_positions.numpy()[offset : offset + dim]

    monkeypatch.setattr(ovrtx_renderer_module.wp, "launch", _fake_launch)

    renderer._update_deformable_points()

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
    renderer._deformable_point_binding = _FakeArrayBinding("points")
    renderer._deformable_extent_binding = _FakeArrayBinding("extent")
    renderer._deformable_visual_mesh_paths = ["/World/envs/env_0/Deformable/mesh"]
    renderer._deformable_point_buffers = [wp.empty(3, dtype=wp.vec3f, device="cpu")]
    renderer._deformable_extent_buffer = np.zeros((1, 2, 3), dtype=np.float64)
    renderer._deformable_particle_offsets = wp.array([1], dtype=wp.int32, device="cpu")
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

    def _fake_launch(kernel, dim, inputs, device):  # noqa: ARG001
        point_buffer, particle_positions, particle_offsets, inverse_world_matrices, mesh_index = inputs
        offset = int(particle_offsets.numpy()[mesh_index])
        points = particle_positions.numpy()[offset : offset + dim]
        inverse_matrix = inverse_world_matrices.numpy()[mesh_index]
        translation = inverse_matrix[:3, 3]
        point_buffer.numpy()[:] = points + translation

    monkeypatch.setattr(ovrtx_renderer_module.wp, "launch", _fake_launch)

    renderer._update_deformable_points()

    np.testing.assert_allclose(
        renderer._deformable_point_buffers[0].numpy(),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    assert renderer._deformable_extent_buffer[0].tolist() == [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]
