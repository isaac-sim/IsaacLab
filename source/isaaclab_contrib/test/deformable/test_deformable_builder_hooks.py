# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sim.spawners.materials import NewtonDeformableMaterialCfg

from isaaclab.cloner import ClonePlan

from isaaclab_contrib.deformable import DeformableObject
from isaaclab_contrib.deformable.deformable_object import (
    DeformableRegistryEntry,
    add_deformable_entry_to_builder,
)


class _FakeBuilder:
    def __init__(self):
        self.particle_count = 0
        self.cloth_meshes = []

    def add_cloth_mesh(self, **kwargs) -> None:
        self.cloth_meshes.append(kwargs)
        self.particle_count += len(kwargs["vertices"])


def _make_surface_entry() -> DeformableRegistryEntry:
    half_sqrt = math.sqrt(0.5)
    return DeformableRegistryEntry(
        prim_path="{ENV_REGEX_NS}/cloth",
        sim_mesh_prim_path="{ENV_REGEX_NS}/cloth/mesh",
        vis_mesh_prim_path="{ENV_REGEX_NS}/cloth/mesh",
        vertices=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        indices=[0, 1, 2],
        init_pos=(1.0, 0.0, 0.0),
        init_rot=(0.0, 0.0, half_sqrt, half_sqrt),
        deformable_type="surface",
    )


def _vec3_as_tuple(value) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def test_deformable_package_exports_public_symbols():
    """Test that deformable symbols are exported from the package root."""
    assert DeformableObject.__name__ == "DeformableObject"


def test_newton_material_defaults_match_registry_defaults():
    """Test that Newton material cfg defaults match the deformable registry defaults."""
    material_cfg = NewtonDeformableMaterialCfg()

    assert material_cfg.density == DeformableRegistryEntry.density
    assert material_cfg.particle_radius == DeformableRegistryEntry.particle_radius


def test_builder_hook_applies_env_quaternion_to_deformable_entry():
    """Test that deformable builder placement honors the environment quaternion."""
    entry = _make_surface_entry()
    builder = _FakeBuilder()
    half_sqrt = math.sqrt(0.5)

    add_deformable_entry_to_builder(
        builder,
        entry,
        env_idx=0,
        env_position=[10.0, 20.0, 30.0],
        env_rotation=[0.0, 0.0, half_sqrt, half_sqrt],
    )

    mesh = builder.cloth_meshes[0]
    rotated_x_axis = wp.quat_rotate(mesh["rot"], wp.vec3(1.0, 0.0, 0.0))

    assert _vec3_as_tuple(mesh["pos"]) == pytest.approx((10.0, 21.0, 30.0))
    assert _vec3_as_tuple(rotated_x_axis) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)
    assert entry.particle_offsets == [0]
    assert entry.particles_per_body == 3


def test_builder_hook_resets_entry_offsets_on_first_environment():
    """Test that repeated model rebuilds do not accumulate stale particle offsets."""
    entry = _make_surface_entry()
    builder = _FakeBuilder()
    identity = [0.0, 0.0, 0.0, 1.0]

    add_deformable_entry_to_builder(builder, entry, 0, [0.0, 0.0, 0.0], identity)
    add_deformable_entry_to_builder(builder, entry, 1, [1.0, 0.0, 0.0], identity)

    assert entry.particle_offsets == [0, 3]

    rebuilt_builder = _FakeBuilder()
    add_deformable_entry_to_builder(rebuilt_builder, entry, 0, [0.0, 0.0, 0.0], identity)

    assert entry.particle_offsets == [0]
    assert entry.particles_per_body == 3


def test_native_geometry_uses_planned_paths_without_stage_sinks(monkeypatch):
    """Sparse destination IDs select native ranges without requiring USD clones."""
    from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

    entry = _make_surface_entry()
    entry.prim_path = "/Scene/copy_[^/]+/cloth"
    entry.vis_mesh_prim_path = entry.prim_path + "/mesh"
    entry.particle_offsets = [7, 19]
    entry.particles_per_body = 3
    plan = ClonePlan(
        sources=("/Source",),
        destinations=("/Scene/copy_{}",),
        clone_mask=np.array([[True, False, True]]),
        env_ids=np.array([7, 12, 42]),
    )
    particles = wp.zeros(22, dtype=wp.vec3f, device="cpu")
    state = SimpleNamespace(particle_q=particles, body_q=None)
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])
    monkeypatch.setattr(NewtonManager, "_cable_bindings", {})
    monkeypatch.setattr(NewtonSceneDataBackend, "state", property(lambda self: state))
    backend = NewtonSceneDataBackend()
    backend.initialize_geometry(plan)
    [(publication, ranges)] = backend.get_geometry_batches()
    assert publication.points is particles
    assert ranges == {"/Scene/copy_7/cloth/mesh": (7, 3), "/Scene/copy_42/cloth/mesh": (19, 3)}


def test_volume_visual_interpolation_is_built_once_before_native_replication(monkeypatch):
    """Distinct visual vertices interpolate directly from native nodes for every planned instance."""
    from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

    from pxr import Sdf, Usd, UsdGeom, UsdShade

    import isaaclab.sim.utils.queries as queries
    from isaaclab.scene_data import SceneDataFormat, SceneDataProvider

    import isaaclab_contrib.deformable.deformable_object as deformable

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Scene/copy_0/Volume")
    root.AddTranslateOp().Set((10, 20, 30))
    tet = UsdGeom.TetMesh.Define(stage, "/Scene/copy_0/Volume/sim")
    tet.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
    UsdGeom.Xformable(tet).AddTranslateOp().Set((1, 0, 0))
    visual = UsdGeom.Mesh.Define(stage, "/Scene/copy_0/Volume/vis")
    visual.CreatePointsAttr([(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)])
    visual.CreateFaceVertexCountsAttr([3])
    visual.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.Xformable(visual).AddTranslateOp().Set((1, 0, 0))
    material = UsdShade.Material.Define(stage, "/Material")
    material.GetPrim().CreateAttribute("newton:density", Sdf.ValueTypeNames.Float).Set(1000.0)
    UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(material, materialPurpose="physics")
    monkeypatch.setattr(queries, "get_current_stage", lambda: stage)
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [])
    remap = Mock(wraps=deformable.build_volume_vis_barycentric_remap)
    monkeypatch.setattr(deformable, "build_volume_vis_barycentric_remap", remap)
    asset = SimpleNamespace(
        cfg=SimpleNamespace(
            prim_path="/Scene/copy_[^/]+/Volume", spawn=SimpleNamespace(spawn_path="/Scene/copy_0/Volume")
        )
    )
    entry = DeformableObject._register_deformable(asset)
    entry.particle_offsets, entry.particles_per_body = [7, 19], 4
    plan = ClonePlan(
        sources=("/Scene/copy_0",),
        destinations=("/Scene/copy_{}",),
        clone_mask=np.array([[True, False, True]]),
        env_ids=np.array([7, 12, 42]),
        global_paths=("/Scene",),
    )
    nodes = np.zeros((23, 3), dtype=np.float32)
    nodes[7:11], nodes[19:23] = np.asarray(entry.vertices), np.asarray(entry.vertices) + [100, 0, 0]
    state = SimpleNamespace(particle_q=wp.array(nodes, dtype=wp.vec3f, device="cpu"), body_q=None)
    monkeypatch.setattr(NewtonManager, "_cable_bindings", {})
    monkeypatch.setattr(NewtonSceneDataBackend, "state", property(lambda self: state))
    backend = NewtonSceneDataBackend()
    backend.initialize_geometry(plan)
    stage.RemovePrim("/Scene")
    [(publication, ranges)] = backend.get_geometry_batches()
    assert publication._cls is SceneDataFormat.WeightedPoints
    assert publication.points is state.particle_q
    assert ranges == {"/Scene/copy_7/Volume/vis": (0, 3), "/Scene/copy_42/Volume/vis": (3, 3)}
    points = SceneDataProvider(backend).get_geometry_points()
    expected = np.array([[11.5, 20, 30], [11, 20.5, 30], [11, 20, 30.5]])
    np.testing.assert_allclose(points["/Scene/copy_7/Volume/vis"].numpy(), expected)
    np.testing.assert_allclose(points["/Scene/copy_42/Volume/vis"].numpy(), expected + [100, 0, 0])
    remap.assert_called_once()
