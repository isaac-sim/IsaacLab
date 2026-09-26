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

from isaaclab.cloner import ClonePlan, PrototypeWorldTopology

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


def test_builder_hook_preserves_placement_materials_and_rebuild_offsets():
    """Clone placement preserves material defaults and rebuilding replaces particle offsets."""
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

    assert tuple(mesh["pos"]) == pytest.approx((10.0, 21.0, 30.0))
    assert tuple(rotated_x_axis) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)
    material = NewtonDeformableMaterialCfg()
    assert mesh["density"] == material.density
    assert mesh["particle_radius"] == material.particle_radius
    identity = [0.0, 0.0, 0.0, 1.0]
    add_deformable_entry_to_builder(builder, entry, 1, [1.0, 0.0, 0.0], identity)
    assert entry.particle_offsets == [0, 3]

    add_deformable_entry_to_builder(_FakeBuilder(), entry, 0, [0.0, 0.0, 0.0], identity)
    assert entry.particle_offsets == [0]
    assert entry.particles_per_body == 3


def test_planned_geometry_aliases_surface_nodes_and_interpolates_volume_once(monkeypatch):
    """Sparse planned instances publish exact visual paths without requiring cloned USD meshes."""
    from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

    from pxr import Sdf, Usd, UsdGeom, UsdShade

    import isaaclab.sim.utils.queries as queries
    from isaaclab.scene_data import SceneDataProvider

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
    surface = _make_surface_entry()
    surface.prim_path = "/Scene/copy_[^/]+/cloth"
    surface.vis_mesh_prim_path = surface.prim_path + "/mesh"
    surface.particle_offsets, surface.particles_per_body = [31, 37], 3
    NewtonManager._deformable_registry.append(surface)
    world_layout = np.zeros(43, dtype=np.int32)
    world_layout[[7, 42]] = 1
    plan = ClonePlan(
        PrototypeWorldTopology(
            num_asset_prototypes=2,
            world_prototypes=np.array([0, 1]),
            world_prototype_starts=np.array([0, 0, 0, 2]),
            world_prototype_layout=world_layout,
        ),
        asset_cfgs=(asset.cfg, SimpleNamespace(prim_path=surface.prim_path)),
    )
    instances = tuple(
        (index, f"/Scene/copy_0/{name}", f"/Scene/copy_{{}}/{name}", np.array([7, 42]))
        for index, name in enumerate(("Volume", "cloth"))
    )
    nodes = np.zeros((40, 3), dtype=np.float32)
    nodes[7:11], nodes[19:23] = np.asarray(entry.vertices), np.asarray(entry.vertices) + [100, 0, 0]
    state = SimpleNamespace(particle_q=wp.array(nodes, dtype=wp.vec3f, device="cpu"), body_q=None)
    monkeypatch.setattr(NewtonManager, "_cable_bindings", {})
    monkeypatch.setattr(NewtonSceneDataBackend, "state", property(lambda self: state))
    backend = NewtonSceneDataBackend()
    backend.initialize_geometry(plan, instances)
    stage.RemovePrim("/Scene")
    points = SceneDataProvider(backend).get_geometry_points()
    assert set(points) == {f"/Scene/copy_{index}/{mesh}" for index in (7, 42) for mesh in ("Volume/vis", "cloth/mesh")}
    for index, offset in zip((7, 42), surface.particle_offsets, strict=True):
        view = points[f"/Scene/copy_{index}/cloth/mesh"]
        assert view.ptr == state.particle_q.ptr + offset * wp.types.type_size_in_bytes(wp.vec3f)
        assert len(view) == surface.particles_per_body
    expected = np.array([[11.5, 20, 30], [11, 20.5, 30], [11, 20, 30.5]])
    np.testing.assert_allclose(points["/Scene/copy_7/Volume/vis"].numpy(), expected)
    np.testing.assert_allclose(points["/Scene/copy_42/Volume/vis"].numpy(), expected + [100, 0, 0])
    remap.assert_called_once()
