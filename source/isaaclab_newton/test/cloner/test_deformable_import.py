# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deformable USD import, native replication, and SDP publication without runtime asset registration."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.cloner import newton_physics_replicate
from isaaclab_newton.physics import NewtonBackendCfg, NewtonCfg, NewtonManager, NewtonVBDManager, VBDSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonBackend, NewtonSceneDataBackend

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.cloner import ClonePlan
from isaaclab.physics import PhysicsManager
from isaaclab.scene_data import SceneDataProvider


def _author_deformable(stage, path, kind):
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    root.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(["OmniPhysicsDeformableBodyAPI"]))
    points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if kind == "volume":
        mesh = UsdGeom.TetMesh.Define(stage, path + "/sim")
        mesh.CreatePointsAttr(points)
        mesh.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
        visual = UsdGeom.Mesh.Define(stage, path + "/vis")
        visual.CreatePointsAttr([(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)])
        visual.CreateFaceVertexCountsAttr([3])
        visual.CreateFaceVertexIndicesAttr([0, 1, 2])
        schema = "OmniPhysicsVolumeDeformableSimAPI"
    else:
        mesh = UsdGeom.Mesh.Define(stage, path + "/sim")
        mesh.CreatePointsAttr(points[:3])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        schema = "OmniPhysicsSurfaceDeformableSimAPI"
    mesh.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit([schema]))
    material = UsdShade.Material.Define(stage, path + "/material")
    for name, value in (("density", 6.0), ("triKe", 123.0), ("kMu", 456.0)):
        material.GetPrim().CreateAttribute("newton:" + name, Sdf.ValueTypeNames.Float).Set(value)
    UsdShade.MaterialBindingAPI.Apply(root).Bind(material, materialPurpose="physics")


@pytest.mark.parametrize("heterogeneous", [False, True], ids=["batched", "heterogeneous"])
def test_imported_deformables_follow_plan_and_publish_geometry(monkeypatch, heterogeneous):
    """Import once, clone only selected rows, and bind native/embedded visuals without cloned USD."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    _author_deformable(stage, "/Sources/A/Cloth", "surface")
    _author_deformable(stage, "/Sources/B/Volume" if heterogeneous else "/Sources/A/Volume", "volume")
    _author_deformable(stage, "/Shared/Cloth", "surface")
    sources = ("/Sources/A", "/Sources/B") if heterogeneous else ("/Sources/A",)
    mapping = np.asarray([[1, 0, 1], [0, 1, 0]] if heterogeneous else [[1, 1, 1]], dtype=np.bool_)
    positions = np.asarray([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=np.float32)
    if heterogeneous:
        UsdGeom.Xform.Define(stage, "/Sources/B").AddTranslateOp().Set(tuple(positions[1].astype(float)))
    rotations = np.asarray([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 2**-0.5, 2**-0.5]], dtype=np.float32)
    plan = ClonePlan(
        sources=sources,
        destinations=("/World/env_{}",) * len(sources),
        clone_mask=mapping,
        env_ids=np.asarray([7, 12, 42]),
        positions=positions,
        global_paths=("/Shared",),
    )
    cfg = NewtonCfg(solver_cfg=VBDSolverCfg(), load_visual_shapes=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", cfg)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    monkeypatch.setattr(
        PhysicsManager,
        "_sim",
        SimpleNamespace(
            physics_manager=NewtonVBDManager,
            device="cpu",
            cfg=SimpleNamespace(physics=cfg, physics_prim_path="/physicsScene"),
        ),
    )
    for name, value in (
        ("_builder", None),
        ("_cl_pending_sites", {}),
        ("_cl_site_index_map", {}),
        ("_cl_fabric_body_bindings", []),
        ("_cl_protos", {}),
        ("_world_xforms", None),
        ("_num_envs", 0),
        ("_mpm_object_registry", []),
        ("_per_world_builder_hooks", []),
        ("_cable_bindings", {}),
    ):
        monkeypatch.setattr(NewtonManager, name, value)
    backend = NewtonSceneDataBackend()
    monkeypatch.setattr(NewtonManager, "_scene_data_backend", backend)
    builder, _ = newton_physics_replicate(
        stage,
        plan.sources,
        plan.destinations,
        plan.env_ids,
        plan.clone_mask,
        positions=positions,
        quaternions=rotations,
        global_paths=plan.global_paths,
        asset_paths=(
            "/Sources/A/Cloth",
            "/Sources/B/Volume" if heterogeneous else "/Sources/A/Volume",
            "/Shared/Cloth",
        ),
    )
    native = NewtonBackend(NewtonBackendCfg(builder=builder, device="cpu"))
    monkeypatch.setattr(NewtonManager, "backend", native)
    monkeypatch.setattr(NewtonSceneDataBackend, "state", property(lambda self: native.state_0))
    backend.initialize_geometry()
    stage.RemovePrim("/Sources")
    stage.RemovePrim("/Shared")

    expected_counts = [3, 4, 3] if heterogeneous else [7, 7, 7]
    assert builder.particle_count == sum(expected_counts) + 3
    np.testing.assert_array_equal(np.bincount(np.asarray(builder.particle_world) + 1), [3, *expected_counts])
    assert native.deformable_ranges["/Shared/Cloth"] == (0, 3, "surface")
    points = SceneDataProvider(backend).get_geometry_points()
    expected_paths = {"/Shared/Cloth/sim"}
    local_vertices = np.asarray([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32)
    for world, env_id in enumerate(plan.env_ids):
        kinds = ("Volume",) if heterogeneous and world == 1 else ("Cloth",) if heterogeneous else ("Cloth", "Volume")
        for name in kinds:
            path = f"/World/env_{env_id}/{name}"
            start, count, _ = native.deformable_ranges[path]
            xform = wp.transform(positions[world], rotations[world])
            expected = np.asarray([wp.transform_point(xform, wp.vec3(p)) for p in local_vertices[:count]])
            np.testing.assert_allclose(native.state_0.particle_q.numpy()[start : start + count], expected, atol=1e-6)
            visual_path = path + ("/sim" if name == "Cloth" else "/vis")
            expected_paths.add(visual_path)
            if name == "Cloth":
                assert points[visual_path].ptr == native.state_0.particle_q.ptr + start * 12
                np.testing.assert_allclose(points[visual_path].numpy(), expected, atol=1e-6)
            else:
                np.testing.assert_allclose(points[visual_path].numpy(), (expected[1:] + expected[0]) / 2, atol=1e-6)
    assert set(points) == expected_paths
    np.testing.assert_allclose(np.asarray(builder.tri_materials)[builder._cloth_tri_start, 0], 123.0)
    np.testing.assert_allclose(np.asarray(builder.tet_materials)[:, 0], 456.0)
    for tet, pose in zip(builder.tet_indices, builder.tet_poses, strict=True):
        vertices = np.asarray(builder.particle_q)[tet]
        np.testing.assert_allclose((vertices[1:] - vertices[0]).T @ pose, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(builder.particle_radius, 0.008)
    native.close()
