# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton replication builder ownership."""

import importlib
from types import SimpleNamespace
from unittest import mock

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.cloner import copy_newton_clone_source, newton_builder_world_hook, newton_physics_replicate
from isaaclab_newton.physics import NewtonCfg, NewtonManager, VBDSolverCfg
from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ClonePlan
from isaaclab.sim import SimulationCfg, build_simulation_context

replicate_module = importlib.import_module("isaaclab_newton.cloner.replicate")


def test_newton_builder_world_hook_owns_one_registration(monkeypatch):
    """The scope rejects duplicates and preserves unrelated hooks during cleanup."""

    def existing(*_args):
        pass

    def temporary(*_args):
        pass

    def added_later(*_args):
        pass

    hooks = [existing]
    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", hooks)

    with pytest.raises(ValueError, match="stop"):
        with newton_builder_world_hook(temporary):
            assert hooks == [existing, temporary]
            with pytest.raises(RuntimeError, match="already registered"):
                with newton_builder_world_hook(temporary):
                    pass
            assert hooks == [existing, temporary]
            hooks.append(added_later)
            raise ValueError("stop")

    assert hooks == [existing, added_later]

    with pytest.raises(RuntimeError, match="already registered"):
        with newton_builder_world_hook(existing):
            pass
    assert hooks == [existing, added_later]


def test_copy_newton_clone_source_owns_mutable_geometry(monkeypatch):
    """Finalizing a copied prototype must not mutate cloner-retained shape sources."""
    source = newton.ModelBuilder()
    body = source.add_body()
    mesh = newton.Mesh(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], indices=[0, 1, 2])
    source.add_shape_mesh(body, mesh=mesh)
    monkeypatch.setattr(NewtonManager, "_cl_protos", {"/World/Source": source})

    copied = copy_newton_clone_source("/World/Source")

    assert copied.shape_source[0] is not source.shape_source[0]


@pytest.mark.parametrize(
    "load_visual_shapes,is_rendering,rgb_array,visual_shapes_required,expected",
    [
        pytest.param(None, False, False, False, False, id="headless"),
        pytest.param(None, True, False, False, True, id="viewer"),
        pytest.param(None, False, True, False, True, id="offscreen"),
        pytest.param(None, False, False, True, True, id="camera"),
        pytest.param(True, False, False, False, True, id="force-visuals"),
        pytest.param(False, True, True, True, False, id="skip-visuals"),
    ],
)
def test_explicit_global_import_uses_global_world(
    monkeypatch, load_visual_shapes, is_rendering, rgb_array, visual_shapes_required, expected
):
    """Global imports honor visual requirements and import shared deformables exactly once."""
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    UsdGeom.Xform.Define(stage, "/World")
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdLux.DistantLight.Define(stage, "/World/Light")
    points = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
    native_mesh = UsdGeom.TetMesh.Define(stage, "/World/Native/sim")
    native_mesh.CreatePointsAttr(points)
    native_mesh.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
    root = stage.GetPrimAtPath("/World/Native")
    root.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(["OmniPhysicsDeformableBodyAPI"]))
    material = UsdShade.Material.Define(stage, "/World/Native/Material")
    material.GetPrim().CreateAttribute("newton:density", Sdf.ValueTypeNames.Float).Set(1000.0)
    UsdShade.MaterialBindingAPI.Apply(root).Bind(material, materialPurpose="physics")
    global_paths = ("/World/Ground", "/World/Light", "/World/Native")

    builder = newton.ModelBuilder()
    add_usd = mock.Mock(wraps=builder.add_usd)
    monkeypatch.setattr(builder, "add_usd", add_usd)
    manager = SimpleNamespace(
        create_builder=mock.Mock(return_value=builder),
        _get_usd_import_schema_resolvers=NewtonManager._get_usd_import_schema_resolvers,
        _inject_terrain_heightfields=mock.Mock(return_value=[]),
    )
    monkeypatch.setattr(
        replicate_module.PhysicsManager,
        "_sim",
        SimpleNamespace(
            physics_manager=manager,
            device="cpu",
            cfg=SimpleNamespace(
                physics=NewtonCfg(load_visual_shapes=load_visual_shapes), physics_prim_path="/physicsScene"
            ),
            is_rendering=is_rendering,
            can_render_rgb_array=lambda: rgb_array,
            visual_shapes_required=visual_shapes_required,
        ),
    )
    monkeypatch.setattr(NewtonManager, "_scene_data_backend", NewtonSceneDataBackend())
    monkeypatch.setattr(replicate_module.NewtonManager, "_cl_inject_sites", mock.Mock(return_value=({}, {}, {})))
    monkeypatch.setattr(NewtonManager, "_per_world_builder_hooks", ())
    monkeypatch.setattr(replicate_module, "replace_newton_builder_shape_colors", mock.Mock())
    monkeypatch.setattr(NewtonManager, "_builder", None)
    monkeypatch.setattr(NewtonManager, "_cl_site_index_map", {})
    monkeypatch.setattr(NewtonManager, "_world_xforms", None)
    monkeypatch.setattr(NewtonManager, "_cl_protos", {})
    monkeypatch.setattr(NewtonManager, "_num_envs", 0)

    builder, _ = replicate_module.newton_physics_replicate(
        stage,
        (),
        (),
        np.arange(2, dtype=np.int64),
        np.empty((0, 2), dtype=np.bool_),
        global_paths=global_paths,
    )

    assert [call.kwargs["root_path"] for call in add_usd.call_args_list] == ["/physicsScene", *global_paths]
    assert all(call.kwargs["load_visual_shapes"] is expected for call in add_usd.call_args_list)
    manager._inject_terrain_heightfields.assert_called_once_with(
        stage, builder, root_paths=("/physicsScene", *global_paths)
    )
    model = builder.finalize("cpu")
    ground_index = model.shape_label.index("/World/Ground")
    assert model.shape_world.numpy()[ground_index] == -1
    assert model.world_count == 2
    assert model.particle_count == len(points)
    np.testing.assert_array_equal(model.particle_world.numpy(), -1)
    assert "/World/Light" not in model.shape_label  # USD lights are not Newton physics entities.


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
def test_imported_deformables_follow_plan_and_publish_geometry(heterogeneous):
    """Import once, clone only selected rows, and bind native/embedded visuals without cloned USD."""
    sim_cfg = SimulationCfg(device="cpu", physics=NewtonCfg(solver_cfg=VBDSolverCfg(), load_visual_shapes=False))
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        stage = sim.stage
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
            cfgs=tuple(
                AssetBaseCfg(prim_path=path)
                for path in (
                    "/Sources/A/Cloth",
                    "/Sources/B/Volume" if heterogeneous else "/Sources/A/Volume",
                    "/Shared/Cloth",
                )
            ),
        )
        sim.set_clone_plan(plan)
        builder, _ = newton_physics_replicate(
            stage,
            plan.sources,
            plan.destinations,
            plan.env_ids,
            plan.clone_mask,
            positions=positions,
            quaternions=rotations,
            global_paths=plan.global_paths,
            cfgs=plan.cfgs,
        )
        sim.reset()
        native = NewtonManager.backend
        stage.RemovePrim("/Sources")
        stage.RemovePrim("/Shared")

        expected_counts = [3, 4, 3] if heterogeneous else [7, 7, 7]
        assert builder.particle_count == sum(expected_counts) + 3
        np.testing.assert_array_equal(np.bincount(np.asarray(builder.particle_world) + 1), [3, *expected_counts])
        assert native.deformable_ranges["/Shared/Cloth"] == (0, 3, "surface")
        points = sim.get_scene_data_provider().get_geometry_points()
        expected_paths = {"/Shared/Cloth/sim"}
        local_vertices = np.asarray([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32)
        for world, env_id in enumerate(plan.env_ids):
            kinds = (
                ("Volume",) if heterogeneous and world == 1 else ("Cloth",) if heterogeneous else ("Cloth", "Volume")
            )
            for name in kinds:
                path = f"/World/env_{env_id}/{name}"
                start, count, _ = native.deformable_ranges[path]
                xform = wp.transform(positions[world], rotations[world])
                expected = np.asarray([wp.transform_point(xform, wp.vec3(p)) for p in local_vertices[:count]])
                np.testing.assert_allclose(
                    native.state_0.particle_q.numpy()[start : start + count], expected, atol=1e-6
                )
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
