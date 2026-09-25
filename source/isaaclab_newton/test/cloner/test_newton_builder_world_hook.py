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
from isaaclab_newton.cloner import copy_newton_clone_source, newton_builder_world_hook
from isaaclab_newton.physics import NewtonCfg, NewtonManager
from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend

from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

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
    monkeypatch.setattr(NewtonManager, "_cl_fabric_body_bindings", [])
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
