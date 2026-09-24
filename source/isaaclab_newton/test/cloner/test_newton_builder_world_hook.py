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
from isaaclab_newton.physics import NewtonManager

from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

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


@pytest.mark.parametrize("load_visual_shapes", [False, True])
def test_scene_import_preserves_global_and_prototype_ownership(monkeypatch, load_visual_shapes):
    """Global lights stay shared; prototype lights follow geometry into selected worlds."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    UsdGeom.Xform.Define(stage, "/World")
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdLux.DistantLight.Define(stage, "/World/Light").GetIntensityAttr().Set(750.0)
    global_paths = ("/World/Ground", "/World/Light")
    source = "/World/envs/env_1/Lamp"
    UsdGeom.Xform.Define(stage, source).AddTranslateOp().Set((21.0, 0.0, 0.0))
    UsdLux.SphereLight.Define(stage, f"{source}/Light")
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, f"{source}/Cube").GetPrim())
    unused_source = "/World/envs/env_0/Unused"
    UsdLux.SphereLight.Define(stage, f"{unused_source}/Light")

    builder = newton.ModelBuilder()
    add_usd = mock.Mock(wraps=builder.add_usd)
    monkeypatch.setattr(builder, "add_usd", add_usd)
    manager = SimpleNamespace(
        create_builder=mock.Mock(side_effect=[builder, newton.ModelBuilder(), newton.ModelBuilder()]),
        _get_usd_import_schema_resolvers=NewtonManager._get_usd_import_schema_resolvers,
        _inject_terrain_heightfields=mock.Mock(return_value=[]),
    )
    monkeypatch.setattr(
        replicate_module.PhysicsManager,
        "_sim",
        SimpleNamespace(physics_manager=manager, cfg=SimpleNamespace(physics_prim_path="/physicsScene")),
    )
    monkeypatch.setattr(replicate_module.NewtonManager, "_deformable_registry", ())
    monkeypatch.setattr(replicate_module.NewtonManager, "_cl_inject_sites", mock.Mock(return_value=({}, {}, {})))
    monkeypatch.setattr(
        replicate_module.NewtonManager,
        "_per_world_builder_hooks",
        (lambda builder, index, *_args: builder.add_body(label=f"/World/envs/env_{index}/Anchor"),),
    )
    monkeypatch.setattr(replicate_module, "replace_newton_builder_shape_colors", mock.Mock())

    builder, *_ = replicate_module._build_newton_builder_from_mapping(
        stage,
        (source, unused_source),
        ("/World/envs/env_{}/Lamp", "/World/envs/env_{}/Unused"),
        np.arange(3, dtype=np.int64),
        np.asarray([[False, True, True], [False, False, False]]),
        positions=np.asarray([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=np.float32),
        quaternions=np.asarray([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 2**-0.5, 2**-0.5]], dtype=np.float32),
        global_paths=global_paths,
        load_visual_shapes=load_visual_shapes,
    )

    assert [call.kwargs["root_path"] for call in add_usd.call_args_list] == ["/physicsScene", *global_paths]
    manager._inject_terrain_heightfields.assert_called_once_with(
        stage, builder, root_paths=("/physicsScene", *global_paths)
    )
    model = builder.finalize("cpu")
    ground_index = model.shape_label.index("/World/Ground")
    assert model.shape_world.numpy()[ground_index] == -1
    assert model.world_count == 3
    assert "/World/Light" not in model.shape_label  # USD lights are not Newton physics entities.
    if load_visual_shapes:
        lights = Usd.Stage.CreateInMemory()
        lights.GetRootLayer().ImportFromString(model.isaaclab.scene_lights[0])
        imported = [UsdLux.DistantLight(prim) for prim in lights.Traverse() if prim.IsA(UsdLux.DistantLight)]
        assert len(imported) == 1
        assert imported[0].GetIntensityAttr().Get() == 750.0
        lamps = [prim for prim in lights.Traverse() if prim.IsA(UsdLux.SphereLight)]
        assert len(lamps) == 2
        transforms = UsdGeom.XformCache()
        for env_id, lamp in zip((1, 2), lamps, strict=True):
            shape_index = model.shape_label.index(f"/World/envs/env_{env_id}/Lamp/Cube")
            pose = model.shape_transform.numpy()[shape_index]
            rotation = Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6])))
            expected = Gf.Matrix4d().SetRotate(rotation).SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
            np.testing.assert_allclose(transforms.GetLocalToWorldTransform(lamp), expected, atol=1e-5)
