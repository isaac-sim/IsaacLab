# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton replication builder ownership."""

import importlib
from types import SimpleNamespace
from unittest import mock

import newton
import pytest
import torch
from isaaclab_newton.cloner import copy_newton_clone_source, newton_builder_world_hook
from isaaclab_newton.physics import NewtonManager

from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

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


def test_explicit_global_import_uses_global_world(monkeypatch):
    """Declared global colliders remain in Newton world -1 after model finalization."""
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    UsdGeom.Xform.Define(stage, "/World")
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdLux.DistantLight.Define(stage, "/World/Light")
    global_paths = ("/World/Ground", "/World/Light")

    builder = newton.ModelBuilder()
    add_usd = mock.Mock(wraps=builder.add_usd)
    monkeypatch.setattr(builder, "add_usd", add_usd)
    manager = SimpleNamespace(
        create_builder=mock.Mock(return_value=builder), _inject_terrain_heightfields=mock.Mock(return_value=[])
    )
    monkeypatch.setattr(
        replicate_module.PhysicsManager,
        "_sim",
        SimpleNamespace(physics_manager=manager, cfg=SimpleNamespace(physics_prim_path="/physicsScene")),
    )
    monkeypatch.setattr(replicate_module.NewtonManager, "_deformable_registry", ())
    monkeypatch.setattr(replicate_module.NewtonManager, "_cl_inject_sites", mock.Mock(return_value=({}, {}, {})))
    monkeypatch.setattr(replicate_module.NewtonManager, "_per_world_builder_hooks", ())
    monkeypatch.setattr(replicate_module, "replace_newton_builder_shape_colors", mock.Mock())

    builder, *_ = replicate_module._build_newton_builder_from_mapping(
        stage,
        (),
        (),
        torch.arange(2),
        torch.empty((0, 2), dtype=torch.bool),
        global_paths=global_paths,
        load_visual_shapes=False,
    )

    assert [call.kwargs["root_path"] for call in add_usd.call_args_list] == ["/physicsScene", *global_paths]
    manager._inject_terrain_heightfields.assert_called_once_with(
        stage, builder, root_paths=("/physicsScene", *global_paths)
    )
    model = builder.finalize("cpu")
    ground_index = model.shape_label.index("/World/Ground")
    assert model.shape_world.numpy()[ground_index] == -1
    assert model.world_count == 2
    assert "/World/Light" not in model.shape_label  # USD lights are not Newton physics entities.
