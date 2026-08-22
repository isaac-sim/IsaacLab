# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for explicit global USD imports during Newton replication."""

import importlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from pxr import Usd, UsdGeom, UsdPhysics

replicate_module = importlib.import_module("isaaclab_newton.cloner.replicate")


def test_context_forwards_explicit_global_paths(monkeypatch):
    """The Newton context carries globals independently from replication rows."""
    context = replicate_module.NewtonReplicateContext(object(), load_visual_shapes=False, commit_to_manager=False)
    context.queue_global_paths(("/World/Ground", "/World/Light"))
    context.queue_mapping(
        ("/World/envs/env_0/Robot",),
        ("/World/envs/env_{}/Robot",),
        torch.arange(2),
        torch.ones((1, 2), dtype=torch.bool),
    )
    build = mock.Mock(return_value=(object(), object(), {}, [], {}))
    monkeypatch.setattr(replicate_module, "_build_newton_builder_from_mapping", build)
    monkeypatch.setattr(replicate_module, "rename_builder_labels", lambda *args: [])

    context.replicate()

    assert build.call_args.kwargs["global_paths"] == ("/World/Ground", "/World/Light")


def test_explicit_global_import_uses_only_declared_roots(monkeypatch):
    """Newton imports the physics scene and each declared global, never an inferred stage root."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Ground")
    UsdGeom.Xform.Define(stage, "/World/Ground/Material")
    UsdGeom.Xform.Define(stage, "/World/Light")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    builder = mock.Mock()
    physics_info = {"path_shape_map": {}, "physics_scene_path": "/physicsScene"}
    builder.add_usd.side_effect = [physics_info, {"path_shape_map": {}}, {"path_shape_map": {}}]
    manager = SimpleNamespace(_inject_terrain_heightfields=mock.Mock(return_value=[]))
    restore = mock.Mock()
    monkeypatch.setattr(replicate_module, "_restore_visible_colliders_without_visual_shapes", restore)

    stage_info = replicate_module._import_global_usd(
        stage,
        builder,
        manager,
        sources=("/World/envs/env_0",),
        global_paths=("/World/Ground", "/World/Ground/Material", "/World/Light"),
        physics_scene_path="/physicsScene",
        schema_resolvers=("newton", "physx"),
        load_visual_shapes=False,
    )

    assert [call.kwargs["root_path"] for call in builder.add_usd.call_args_list] == [
        "/physicsScene",
        "/World/Ground",
        "/World/Light",
    ]
    assert stage_info is physics_info
    manager._inject_terrain_heightfields.assert_called_once_with(
        stage, builder, root_paths=("/physicsScene", "/World/Ground", "/World/Light")
    )
    assert restore.call_count == 3


def test_explicit_global_import_rejects_a_root_containing_clone_sources():
    """A declared global must not silently pull the replicated environment subtree back in."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdPhysics.Scene.Define(stage, "/physicsScene")

    with (
        mock.patch.object(replicate_module, "_restore_visible_colliders_without_visual_shapes"),
        pytest.raises(ValueError, match="contains clone source"),
    ):
        replicate_module._import_global_usd(
            stage,
            mock.Mock(),
            SimpleNamespace(_inject_terrain_heightfields=mock.Mock(return_value=[])),
            sources=("/World/envs/env_0",),
            global_paths=("/World",),
            physics_scene_path="/physicsScene",
            schema_resolvers=(),
            load_visual_shapes=False,
        )


def test_unspecified_globals_preserve_the_legacy_full_stage_import(monkeypatch):
    """Direct callers without the new declaration retain the previous discovery path."""
    builder = mock.Mock()
    stage_info = {"path_shape_map": {}}
    builder.add_usd.return_value = stage_info
    manager = SimpleNamespace(_inject_terrain_heightfields=mock.Mock(return_value=["/World/Terrain"]))
    restore = mock.Mock()
    monkeypatch.setattr(replicate_module, "_restore_visible_colliders_without_visual_shapes", restore)

    result = replicate_module._import_global_usd(
        object(),
        builder,
        manager,
        sources=("/World/envs/env_0/Robot",),
        global_paths=None,
        physics_scene_path="/physicsScene",
        schema_resolvers=("newton", "physx"),
        load_visual_shapes=False,
    )

    assert result is stage_info
    manager._inject_terrain_heightfields.assert_called_once_with(mock.ANY, builder)
    builder.add_usd.assert_called_once_with(
        mock.ANY,
        ignore_paths=["/World/envs", "/World/envs/env_0/Robot", "/World/Terrain"],
        schema_resolvers=("newton", "physx"),
        load_visual_shapes=False,
    )
