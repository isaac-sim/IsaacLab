# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for explicit global USD imports during Newton replication."""

import importlib
from types import SimpleNamespace
from unittest import mock

replicate_module = importlib.import_module("isaaclab_newton.cloner.replicate")


def test_explicit_global_import_uses_only_declared_roots(monkeypatch):
    """Newton imports the physics scene and each declared global, never an inferred stage root."""
    stage = object()
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
        global_paths=("/World/Ground", "/World/Light"),
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
