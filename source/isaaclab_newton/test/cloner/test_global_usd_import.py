# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for explicit global USD imports during Newton replication."""

import importlib
from types import SimpleNamespace
from unittest import mock

import torch

replicate_module = importlib.import_module("isaaclab_newton.cloner.replicate")


def test_explicit_global_import_uses_only_declared_roots(monkeypatch):
    """Newton imports the physics scene and each declared global, never an inferred stage root."""
    stage = object()
    builder = mock.Mock()
    physics_info = {"path_shape_map": {}, "physics_scene_path": "/physicsScene"}
    builder.add_usd.side_effect = [physics_info, {"path_shape_map": {}}, {"path_shape_map": {}}]
    manager = SimpleNamespace(
        create_builder=mock.Mock(return_value=builder), _inject_terrain_heightfields=mock.Mock(return_value=[])
    )
    restore = mock.Mock()
    monkeypatch.setattr(
        replicate_module.PhysicsManager,
        "_sim",
        SimpleNamespace(physics_manager=manager, cfg=SimpleNamespace(physics_prim_path="/physicsScene")),
    )
    monkeypatch.setattr(replicate_module, "_restore_visible_colliders_without_visual_shapes", restore)
    monkeypatch.setattr(replicate_module, "replace_newton_builder_shape_colors", mock.Mock())
    monkeypatch.setattr(replicate_module, "build_source_builders", mock.Mock(return_value={}))
    monkeypatch.setattr(replicate_module.NewtonManager, "_deformable_registry", ())
    monkeypatch.setattr(replicate_module.NewtonManager, "_cl_inject_sites", mock.Mock(return_value=({}, {}, {})))
    monkeypatch.setattr(replicate_module, "replicate_builder_mapping", mock.Mock(return_value=({}, [])))

    _, stage_info, _, _, _ = replicate_module._build_newton_builder_from_mapping(
        stage=stage,
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        env_ids=torch.arange(2),
        mapping=torch.ones((1, 2), dtype=torch.bool),
        global_paths=("/World/Ground", "/World/Light"),
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
