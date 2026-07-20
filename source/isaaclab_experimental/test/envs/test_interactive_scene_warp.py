# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for mask-first scene reset dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import warp as wp
from isaaclab_experimental.envs.interactive_scene_warp import InteractiveSceneWarp

from isaaclab.scene import InteractiveScene
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.warp import ProxyArray


class TestInteractiveSceneWarp:
    """Tests for :class:`InteractiveSceneWarp`."""

    def test_frontend_uses_canonical_terrain_storage_without_mutating_cfg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Warp scenes should inherit canonical origin storage without replacing the configured class."""
        terrain_cfg = TerrainImporterCfg(prim_path="/World/Ground", terrain_type="plane")
        cfg = SimpleNamespace(terrain=terrain_cfg)
        class_type = terrain_cfg.class_type
        origins_wp = wp.zeros(2, dtype=wp.vec3f, device="cpu")
        terrain = SimpleNamespace(env_origins=ProxyArray(origins_wp))

        def initialize_scene(scene: InteractiveScene, scene_cfg: SimpleNamespace) -> None:
            scene.cfg = scene_cfg
            scene._terrain = terrain

        monkeypatch.setattr(InteractiveScene, "__init__", initialize_scene)

        scene = InteractiveSceneWarp(cfg)

        assert terrain_cfg.class_type is class_type
        assert isinstance(scene.env_origins, torch.Tensor)
        assert scene.env_origins is terrain.env_origins.torch
        assert scene.env_origins_wp is origins_wp
        assert InteractiveSceneWarp.env_origins is InteractiveScene.env_origins
        assert InteractiveSceneWarp.env_origins_wp is InteractiveScene.env_origins_wp

    def test_mask_reset_stays_mask_based_for_supported_entities(self):
        """Mask-based reset should not pass environment IDs to Warp-capable entities."""
        scene = InteractiveSceneWarp.__new__(InteractiveSceneWarp)
        articulation = Mock()
        deformable = Mock()
        rigid_object = Mock()
        rigid_collection = Mock()
        sensor = Mock()
        surface_gripper = Mock()
        scene._articulations = {"articulation": articulation}
        scene._deformable_objects = {"deformable": deformable}
        scene._rigid_objects = {"rigid_object": rigid_object}
        scene._rigid_object_collections = {"collection": rigid_collection}
        scene._sensors = {"sensor": sensor}
        scene._surface_grippers = {"gripper": surface_gripper}
        scene.cfg = SimpleNamespace(num_envs=3)
        scene.sim = SimpleNamespace(device="cpu")
        env_mask = wp.array([True, False, True], dtype=wp.bool, device="cpu")

        scene.reset(env_mask=env_mask)

        for entity in (articulation, deformable, rigid_object, rigid_collection, sensor):
            entity.reset.assert_called_once_with(env_mask=env_mask)
        surface_gripper.reset.assert_not_called()

    def test_surface_gripper_reset_is_an_explicit_id_boundary(self):
        """Surface grippers should be reset separately because their API is ID-based."""
        scene = InteractiveSceneWarp.__new__(InteractiveSceneWarp)
        surface_gripper = Mock()
        scene._surface_grippers = {"gripper": surface_gripper}

        scene.reset_host([0, 2])

        surface_gripper.reset.assert_called_once_with([0, 2])

    def test_mask_shape_is_validated_before_entity_dispatch(self):
        """A malformed mask should fail before reaching scene entities."""
        scene = InteractiveSceneWarp.__new__(InteractiveSceneWarp)
        scene.cfg = SimpleNamespace(num_envs=3)
        scene.sim = SimpleNamespace(device="cpu")
        env_mask = wp.array([True, False], dtype=wp.bool, device="cpu")

        with pytest.raises(ValueError, match="shape"):
            scene.reset(env_mask=env_mask)
