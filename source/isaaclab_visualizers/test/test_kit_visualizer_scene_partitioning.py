# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Kit visualizer scene-partition behavior."""

from unittest.mock import MagicMock

import isaaclab_visualizers.kit.kit_visualizer as kit_visualizer_module
import pytest
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer

from pxr import Sdf, Usd, UsdGeom


@pytest.mark.parametrize(("show_global_view", "expected_partition"), [(True, None), (False, "env_2")])
def test_viewport_camera_partition_follows_global_view_setting(
    monkeypatch: pytest.MonkeyPatch, show_global_view: bool, expected_partition: str | None
) -> None:
    """Global view should leave the viewport unpartitioned; fallback view should select one environment."""
    stage = Usd.Stage.CreateInMemory()
    env_prim = stage.DefinePrim("/World/envs/env_0", "Xform")
    env_prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    camera = UsdGeom.Camera.Define(stage, "/OmniverseKit_Persp")
    camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")

    visualizer = object.__new__(KitVisualizer)
    visualizer._controlled_camera_path = "/OmniverseKit_Persp"
    visualizer._resolved_visible_env_ids = [2]
    settings = MagicMock()
    settings.get.return_value = show_global_view
    monkeypatch.setattr(kit_visualizer_module, "get_settings_manager", lambda: settings)

    visualizer._apply_viewport_camera_scene_partition(stage, num_envs=4)

    assert camera.GetPrim().GetAttribute("omni:scenePartition").Get() == expected_partition
