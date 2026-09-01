# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Kit visualizer scene-partition behavior."""

from unittest.mock import MagicMock

import isaaclab_visualizers.kit.kit_visualizer as kit_visualizer_module
import pytest
import torch
from isaaclab_visualizers.kit.kit_visualization_markers import KitVisualizationMarkers
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer

from pxr import Sdf, Usd, UsdGeom

from isaaclab.utils.renderers import ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING


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

    settings.get.assert_called_once_with(ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING, False)
    partition_attr = camera.GetPrim().GetAttribute("omni:scenePartition")
    if expected_partition is None:
        assert not partition_attr.IsValid()
    else:
        assert partition_attr.Get() == expected_partition


def test_marker_partition_detection_uses_canonical_environment_root() -> None:
    """The renderer-authored env_0 root should be the stage-level partition signal."""
    stage = Usd.Stage.CreateInMemory()
    env_1 = stage.DefinePrim("/World/envs/env_1", "Xform")
    env_1.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_1")
    markers = object.__new__(KitVisualizationMarkers)
    markers.stage = stage
    markers._environment_ids = (1,)

    assert not markers._scene_partitioning_is_active()

    env_0 = stage.DefinePrim("/World/envs/env_0", "Xform")
    env_0.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")

    assert markers._scene_partitioning_is_active()


def test_marker_environment_ids_are_sticky_until_count_changes() -> None:
    """Omitted environment IDs should persist only while the marker count is unchanged."""
    stage = Usd.Stage.CreateInMemory()
    for env_id in range(2):
        env_prim = stage.DefinePrim(f"/World/envs/env_{env_id}", "Xform")
        env_prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set(f"env_{env_id}")
    instancer = UsdGeom.PointInstancer.Define(stage, "/World/Visuals/markers")
    markers = object.__new__(KitVisualizationMarkers)
    markers.stage = stage
    markers._instancer_manager = instancer
    markers._environment_ids = None
    markers._count = 0

    markers.visualize(
        translations=torch.zeros((2, 3)),
        orientations=None,
        scales=None,
        marker_indices=None,
        environment_ids=torch.tensor([0, 1]),
    )
    primvar = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("omni:scenePartition")
    assert list(primvar.Get()) == ["env_0", "env_1"]

    markers.visualize(
        translations=torch.ones((2, 3)),
        orientations=None,
        scales=None,
        marker_indices=None,
    )
    assert list(primvar.Get()) == ["env_0", "env_1"]

    markers.visualize(
        translations=torch.ones((1, 3)),
        orientations=None,
        scales=None,
        marker_indices=None,
    )

    assert not primvar.GetAttr().HasAuthoredValueOpinion()
