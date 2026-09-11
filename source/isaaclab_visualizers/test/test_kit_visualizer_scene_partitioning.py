# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Kit visualizer scene-partition behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import isaaclab_visualizers.kit.kit_visualizer as kit_visualizer_module
import pytest
import torch
from isaaclab_visualizers.kit.kit_visualization_markers import KitVisualizationMarkers
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg

from pxr import Sdf, Usd, UsdGeom

from isaaclab.utils.renderers import ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING


@pytest.mark.parametrize("color", [(0.1, 0.2, 0.3), None])
def test_background_color_applies_to_render_product_session_layer(
    color: tuple[float, float, float] | None,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    render_product = stage.DefinePrim("/Render/Viewport", "RenderProduct")
    visualizer = KitVisualizer(KitVisualizerCfg(background_color=color))

    visualizer._apply_render_product_background(stage, render_product.GetPath())

    source_type = render_product.GetAttribute("omni:rtx:background:source:type")
    source_color = render_product.GetAttribute("omni:rtx:background:source:color")
    if color is None:
        assert not source_type.IsValid()
        assert not source_color.IsValid()
    else:
        assert source_type.Get() == "color"
        assert tuple(source_color.Get()) == pytest.approx(color)
        assert stage.GetRootLayer().GetAttributeAtPath("/Render/Viewport.omni:rtx:background:source:type") is None
        assert stage.GetRootLayer().GetAttributeAtPath("/Render/Viewport.omni:rtx:background:source:color") is None


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

    visualizer = KitVisualizer(KitVisualizerCfg())
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


def test_viewport_partition_updates_after_center_camera_selection(monkeypatch):
    from isaaclab.sim import SimulationContext

    stage = Usd.Stage.CreateInMemory()
    env_prim = stage.DefinePrim("/World/envs/env_0", "Xform")
    env_prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    camera = UsdGeom.Camera.Define(stage, "/OmniverseKit_Persp")
    camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    viz = KitVisualizer(KitVisualizerCfg(origin_type="env", origin_env_index="center"))
    viz._controlled_camera_path = "/OmniverseKit_Persp"
    viz._resolved_visible_env_ids = [0, 1, 2]
    viz._scene_data_provider = SimpleNamespace(usd_stage=stage, num_envs=3)
    scene = SimpleNamespace(num_envs=3, env_origins=torch.tensor([[-10.0, 0, 0], [10.0, 0, 0], [0.0, 0, 0]]))
    monkeypatch.setattr(SimulationContext, "instance", lambda: SimpleNamespace(_interactive_scene=scene))
    monkeypatch.setattr(kit_visualizer_module, "get_settings_manager", lambda: SimpleNamespace(get=lambda *args: False))
    monkeypatch.setattr(viz, "set_camera_view", lambda *args: None)
    viz._update_camera_tracking()
    assert viz.camera_env_index == 2
    assert camera.GetPrim().GetAttribute("omni:scenePartition").Get() == "env_2"
