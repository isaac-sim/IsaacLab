# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Kit visualizer scene-partition behavior."""

import contextlib
import signal
import sys
from types import ModuleType
from unittest.mock import MagicMock

import isaaclab_visualizers.kit.kit_visualizer as kit_visualizer_module
import pytest
import torch
from isaaclab_visualizers.kit.kit_visualization_markers import KitVisualizationMarkers
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg

import omni
from pxr import Sdf, Usd, UsdGeom, UsdLux

from isaaclab.utils.renderers import ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING


@pytest.fixture(autouse=True)
def reset_kit_timeline_control_state():
    KitVisualizer._timeline_auto_update = None
    KitVisualizer._timeline_control_owner_count = 0
    yield
    KitVisualizer._timeline_auto_update = None
    KitVisualizer._timeline_control_owner_count = 0


@pytest.fixture
def kit_app(monkeypatch):
    app = MagicMock()
    app.is_running.return_value = True
    app_module = ModuleType("omni.kit.app")
    app_module.get_app = lambda: app
    kit_module = ModuleType("omni.kit")
    kit_module.app = app_module
    monkeypatch.setitem(sys.modules, "omni.kit", kit_module)
    monkeypatch.setitem(sys.modules, "omni.kit.app", app_module)
    monkeypatch.setattr(omni, "kit", kit_module, raising=False)
    return app


@pytest.fixture
def kit_timeline(monkeypatch):
    timeline = MagicMock()
    timeline_module = ModuleType("omni.timeline")
    timeline_module.get_timeline_interface = lambda: timeline
    monkeypatch.setitem(sys.modules, "omni.timeline", timeline_module)
    monkeypatch.setattr(omni, "timeline", timeline_module, raising=False)
    return timeline


@pytest.mark.parametrize("headless", [False, True])
def test_viewport_pose_publication_is_deferred_for_headless_capture(monkeypatch, kit_app, headless):
    visualizer = KitVisualizer(KitVisualizerCfg(headless=headless, origin_type="asset"))
    visualizer._is_initialized = True
    visualizer._fabric = MagicMock()
    visualizer._sim = sim = MagicMock(render_generation=12)
    provider = sim.get_scene_data_provider.return_value
    monkeypatch.setattr(visualizer, "is_training_paused", lambda: True)
    tracking = MagicMock()
    monkeypatch.setattr(visualizer, "_update_asset_tracking_camera", tracking)
    monkeypatch.setattr(visualizer, "_refresh_partial_viz_point_instancers_if_needed", MagicMock())
    settings = MagicMock()
    settings.get.return_value = None
    monkeypatch.setattr(kit_visualizer_module, "get_settings_manager", lambda: settings)

    visualizer.step(0.1)

    assert tracking.call_count == int(not headless)
    if headless:
        visualizer._fabric.update_transforms.assert_not_called()
        visualizer._fabric.update_geometries.assert_not_called()
        settings.set.assert_not_called()
        kit_app.update.assert_not_called()
    else:
        visualizer._fabric.update_transforms.assert_called_once_with(provider)
        visualizer._fabric.update_geometries.assert_called_once_with(provider, 12)
        assert [call.args for call in settings.set.call_args_list] == [
            ("/app/player/playSimulations", False),
            ("/app/player/playSimulations", True),
        ]
        kit_app.update.assert_called_once_with()


def test_app_update_restores_signal_handlers(monkeypatch, kit_app):
    """Defer Kit-swallowed interrupts without replacing an existing forced-exit handler."""
    visualizer = KitVisualizer(KitVisualizerCfg())
    visualizer._is_initialized = True
    visualizer._fabric = MagicMock()
    visualizer._sim = MagicMock(render_generation=1)
    monkeypatch.setattr(visualizer, "_refresh_partial_viz_point_instancers_if_needed", MagicMock())
    settings = MagicMock()
    settings.get.return_value = False
    monkeypatch.setattr(kit_visualizer_module, "get_settings_manager", lambda: settings)
    previous_handler = signal.getsignal(signal.SIGINT)

    def update():
        with contextlib.suppress(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
        assert signal.getsignal(signal.SIGINT) == signal.SIG_DFL

    kit_app.update.side_effect = update

    with pytest.raises(KeyboardInterrupt):
        visualizer.step(0.1)

    assert signal.getsignal(signal.SIGINT) is previous_handler
    assert [call.args for call in settings.set.call_args_list] == [
        ("/app/player/playSimulations", False),
        ("/app/player/playSimulations", False),
    ]

    def update_with_forced_exit():
        assert signal.getsignal(signal.SIGINT) == signal.SIG_DFL

    kit_app.update.side_effect = update_with_forced_exit
    previous_handler = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        visualizer.step(0.1)
        assert signal.getsignal(signal.SIGINT) == signal.SIG_DFL
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def test_timeline_controls_share_ownership_and_restore_after_last_stop(monkeypatch, kit_app, kit_timeline):
    kit_timeline.is_auto_updating.return_value = True
    settings = MagicMock()
    settings.get.return_value = True
    monkeypatch.setattr(kit_visualizer_module, "get_settings_manager", lambda: settings)
    visualizers = [KitVisualizer(KitVisualizerCfg()), KitVisualizer(KitVisualizerCfg())]
    for visualizer in visualizers:
        visualizer._controls_kit_timeline = True

    visualizers[0].play()

    assert kit_timeline.is_auto_updating.call_count == 1
    kit_timeline.play.assert_called_once_with()
    kit_timeline.is_playing.return_value = False
    kit_timeline.is_stopped.return_value = False
    assert visualizers[0].is_training_paused()
    kit_timeline.is_stopped.return_value = True
    assert not visualizers[0].is_training_paused()

    visualizers[0].pause()
    kit_timeline.pause.assert_called_once_with()
    visualizers[1].play()

    assert kit_timeline.is_auto_updating.call_count == 1
    assert [call.args for call in kit_timeline.set_auto_update.call_args_list] == [(False,), (False,)]

    visualizers[0].stop()

    kit_timeline.stop.assert_not_called()
    assert [call.args for call in kit_timeline.set_auto_update.call_args_list] == [(False,), (False,)]

    visualizers[1].stop()

    kit_timeline.stop.assert_called_once_with()
    assert [call.args for call in kit_timeline.set_auto_update.call_args_list] == [(False,), (False,), (True,)]
    assert kit_app.update.call_count == 4


def test_close_cleans_resources_when_timeline_stop_is_interrupted(monkeypatch, kit_timeline):
    visualizer = KitVisualizer(KitVisualizerCfg())
    visualizer._is_initialized = True
    visualizer._controls_kit_timeline = True
    visualizer._owns_kit_timeline = True
    KitVisualizer._timeline_auto_update = True
    KitVisualizer._timeline_control_owner_count = 1
    teardown_label = MagicMock()
    restore_visibility = MagicMock()
    monkeypatch.setattr(visualizer, "_teardown_backend_menubar_label", teardown_label)
    monkeypatch.setattr(visualizer, "_restore_env_visibility", restore_visibility)
    monkeypatch.setattr(visualizer, "_pump_kit_app_update", MagicMock(side_effect=KeyboardInterrupt))
    annotator = visualizer._rgb_annotator = MagicMock()
    render_product = visualizer._rgb_render_product = MagicMock()

    with pytest.raises(KeyboardInterrupt):
        visualizer.close()

    kit_timeline.stop.assert_called_once_with()
    kit_timeline.set_auto_update.assert_called_once_with(True)
    teardown_label.assert_called_once_with()
    restore_visibility.assert_called_once_with()
    annotator.detach.assert_called_once_with()
    render_product.destroy.assert_called_once_with()
    assert visualizer._rgb_annotator is None
    assert visualizer._rgb_render_product is None
    assert visualizer.is_closed
    assert not visualizer._is_initialized


@pytest.mark.parametrize(
    ("physics_backend", "controls_timeline"),
    [("ovphysx", False), ("newton", True)],
)
def test_timeline_controls_only_non_physx_backends(monkeypatch, physics_backend, controls_timeline):
    visualizer = KitVisualizer(KitVisualizerCfg())
    sim = MagicMock(has_gui=True, stage=Usd.Stage.CreateInMemory(), device="cpu")
    sim.get_scene_data_provider.return_value.num_envs = 0
    fabric = sim.get_or_create_backend.return_value
    fabric_module = ModuleType("isaaclab_physx.renderers.fabric")
    fabric_module.FabricBackendCfg = MagicMock()
    monkeypatch.setitem(sys.modules, "isaaclab_physx.renderers.fabric", fabric_module)
    monkeypatch.setattr(KitVisualizer, "physics_backend", property(lambda _: physics_backend))
    monkeypatch.setattr(visualizer, "_ensure_simulation_app", MagicMock())
    monkeypatch.setattr(visualizer, "_setup_viewport", MagicMock())
    monkeypatch.setattr(visualizer, "_apply_viewport_camera_scene_partition", MagicMock())
    monkeypatch.setattr(visualizer, "_log_initialization_table", MagicMock())
    monkeypatch.setattr(visualizer, "_setup_initial_camera_view", MagicMock())
    play = MagicMock()
    monkeypatch.setattr(visualizer, "play", play)

    visualizer.initialize(sim, cameras=[])

    assert visualizer._controls_kit_timeline is controls_timeline
    assert play.call_count == int(controls_timeline)
    fabric.bind_transforms.assert_called_once_with(sim.get_scene_data_provider.return_value)


@pytest.mark.parametrize("color", [(0.1, 0.2, 0.3), None])
def test_background_color_preserves_scene_lighting(
    color: tuple[float, float, float] | None,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    render_product = stage.DefinePrim("/Render/Viewport", "RenderProduct")
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.GetTextureFileAttr().Set(Sdf.AssetPath("sky.hdr"))
    dome.GetIntensityAttr().Set(750.0)
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
    assert dome.GetTextureFileAttr().Get().path == "sky.hdr"
    assert dome.GetIntensityAttr().Get() == 750.0


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
    visualizer._env_ids = [2]
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


def test_marker_environment_ids_are_sticky_and_revalidated_on_change() -> None:
    """Preserve ownership for stable counts and reject changed invalid IDs."""
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

    environment_ids = torch.tensor([0, 1])
    markers.visualize(
        translations=torch.zeros((2, 3)),
        orientations=None,
        scales=None,
        marker_indices=None,
        environment_ids=environment_ids,
    )
    primvar = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("omni:scenePartition")
    assert list(primvar.Get()) == ["env_0", "env_1"]

    environment_ids[0] = -1
    with pytest.raises(ValueError, match="non-negative indices"):
        markers.visualize(None, None, None, None, environment_ids)
    environment_ids[0] = 0

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
