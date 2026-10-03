# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SimulationContext tests that check the Kit timeline, Kit app pumping, and PhysX tensor views."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(physics="isaacsim_physx")

from unittest.mock import Mock

import pytest
from isaaclab_physx.physics import PhysxCfg

import omni.physics.tensors
import omni.timeline

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import instantiate

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Setup and teardown for each test."""
    SimulationContext.clear_instance()
    sim_utils.create_new_stage()
    yield
    SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
def test_timeline_play_stop(monkeypatch):
    """Playing shares one native view; stopping releases it before the next play."""
    create_view = Mock(wraps=omni.physics.tensors.create_simulation_view)
    monkeypatch.setattr(omni.physics.tensors, "create_simulation_view", create_view)
    sim = SimulationContext(SimulationCfg(physics=PhysxCfg()))
    scene_data = sim.physics_manager.get_scene_data_backend()
    publication = scene_data.transforms
    cube_cfg = AssetBaseCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
    )
    plan = clone_plan_from_env_0(CloneCfg(), (cube_cfg,), 1, 0.0)
    instantiate(cube_cfg)
    replicate(plan)

    # initially simulation should be stopped
    assert sim.is_stopped()
    assert not sim.is_playing()

    # start the simulation
    sim.play()
    assert sim.is_playing()
    assert not sim.is_stopped()
    resource = scene_data.backend
    assert resource is sim.physics_manager.backend
    view = sim.physics_sim_view
    assert resource.simulation_view is view
    assert scene_data.transforms.transforms is not None
    assert create_view.call_count == 1
    invalidate = Mock(wraps=view.invalidate)
    monkeypatch.setattr(view, "invalidate", invalidate)

    # disable callback to prevent app from continuing
    sim._disable_app_control_on_stop_handle = True  # type: ignore
    # stop the simulation
    sim.stop()
    assert sim.is_stopped()
    assert not sim.is_playing()
    assert sim.physics_sim_view is scene_data.get_rigid_body_view() is None
    assert resource.simulation_view is publication.transforms is None
    assert scene_data.transforms is publication
    resource.close()
    invalidate.assert_called_once_with()

    sim.play()
    assert create_view.call_count == 2
    assert sim.physics_sim_view is not view
    assert scene_data.backend.simulation_view is sim.physics_sim_view
    sim._disable_app_control_on_stop_handle = True
    sim.stop()


@pytest.mark.isaacsim_ci
def test_render_pumps_app_update_without_visualizer():
    """Regression test for issue #5052: headless video must pump Kit when no visualizer does.

    Originally ``SimulationContext.render()`` called ``omni.kit.app.get_app().update()`` when
    no visualizer had ``pumps_app_update()`` (see PR #5056). The same contract is now implemented
    by physics-backend render callbacks registered via :meth:`~SimulationContext.add_render_callback`
    (e.g. ``PhysxManager.initialize()`` registers a headless video pump). These callbacks call
    :func:`~isaaclab_physx.renderers.isaac_rtx_renderer_utils.pump_kit_app_for_headless_video_render_if_needed`
    when ``/isaaclab/video/enabled`` is set (as with ``--video``), which in turn calls
    ``ensure_isaac_rtx_render_update()`` (guarded by ``is_rendering`` and the no-pumping-visualizer check).

    Without this path, replicator render products used for ``rgb_array`` / RecordVideo stay stale (black frames).
    """
    from unittest.mock import MagicMock, patch

    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)
    sim.reset()

    sim.set_setting("/isaaclab/video/enabled", True)
    sim.set_setting("/isaaclab/render/rtx_sensors", True)

    mock_app = MagicMock()
    mock_app.is_running.return_value = True

    with (
        patch("isaaclab.utils.version.has_kit", return_value=True),
        patch(
            "isaaclab_physx.renderers.isaac_rtx_renderer_utils._get_stage_streaming_busy",
            return_value=False,
        ),
        patch("omni.kit.app.get_app", return_value=mock_app),
    ):
        sim.render()

    mock_app.update.assert_called_once()


@pytest.mark.isaacsim_ci
def test_render_skips_app_update_when_visualizer_pumps_it():
    """Regression test: do not pump Kit in the headless-video path when a visualizer already does.

    A visualizer with ``pumps_app_update() == True`` (e.g. KitVisualizer) calls ``app.update()`` in
    its own ``step()``. The render callback registered by the physics backend must then skip
    ``ensure_isaac_rtx_render_update`` so we do not double-pump the Kit loop.
    """
    from unittest.mock import MagicMock, patch

    from isaaclab.visualizers.base_visualizer import BaseVisualizer

    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)
    sim.reset()

    sim.set_setting("/isaaclab/video/enabled", True)
    sim.set_setting("/isaaclab/render/rtx_sensors", True)

    mock_viz = MagicMock(spec=BaseVisualizer)
    mock_viz.pumps_app_update.return_value = True
    mock_viz.is_closed = False
    mock_viz.is_running.return_value = True
    mock_viz.is_rendering_paused.return_value = False
    mock_viz.is_training_paused.return_value = False
    mock_viz.get_rendering_dt.return_value = None
    sim._visualizers = [mock_viz]

    mock_app = MagicMock()
    mock_app.is_running.return_value = True

    with (
        patch("isaaclab.utils.version.has_kit", return_value=True),
        patch("omni.kit.app.get_app", return_value=mock_app),
    ):
        sim.render()

    mock_app.update.assert_not_called()

    sim._visualizers = []


@pytest.mark.isaacsim_ci
def test_timeline_callbacks_on_play():
    """Test that timeline callbacks are triggered on play, pause, and stop events."""
    cfg = SimulationCfg(physics=PhysxCfg(), dt=0.01)
    sim = SimulationContext(cfg)

    # create a simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # create a flag to track callback execution
    callback_state = {"play_called": False, "stop_called": False, "pause_called": False}

    # define callback functions
    def on_play_callback(event):
        callback_state["play_called"] = True

    def on_stop_callback(event):
        callback_state["stop_called"] = True

    def on_pause_callback(event):
        callback_state["pause_called"] = True

    # register callbacks
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    play_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY),
        lambda event: on_play_callback(event),
        order=20,
    )
    stop_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.STOP),
        lambda event: on_stop_callback(event),
        order=20,
    )
    pause_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PAUSE), lambda event: on_pause_callback(event), order=20
    )

    try:
        # ensure callbacks haven't been called yet
        assert not callback_state["play_called"]
        assert not callback_state["stop_called"]

        # play the simulation - this should trigger play callback
        sim.play()
        assert callback_state["play_called"]
        assert not callback_state["stop_called"]

        assert not callback_state["pause_called"]

        # pause the simulation - this should trigger pause callback
        sim.pause()
        assert callback_state["pause_called"]

        # reset flags
        callback_state["play_called"] = False

        # disable app control to prevent hanging
        sim._disable_app_control_on_stop_handle = True  # type: ignore

        # stop the simulation - this should trigger stop callback
        sim.stop()
        assert callback_state["stop_called"]

    finally:
        # cleanup callbacks
        if play_handle is not None:
            play_handle.unsubscribe()
        if stop_handle is not None:
            stop_handle.unsubscribe()
        if pause_handle is not None:
            pause_handle.unsubscribe()
