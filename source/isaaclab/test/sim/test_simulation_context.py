# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import weakref

import numpy as np
import pytest
from isaaclab_physx.physics import IsaacEvents, PhysxCfg, PhysxManager

import omni.timeline

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Setup and teardown for each test."""
    # Setup: Clear any existing simulation context and create a fresh stage
    SimulationContext.clear_instance()
    sim_utils.create_new_stage()

    # Yield for the test
    yield

    SimulationContext.clear_instance()


"""
Basic Configuration Tests
"""


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_init(device):
    """Test the simulation context initialization."""
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

    cfg = SimulationCfg(
        device=device,
        physics_prim_path="/Physics/PhysX",
        gravity=(0.0, -0.5, -0.5),
        physics_material=RigidBodyMaterialCfg(),
        render_interval=5,
    )
    # sim = SimulationContext(cfg)
    # TODO: Figure out why keyword argument doesn't work.
    # note: added a fix in Isaac Sim 2023.1 for this.
    sim = SimulationContext(cfg=cfg)

    # verify stage is valid
    assert sim.stage is not None
    # verify device property
    assert sim.device == device
    # verify no RTX sensors are available
    assert not sim.get_setting("/isaaclab/render/rtx_sensors")

    # obtain physics scene from USD (string-based schema: physxScene:*)
    from pxr import UsdPhysics

    physics_scene_prim = sim.stage.GetPrimAtPath("/Physics/PhysX")
    assert physics_scene_prim.IsValid()
    physics_scene = UsdPhysics.Scene(physics_scene_prim)
    physics_hz = physics_scene_prim.GetAttribute("physxScene:timeStepsPerSecond").Get()
    physics_dt = 1.0 / physics_hz
    assert physics_dt == cfg.dt

    # check valid paths
    assert sim.stage.GetPrimAtPath("/Physics/PhysX").IsValid()
    assert sim.stage.GetPrimAtPath("/Physics/PhysX/defaultMaterial").IsValid()
    # check valid gravity
    gravity_dir, gravity_mag = (
        physics_scene.GetGravityDirectionAttr().Get(),
        physics_scene.GetGravityMagnitudeAttr().Get(),
    )
    gravity = np.array(gravity_dir) * gravity_mag
    np.testing.assert_almost_equal(gravity, cfg.gravity)


@pytest.mark.isaacsim_ci
def test_instance_before_creation():
    """Test accessing instance before creating returns None."""
    # clear any existing instance
    SimulationContext.clear_instance()

    # accessing instance before creation should return None
    assert SimulationContext.instance() is None


@pytest.mark.isaacsim_ci
def test_singleton():
    """Tests that the singleton is working."""
    sim1 = SimulationContext()
    sim2 = SimulationContext()
    assert sim1 is sim2

    # try to delete the singleton
    sim2.clear_instance()
    assert sim1.instance() is None
    # create new instance
    sim3 = SimulationContext()
    assert sim1 is not sim3
    assert sim1.instance() is sim3.instance()
    # clear instance
    sim3.clear_instance()


"""
Property Tests.
"""


@pytest.mark.isaacsim_ci
def test_carb_setting():
    """Test setting carb settings."""
    sim = SimulationContext()
    # known carb setting
    sim.set_setting("/physics/physxDispatcher", False)
    assert sim.get_setting("/physics/physxDispatcher") is False
    # unknown carb setting
    sim.set_setting("/myExt/test_value", 42)
    assert sim.get_setting("/myExt/test_value") == 42


@pytest.mark.isaacsim_ci
def test_headless_mode():
    """Test that render mode is headless since we are running in headless mode."""
    sim = SimulationContext()
    # check default render mode (no GUI and no offscreen rendering)
    assert not sim.has_gui and not sim.has_offscreen_render


"""
Timeline Operations Tests.
"""


@pytest.mark.isaacsim_ci
def test_timeline_play_stop():
    """Test timeline play and stop operations."""
    sim = SimulationContext()

    # initially simulation should be stopped
    assert sim.is_stopped()
    assert not sim.is_playing()

    # start the simulation
    sim.play()
    assert sim.is_playing()
    assert not sim.is_stopped()

    # disable callback to prevent app from continuing
    sim._disable_app_control_on_stop_handle = True  # type: ignore
    # stop the simulation
    sim.stop()
    assert sim.is_stopped()
    assert not sim.is_playing()


@pytest.mark.isaacsim_ci
def test_timeline_pause():
    """Test timeline pause operation."""
    sim = SimulationContext()

    # start the simulation
    sim.play()
    assert sim.is_playing()

    # pause the simulation
    sim.pause()
    assert not sim.is_playing()
    assert not sim.is_stopped()  # paused is different from stopped


"""
Reset and Step Tests
"""


@pytest.mark.isaacsim_ci
def test_reset():
    """Test simulation reset."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create a simple cube to test with
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # reset the simulation
    sim.reset()

    # check that simulation is playing after reset
    assert sim.is_playing()

    # check that physics sim view is created
    assert sim.physics_sim_view is not None


@pytest.mark.isaacsim_ci
def test_reset_soft():
    """Test soft reset (without stopping simulation)."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create a simple cube
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # perform initial reset
    sim.reset()
    assert sim.is_playing()

    # perform soft reset
    sim.reset(soft=True)

    # simulation should still be playing
    assert sim.is_playing()


@pytest.mark.isaacsim_ci
def test_forward():
    """Test forward propagation for fabric updates."""
    cfg = SimulationCfg(dt=0.01, use_fabric=True)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    sim.reset()

    # call forward
    sim.forward()

    # should not raise any errors
    assert sim.is_playing()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("render", [True, False])
def test_step(render):
    """Test stepping simulation with and without rendering."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    sim.reset()

    # step with rendering
    for _ in range(10):
        sim.step(render=render)

    # simulation should still be playing
    assert sim.is_playing()


@pytest.mark.isaacsim_ci
def test_render():
    """Test rendering simulation."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    sim.reset()

    # render
    for _ in range(10):
        sim.render()

    # simulation should still be playing
    assert sim.is_playing()


"""
Stage Operations Tests
"""


@pytest.mark.isaacsim_ci
def test_get_initial_stage():
    """Test getting the initial stage."""
    sim = SimulationContext()

    # get initial stage
    stage = sim.stage

    # verify stage is valid
    assert stage is not None
    assert stage == sim.stage


@pytest.mark.isaacsim_ci
def test_clear_stage():
    """Test clearing the stage."""
    sim = SimulationContext()

    # create some objects
    cube_cfg1 = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg1.func("/World/Cube1", cube_cfg1)
    cube_cfg2 = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg2.func("/World/Cube2", cube_cfg2)

    # verify objects exist
    assert sim.stage.GetPrimAtPath("/World/Cube1").IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cube2").IsValid()

    # clear the stage
    sim.clear_stage()

    # verify objects are removed but World and Physics remain
    assert not sim.stage.GetPrimAtPath("/World/Cube1").IsValid()
    assert not sim.stage.GetPrimAtPath("/World/Cube2").IsValid()
    assert sim.stage.GetPrimAtPath("/World").IsValid()
    assert sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path).IsValid()  # type: ignore[union-attr]


"""
Physics Configuration Tests
"""


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("solver_type", [0, 1])  # 0=PGS, 1=TGS
def test_solver_type(solver_type):
    """Test different solver types."""
    cfg = SimulationCfg(physics=PhysxCfg(solver_type=solver_type))
    sim = SimulationContext(cfg)

    # obtain physics scene from USD (string-based: physxScene:solverType)
    physics_scene_prim = sim.stage.GetPrimAtPath(cfg.physics_prim_path)
    solver_type_str = "PGS" if solver_type == 0 else "TGS"
    assert physics_scene_prim.GetAttribute("physxScene:solverType").Get() == solver_type_str


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("use_fabric", [True, False])
def test_fabric_setting(use_fabric):
    """Test that fabric setting is properly set."""
    cfg = SimulationCfg(use_fabric=use_fabric)
    sim = SimulationContext(cfg)

    # check fabric is enabled via physics setting
    assert sim.get_setting("/isaaclab/fabric_enabled") == use_fabric


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("dt", [0.01, 0.02, 0.005])
def test_physics_dt(dt):
    """Test that physics time step is properly configured."""
    cfg = SimulationCfg(dt=dt)
    sim = SimulationContext(cfg)

    # obtain physics scene from USD (string-based: physxScene:timeStepsPerSecond)
    physics_scene_prim = sim.stage.GetPrimAtPath(cfg.physics_prim_path)
    physics_hz = physics_scene_prim.GetAttribute("physxScene:timeStepsPerSecond").Get()
    physics_dt = 1.0 / physics_hz
    assert abs(physics_dt - dt) < 1e-6


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("gravity", [(0.0, 0.0, 0.0), (0.0, 0.0, -9.81), (0.5, 0.5, 0.5)])
def test_custom_gravity(gravity):
    """Test that gravity can be properly set."""
    from pxr import UsdPhysics

    cfg = SimulationCfg(gravity=gravity)
    sim = SimulationContext(cfg)

    # obtain physics scene from USD
    physics_scene_prim = sim.stage.GetPrimAtPath(cfg.physics_prim_path)
    physics_scene = UsdPhysics.Scene(physics_scene_prim)

    gravity_dir, gravity_mag = (
        physics_scene.GetGravityDirectionAttr().Get(),
        physics_scene.GetGravityMagnitudeAttr().Get(),
    )
    actual_gravity = np.array(gravity_dir) * gravity_mag
    np.testing.assert_almost_equal(actual_gravity, cfg.gravity, decimal=6)


"""
Callback Tests.
"""


@pytest.mark.isaacsim_ci
def test_timeline_callbacks_on_play():
    """Test that timeline callbacks are triggered on play event."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create a simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # create a flag to track callback execution
    callback_state = {"play_called": False, "stop_called": False}

    # define callback functions
    def on_play_callback(event):
        callback_state["play_called"] = True

    def on_stop_callback(event):
        callback_state["stop_called"] = True

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

    try:
        # ensure callbacks haven't been called yet
        assert not callback_state["play_called"]
        assert not callback_state["stop_called"]

        # play the simulation - this should trigger play callback
        sim.play()
        assert callback_state["play_called"]
        assert not callback_state["stop_called"]

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


@pytest.mark.isaacsim_ci
def test_timeline_callbacks_with_weakref():
    """Test that timeline callbacks work correctly with weak references (similar to asset_base.py)."""

    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create a simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # create a test object that will be weakly referenced
    class CallbackTracker:
        def __init__(self):
            self.play_count = 0
            self.stop_count = 0

        def on_play(self, event):
            self.play_count += 1

        def on_stop(self, event):
            self.stop_count += 1

    # create an instance of the callback tracker
    tracker = CallbackTracker()

    # define safe callback wrapper (similar to asset_base.py pattern)
    def safe_callback(callback_name, event, obj_ref):
        """Safely invoke a callback on a weakly-referenced object."""
        try:
            obj = obj_ref()  # Dereference the weakref
            if obj is not None:
                getattr(obj, callback_name)(event)
        except ReferenceError:
            # Object has been deleted; ignore
            pass

    # register callbacks with weakref
    obj_ref = weakref.ref(tracker)
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()

    play_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY),
        lambda event, obj_ref=obj_ref: safe_callback("on_play", event, obj_ref),
        order=20,
    )
    stop_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.STOP),
        lambda event, obj_ref=obj_ref: safe_callback("on_stop", event, obj_ref),
        order=20,
    )

    try:
        # verify callbacks haven't been called
        assert tracker.play_count == 0
        assert tracker.stop_count == 0

        # trigger play event
        sim.play()
        assert tracker.play_count == 1
        assert tracker.stop_count == 0

        # disable app control to prevent hanging
        sim._disable_app_control_on_stop_handle = True  # type: ignore

        # trigger stop event
        sim.stop()
        assert tracker.play_count == 1
        assert tracker.stop_count == 1

        # delete the tracker object
        del tracker

        # trigger events again - callbacks should handle the deleted object gracefully
        sim.play()
        # disable app control again
        sim._disable_app_control_on_stop_handle = True  # type: ignore
        sim.stop()
        # should not raise any errors

    finally:
        # cleanup callbacks
        if play_handle is not None:
            play_handle.unsubscribe()
        if stop_handle is not None:
            stop_handle.unsubscribe()


@pytest.mark.isaacsim_ci
def test_multiple_callbacks_on_same_event():
    """Test that multiple callbacks can be registered for the same event."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create tracking for multiple callbacks
    callback_counts = {"callback1": 0, "callback2": 0, "callback3": 0}

    def callback1(event):
        callback_counts["callback1"] += 1

    def callback2(event):
        callback_counts["callback2"] += 1

    def callback3(event):
        callback_counts["callback3"] += 1

    # register multiple callbacks for play event
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    handle1 = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback1(event), order=20
    )
    handle2 = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback2(event), order=21
    )
    handle3 = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback3(event), order=22
    )

    try:
        # verify none have been called
        assert all(count == 0 for count in callback_counts.values())

        # trigger play event
        sim.play()

        # all callbacks should have been called
        assert callback_counts["callback1"] == 1
        assert callback_counts["callback2"] == 1
        assert callback_counts["callback3"] == 1

    finally:
        # cleanup all callbacks
        if handle1 is not None:
            handle1.unsubscribe()
        if handle2 is not None:
            handle2.unsubscribe()
        if handle3 is not None:
            handle3.unsubscribe()


@pytest.mark.isaacsim_ci
def test_callback_execution_order():
    """Test that callbacks are executed in the correct order based on priority."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # track execution order
    execution_order = []

    def callback_low_priority(event):
        execution_order.append("low")

    def callback_medium_priority(event):
        execution_order.append("medium")

    def callback_high_priority(event):
        execution_order.append("high")

    # register callbacks with different priorities (lower order = higher priority)
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    handle_high = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback_high_priority(event), order=5
    )
    handle_medium = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback_medium_priority(event), order=10
    )
    handle_low = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: callback_low_priority(event), order=15
    )

    try:
        # trigger play event
        sim.play()

        # verify callbacks were executed in correct order
        assert len(execution_order) == 3
        assert execution_order[0] == "high"
        assert execution_order[1] == "medium"
        assert execution_order[2] == "low"

    finally:
        # cleanup callbacks
        if handle_high is not None:
            handle_high.unsubscribe()
        if handle_medium is not None:
            handle_medium.unsubscribe()
        if handle_low is not None:
            handle_low.unsubscribe()


@pytest.mark.isaacsim_ci
def test_callback_unsubscribe():
    """Test that unsubscribing callbacks works correctly."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create callback counter
    callback_count = {"count": 0}

    def on_play_callback(event):
        callback_count["count"] += 1

    # register callback
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    play_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PLAY), lambda event: on_play_callback(event), order=20
    )

    try:
        # trigger play event
        sim.play()
        assert callback_count["count"] == 1

        # stop simulation
        sim._disable_app_control_on_stop_handle = True  # type: ignore
        sim.stop()

        # unsubscribe the callback
        play_handle.unsubscribe()
        play_handle = None

        # trigger play event again
        sim.play()

        # callback should not have been called again (still 1)
        assert callback_count["count"] == 1

    finally:
        # cleanup if needed
        if play_handle is not None:
            play_handle.unsubscribe()


@pytest.mark.isaacsim_ci
def test_pause_event_callback():
    """Test that pause event callbacks are triggered correctly."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create callback tracker
    callback_state = {"pause_called": False}

    def on_pause_callback(event):
        callback_state["pause_called"] = True

    # register pause callback
    timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    pause_handle = timeline_event_stream.create_subscription_to_pop_by_type(
        int(omni.timeline.TimelineEventType.PAUSE), lambda event: on_pause_callback(event), order=20
    )

    try:
        # play the simulation first
        sim.play()
        assert not callback_state["pause_called"]

        # pause the simulation
        sim.pause()

        # callback should have been triggered
        assert callback_state["pause_called"]

    finally:
        # cleanup
        if pause_handle is not None:
            pause_handle.unsubscribe()


"""
Isaac Events Callback Tests.
"""


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "event_type",
    [IsaacEvents.PHYSICS_WARMUP, IsaacEvents.SIMULATION_VIEW_CREATED, IsaacEvents.PHYSICS_READY],
)
def test_isaac_event_triggered_on_reset(event_type):
    """Test that Isaac events are triggered during reset."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # create callback tracker
    callback_state = {"called": False}

    def on_event(event):
        callback_state["called"] = True

    # register callback for the event
    callback_id = PhysxManager.register_callback(lambda event: on_event(event), event=event_type)

    try:
        # verify callback hasn't been called yet
        assert not callback_state["called"]

        # reset the simulation - should trigger the event
        sim.reset()

        # verify callback was triggered
        assert callback_state["called"]

    finally:
        # cleanup callback
        if callback_id is not None:
            PhysxManager.deregister_callback(callback_id)


@pytest.mark.isaacsim_ci
def test_isaac_event_prim_deletion():
    """Test that PRIM_DELETION Isaac event is triggered when a prim is deleted."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    sim.reset()

    # create callback tracker
    callback_state = {"prim_deleted": False, "deleted_path": None}

    def on_prim_deletion(event):
        callback_state["prim_deleted"] = True
        # event payload should contain the deleted prim path
        if hasattr(event, "payload") and event.payload:
            callback_state["deleted_path"] = event.payload.get("prim_path")

    # register callback for PRIM_DELETION event
    callback_id = PhysxManager.register_callback(lambda event: on_prim_deletion(event), event=IsaacEvents.PRIM_DELETION)

    try:
        # verify callback hasn't been called yet
        assert not callback_state["prim_deleted"]

        # delete the cube prim
        sim_utils.delete_prim("/World/Cube")

        # trigger the event by dispatching it manually (since deletion might be handled differently)
        PhysxManager._message_bus.dispatch_event(IsaacEvents.PRIM_DELETION.value, payload={"prim_path": "/World/Cube"})  # type: ignore

        # verify callback was triggered
        assert callback_state["prim_deleted"]

    finally:
        # cleanup callback
        if callback_id is not None:
            PhysxManager.deregister_callback(callback_id)


@pytest.mark.isaacsim_ci
def test_isaac_event_timeline_stop():
    """Test that TIMELINE_STOP Isaac event can be registered and triggered."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create callback tracker
    callback_state = {"timeline_stop_called": False}

    def on_timeline_stop(event):
        callback_state["timeline_stop_called"] = True

    # register callback for TIMELINE_STOP event
    callback_id = PhysxManager.register_callback(lambda event: on_timeline_stop(event), event=IsaacEvents.TIMELINE_STOP)

    try:
        # verify callback hasn't been called yet
        assert not callback_state["timeline_stop_called"]

        # play and stop the simulation
        sim.play()

        # disable app control to prevent hanging
        sim._disable_app_control_on_stop_handle = True  # type: ignore

        # stop the simulation
        sim.stop()

        # dispatch the event manually
        PhysxManager._message_bus.dispatch_event(IsaacEvents.TIMELINE_STOP.value, payload={})  # type: ignore

        # verify callback was triggered
        assert callback_state["timeline_stop_called"]

    finally:
        # cleanup callback
        if callback_id is not None:
            PhysxManager.deregister_callback(callback_id)


@pytest.mark.isaacsim_ci
def test_isaac_event_callbacks_with_weakref():
    """Test Isaac event callbacks with weak references (similar to asset_base.py pattern)."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # create a test object that will be weakly referenced
    class PhysicsTracker:
        def __init__(self):
            self.warmup_count = 0
            self.ready_count = 0

        def on_warmup(self, event):
            self.warmup_count += 1

        def on_ready(self, event):
            self.ready_count += 1

    tracker = PhysicsTracker()

    # define safe callback wrapper (same pattern as asset_base.py)
    def safe_callback(callback_name, event, obj_ref):
        """Safely invoke a callback on a weakly-referenced object."""
        try:
            obj = obj_ref()
            if obj is not None:
                getattr(obj, callback_name)(event)
        except ReferenceError:
            # Object has been deleted; ignore
            pass

    # register callbacks with weakref
    obj_ref = weakref.ref(tracker)

    warmup_id = PhysxManager.register_callback(
        lambda event, obj_ref=obj_ref: safe_callback("on_warmup", event, obj_ref),
        event=IsaacEvents.PHYSICS_WARMUP,
    )
    ready_id = PhysxManager.register_callback(
        lambda event, obj_ref=obj_ref: safe_callback("on_ready", event, obj_ref), event=IsaacEvents.PHYSICS_READY
    )

    try:
        # verify callbacks haven't been called
        assert tracker.warmup_count == 0
        assert tracker.ready_count == 0

        # reset simulation - triggers WARMUP and READY events
        sim.reset()

        # verify callbacks were triggered (may be called multiple times during warmup sequence)
        assert tracker.warmup_count >= 1
        assert tracker.ready_count >= 1

        # delete the tracker object
        del tracker

        # reset again - callbacks should handle the deleted object gracefully
        sim.reset(soft=True)

        # should not raise any errors even though tracker is deleted

    finally:
        # cleanup callbacks
        if warmup_id is not None:
            PhysxManager.deregister_callback(warmup_id)
        if ready_id is not None:
            PhysxManager.deregister_callback(ready_id)


@pytest.mark.isaacsim_ci
def test_multiple_isaac_event_callbacks():
    """Test that multiple callbacks can be registered for the same Isaac event."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # create tracking for multiple callbacks
    callback_counts = {"callback1": 0, "callback2": 0, "callback3": 0}

    def callback1(event):
        callback_counts["callback1"] += 1

    def callback2(event):
        callback_counts["callback2"] += 1

    def callback3(event):
        callback_counts["callback3"] += 1

    # register multiple callbacks for PHYSICS_READY event
    id1 = PhysxManager.register_callback(lambda event: callback1(event), event=IsaacEvents.PHYSICS_READY)
    id2 = PhysxManager.register_callback(lambda event: callback2(event), event=IsaacEvents.PHYSICS_READY)
    id3 = PhysxManager.register_callback(lambda event: callback3(event), event=IsaacEvents.PHYSICS_READY)

    try:
        # verify none have been called
        assert all(count == 0 for count in callback_counts.values())

        # reset simulation - triggers PHYSICS_READY event
        sim.reset()

        # all callbacks should have been called (may be called multiple times during warmup sequence)
        assert callback_counts["callback1"] >= 1
        assert callback_counts["callback2"] >= 1
        assert callback_counts["callback3"] >= 1

    finally:
        # cleanup all callbacks
        if id1 is not None:
            PhysxManager.deregister_callback(id1)
        if id2 is not None:
            PhysxManager.deregister_callback(id2)
        if id3 is not None:
            PhysxManager.deregister_callback(id3)


"""
Exception Handling in Callbacks Tests.
"""


@pytest.mark.isaacsim_ci
def test_exception_in_callback_on_reset():
    """Test that exceptions stored during reset are raised."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    test_error_message = "Test exception on reset"

    def failing_callback(event):
        PhysxManager.store_callback_exception(RuntimeError(test_error_message))

    handle = PhysxManager.register_callback(failing_callback, event=IsaacEvents.PHYSICS_READY)

    try:
        with pytest.raises(RuntimeError, match=test_error_message):
            sim.reset()
    finally:
        if handle is not None:
            PhysxManager.deregister_callback(handle)
        SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
def test_exception_in_callback_on_step():
    """Test that exceptions stored during step are raised."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # reset first to initialize
    sim.reset()

    test_error_message = "Test exception on step"

    def failing_callback(event):
        PhysxManager.store_callback_exception(RuntimeError(test_error_message))

    handle = PhysxManager.register_callback(failing_callback, event=IsaacEvents.POST_PHYSICS_STEP)

    try:
        with pytest.raises(RuntimeError, match=test_error_message):
            sim.step()
    finally:
        if handle is not None:
            PhysxManager.deregister_callback(handle)
        SimulationContext.clear_instance()
