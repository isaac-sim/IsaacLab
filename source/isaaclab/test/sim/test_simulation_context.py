# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from isaaclab.test.utils import resolve_test_sim_device, test_devices

# launch omniverse app
simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

"""Rest everything follows."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import IsaacEvents, PhysxCfg, PhysxManager

import omni.physics.tensors
import omni.timeline
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysicsEvent
from isaaclab.renderers import RendererCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.visualizers.base_visualizer import BaseVisualizer

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def fresh_stage():
    """Start every test without a live context and on a new stage."""
    SimulationContext.clear_instance()
    sim_utils.create_new_stage()
    yield
    SimulationContext.clear_instance()


def _spawn_cube(prim_path: str = "/World/Cube") -> None:
    cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cfg.func(prim_path, cfg)


def _scene_gravity(sim: SimulationContext) -> np.ndarray:
    scene = UsdPhysics.Scene(sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path))
    return np.array(scene.GetGravityDirectionAttr().Get()) * scene.GetGravityMagnitudeAttr().Get()


def _stop(sim: SimulationContext) -> None:
    """Stop without handing control back to the app, which would block the test."""
    sim._disable_app_control_on_stop_handle = True  # type: ignore[attr-defined]
    sim.stop()


"""
Configuration
"""


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("device", test_devices())
def test_init(device, monkeypatch):
    """Construction authors the physics scene, selects the device, and exposes settings."""
    original_initialize = PhysxManager.initialize.__func__

    def initialize(cls, sim_context):
        assert wp.get_device() == wp.get_device(device)
        return original_initialize(cls, sim_context)

    monkeypatch.setattr(PhysxManager, "initialize", classmethod(initialize))
    cfg = SimulationCfg(
        device=device,
        physics_prim_path="/Physics/PhysX",
        gravity=(0.0, -0.5, -0.5),
        physics_material=RigidBodyMaterialCfg(),
        render_interval=5,
    )
    sim = SimulationContext(cfg=cfg)

    assert sim.stage is not None
    assert sim.device == device
    assert not sim.get_setting("/isaaclab/render/rtx_sensors")
    # headless: no GUI and no offscreen rendering
    assert not sim.has_gui and not sim.has_offscreen_render

    physics_scene_prim = sim.stage.GetPrimAtPath("/Physics/PhysX")
    assert physics_scene_prim.IsValid()
    assert sim.stage.GetPrimAtPath("/Physics/PhysX/defaultMaterial").IsValid()
    assert 1.0 / physics_scene_prim.GetAttribute("physxScene:timeStepsPerSecond").Get() == cfg.dt
    np.testing.assert_almost_equal(_scene_gravity(sim), cfg.gravity)

    # known and unknown carb settings round-trip
    sim.set_setting("/physics/physxDispatcher", False)
    assert sim.get_setting("/physics/physxDispatcher") is False
    sim.set_setting("/myExt/test_value", 42)
    assert sim.get_setting("/myExt/test_value") == 42


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    ("solver_type", "use_fabric", "dt", "gravity"),
    [(0, True, 0.02, (0.0, 0.0, 0.0)), (1, False, 0.005, (0.5, 0.5, 0.5))],
    ids=["pgs", "tgs"],
)
def test_physics_scene_authoring(solver_type, use_fabric, dt, gravity):
    """Solver, fabric, time step, and gravity settings reach the USD physics scene."""
    cfg = SimulationCfg(physics=PhysxCfg(solver_type=solver_type), use_fabric=use_fabric, dt=dt, gravity=gravity)
    sim = SimulationContext(cfg)

    physics_scene_prim = sim.stage.GetPrimAtPath(cfg.physics_prim_path)
    assert physics_scene_prim.GetAttribute("physxScene:solverType").Get() == ("PGS", "TGS")[solver_type]
    assert sim.get_setting("/isaaclab/fabric_enabled") == use_fabric
    assert 1.0 / physics_scene_prim.GetAttribute("physxScene:timeStepsPerSecond").Get() == pytest.approx(dt)
    np.testing.assert_almost_equal(_scene_gravity(sim), gravity, decimal=6)


@pytest.mark.isaacsim_ci
def test_singleton():
    """Construction creates a context; only instance() retrieves the live context."""
    assert SimulationContext.instance() is None

    sim = SimulationContext(SimulationCfg(dt=0.01))
    live_device, live_dt = sim.cfg.device, sim.cfg.dt
    other_device = "cpu" if live_device.startswith("cuda") else "cuda:0"
    for args in (
        (),
        (None,),
        (sim.cfg,),
        (sim.cfg.copy(),),
        (sim.cfg.replace(dt=2.0 * live_dt),),
        (sim.cfg.replace(device=other_device),),
    ):
        with pytest.raises(RuntimeError, match=r"SimulationContext\.instance\(\)"):
            SimulationContext(*args)
        assert SimulationContext.instance() is sim
        assert sim.cfg.dt == live_dt
        assert sim.cfg.device == sim.device == live_device

    SimulationContext.clear_instance()
    assert SimulationContext.instance() is None
    replacement = SimulationContext(SimulationCfg(dt=2.0 * live_dt))
    assert replacement is not sim
    assert SimulationContext.instance() is replacement
    assert replacement.cfg.dt == 2.0 * live_dt


"""
Teardown
"""


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "physics_cfg",
    [PhysxCfg(), NewtonCfg(solver_cfg=MJWarpSolverCfg())],
    ids=["physx", "newton"],
)
def test_stop_is_dispatched_for_lazy_class_type(physics_cfg):
    """``PhysicsEvent.STOP`` must be dispatched even when ``class_type`` is declared lazily.

    Configs declare ``class_type`` as a ``"module:Class"`` string, which proxies attribute access
    but is a ``str``. The active-manager identity check in ``PhysicsManager.close`` therefore
    never matched the class, and ``STOP`` never reached any sensor or asset.
    """
    sim = SimulationContext(SimulationCfg(physics=physics_cfg))
    stopped = []
    sim.physics_manager.register_callback(lambda _: stopped.append(True), PhysicsEvent.STOP, name="test_stop")

    SimulationContext.clear_instance()

    assert stopped, "PhysicsEvent.STOP was not dispatched at teardown"


@pytest.mark.isaacsim_ci
def test_clear_instance_closes_renderers():
    """``clear_instance`` must close registered renderers rather than leave them to garbage collection.

    A renderer is shared by every camera whose config resolves to it, so the stage-bound resources it
    owns cannot be released from a single camera's ``cleanup``. Nothing else releases them either:
    the OVRTX ovstage path holds its stage in a ``contextlib.ExitStack``, which has no finalizer, so
    collection never runs the context managers that own it.
    """
    sim = SimulationContext(SimulationCfg(physics=PhysxCfg()))
    closed = []

    class _Renderer:
        def __init__(self, cfg):
            pass

        def close(self):
            closed.append(True)

    sim.get_or_create_backend(RendererCfg(class_type=_Renderer))
    SimulationContext.clear_instance()

    assert closed, "registered renderers were not closed at teardown"


@pytest.mark.isaacsim_ci
def test_clear_stage():
    """Clearing the stage removes spawned prims but keeps /World and the physics scene."""
    sim = SimulationContext()
    _spawn_cube("/World/Cube1")
    _spawn_cube("/World/Cube2")
    assert sim.stage.GetPrimAtPath("/World/Cube1").IsValid()

    sim.clear_stage()

    assert not sim.stage.GetPrimAtPath("/World/Cube1").IsValid()
    assert not sim.stage.GetPrimAtPath("/World/Cube2").IsValid()
    assert sim.stage.GetPrimAtPath("/World").IsValid()
    assert sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path).IsValid()


"""
Timeline
"""


@pytest.mark.isaacsim_ci
def test_timeline_play_stop(monkeypatch):
    """Playing shares one native view; stopping releases it before the next play."""
    create_view = Mock(wraps=omni.physics.tensors.create_simulation_view)
    monkeypatch.setattr(omni.physics.tensors, "create_simulation_view", create_view)
    sim = SimulationContext()
    scene_data = sim.physics_manager.get_scene_data_backend()
    publication = scene_data.transforms
    cube_cfg = sim_utils.CuboidCfg(
        size=(0.1, 0.1, 0.1),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )
    cube_cfg.func("/World/Cube", cube_cfg)

    assert sim.is_stopped()
    assert not sim.is_playing()

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

    _stop(sim)
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
    _stop(sim)


@pytest.mark.isaacsim_ci
def test_timeline_events():
    """Timeline subscriptions fire on play/pause/stop in priority order and stop after unsubscribing."""
    sim = SimulationContext(SimulationCfg(dt=0.01))
    _spawn_cube()

    events: list[str] = []
    stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
    handles = [
        stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY), lambda _event, name=name: events.append(name), order=order
        )
        for name, order in (("play_low", 15), ("play_high", 5), ("play_medium", 10))
    ]
    handles += [
        stream.create_subscription_to_pop_by_type(
            int(getattr(omni.timeline.TimelineEventType, kind)), lambda _event, kind=kind: events.append(kind), order=20
        )
        for kind in ("PAUSE", "STOP")
    ]

    try:
        sim.play()
        assert events == ["play_high", "play_medium", "play_low"]

        sim.pause()
        assert not sim.is_playing() and not sim.is_stopped()
        assert events[-1] == "PAUSE"

        events.clear()
        sim.play()
        _stop(sim)
        assert events == ["play_high", "play_medium", "play_low", "STOP"]

        # unsubscribed callbacks no longer fire; play() after stop() must also recreate the view
        # instead of crashing in PhysX's tensor view registry (see PhysxManager._on_stop)
        for handle in handles[:3]:
            handle.unsubscribe()
        events.clear()
        sim.play()
        assert events == []
        assert PhysxManager.get_physics_sim_view() is not None
        _stop(sim)
    finally:
        for handle in handles[3:]:
            handle.unsubscribe()


"""
Reset, step, and render
"""


@pytest.mark.isaacsim_ci
def test_reset_step_render():
    """Reset starts playing and creates the physics view; step and render advance their counters."""
    sim = SimulationContext(SimulationCfg(dt=0.01))
    _spawn_cube()

    sim.reset()
    assert sim.is_playing()
    assert sim.physics_sim_view is not None

    sim.reset(soft=True)
    assert sim.is_playing()

    sim.forward()
    for render in (True, False):
        sim.step(render=render)
    assert sim.get_physics_step_count() == 2

    render_generation = sim.render_generation
    sim.render()
    assert sim.render_generation == render_generation + 1
    assert sim.is_playing()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("visualizer_pumps", [False, True], ids=["no_visualizer", "pumping_visualizer"])
def test_headless_video_pumps_kit_once(visualizer_pumps):
    """Regression test for issue #5052: headless video must pump Kit exactly once per render.

    The physics backend registers a render callback (see ``PhysxManager.initialize``) that pumps
    ``app.update()`` when ``/isaaclab/video/enabled`` is set and no visualizer already does so.
    Without it, replicator render products used for ``rgb_array`` / RecordVideo stay stale; with a
    visualizer whose ``pumps_app_update()`` is True (e.g. KitVisualizer) the pump must be skipped
    so the Kit loop is not driven twice.
    """
    sim = SimulationContext(SimulationCfg(dt=0.01))
    sim.reset()
    sim.set_setting("/isaaclab/video/enabled", True)
    sim.set_setting("/isaaclab/render/rtx_sensors", True)

    if visualizer_pumps:
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
        patch("isaaclab_physx.renderers.isaac_rtx_renderer_utils._get_stage_streaming_busy", return_value=False),
        patch("omni.kit.app.get_app", return_value=mock_app),
    ):
        sim.render()

    assert mock_app.update.call_count == (0 if visualizer_pumps else 1)
    sim._visualizers = []


def test_render_callbacks():
    """Render callbacks fire in ``order`` on every render; same-name registration replaces, removal is a no-op-safe."""
    sim = SimulationContext(SimulationCfg(dt=0.01))
    sim.reset()

    call_log: list[str] = []
    replaced = MagicMock()
    removed = MagicMock()
    sim.add_render_callback("second", lambda _: call_log.append("second"), order=10)
    sim.add_render_callback("first", lambda _: call_log.append("first"), order=0)
    sim.add_render_callback("third", lambda _: call_log.append("third"), order=20)
    sim.add_render_callback("replaced", replaced)
    sim.add_render_callback("replaced", lambda arg: call_log.append(f"replacement:{arg}"))
    sim.add_render_callback("removed", removed)
    sim.remove_render_callback("removed")
    sim.remove_render_callback("nonexistent")

    sim.render()
    sim.render()

    assert call_log == ["first", "replacement:None", "second", "third"] * 2
    replaced.assert_not_called()
    removed.assert_not_called()


"""
Physics-backend events
"""


@pytest.mark.isaacsim_ci
def test_isaac_events():
    """Reset dispatches the warmup/view/ready events to every subscriber; explicit dispatches carry their payload."""
    sim = SimulationContext(SimulationCfg(dt=0.01))
    _spawn_cube()

    received: dict[str, list] = {}

    def subscribe(name: str, event: IsaacEvents) -> int:
        received[name] = []
        return PhysxManager.register_callback(lambda payload, name=name: received[name].append(payload), event=event)

    callback_ids = [
        subscribe("warmup", IsaacEvents.PHYSICS_WARMUP),
        subscribe("view_created", IsaacEvents.SIMULATION_VIEW_CREATED),
        subscribe("ready_a", IsaacEvents.PHYSICS_READY),
        subscribe("ready_b", IsaacEvents.PHYSICS_READY),
        subscribe("prim_deletion", IsaacEvents.PRIM_DELETION),
        subscribe("timeline_stop", IsaacEvents.TIMELINE_STOP),
    ]
    try:
        sim.reset()
        for name in ("warmup", "view_created", "ready_a", "ready_b"):
            assert received[name], f"{name} was not dispatched during reset"

        sim_utils.delete_prim("/World/Cube")
        PhysxManager._message_bus.dispatch_event(IsaacEvents.PRIM_DELETION.value, payload={"prim_path": "/World/Cube"})
        assert received["prim_deletion"][-1].payload.get("prim_path") == "/World/Cube"

        num_stops = len(received["timeline_stop"])
        _stop(sim)
        assert len(received["timeline_stop"]) == num_stops + 1
    finally:
        for callback_id in callback_ids:
            PhysxManager.deregister_callback(callback_id)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    ("event", "trigger"),
    [(IsaacEvents.PHYSICS_READY, "reset"), (IsaacEvents.POST_PHYSICS_STEP, "step")],
    ids=["reset", "step"],
)
def test_exception_stored_in_callback_is_raised(event, trigger):
    """Exceptions stored by callbacks surface from the reset or step that ran them."""
    sim = SimulationContext(SimulationCfg(dt=0.01))
    _spawn_cube()
    if trigger == "step":
        sim.reset()

    message = f"Test exception on {trigger}"
    handle = PhysxManager.register_callback(
        lambda _event: PhysxManager.store_callback_exception(RuntimeError(message)), event=event
    )
    try:
        with pytest.raises(RuntimeError, match=message):
            getattr(sim, trigger)()
    finally:
        PhysxManager.deregister_callback(handle)
