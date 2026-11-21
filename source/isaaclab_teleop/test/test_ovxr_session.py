# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVXR bootstrap retirement across the teleoperation/renderer boundary."""

import builtins
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ov

from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.sim import SimulationCfg

pytest.importorskip("ovrtx")
from isaaclab_teleop.ovxr_renderer import OvxrRenderer, OvxrRendererCfg


@pytest.mark.parametrize("operation", ["close", "abort"])
def test_bootstrap_failed_transfer_cancellation_keeps_owner_retryable(bootstrap, operation):
    """A wrong-thread close cannot destroy a pending lease or prevent its owner retiring it."""
    import threading

    from isaaclab_teleop import ovxr_session

    bootstrap.offer_render_owner_transfer()
    context = ovxr_session.SimulationContext.instance()
    errors = []

    def wrong_owner():
        try:
            getattr(bootstrap, operation)()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=wrong_owner)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert len(errors) == 1 and "owner thread must cancel" in errors[0]
    assert not bootstrap.closed
    assert ovxr_session.SimulationContext.instance() is context
    # The real owner must still be able to retire the real renderer lease.
    getattr(bootstrap, operation)()
    assert bootstrap.closed
    assert ovxr_session.SimulationContext.instance() is (None if operation == "abort" else context)


@pytest.fixture
def bootstrap(monkeypatch):
    """Run the real bootstrap constructor and renderer lease; only the physics-backed context is replaced."""
    from isaaclab_teleop import ovxr_session

    from isaaclab.app import get_settings_manager
    from isaaclab.renderers import RenderContext

    monkeypatch.delattr(builtins, ovxr_session._TERMINAL_MODE_SENTINEL, raising=False)
    renderer = OvxrRenderer(
        OvxrRendererCfg(),
        sim=SimpleNamespace(
            device="cpu", get_scene_data_provider=lambda: None, vis_marker_registry=VisMarkerRegistry()
        ),
    )

    class Context:
        current = None

        def __init__(self, cfg):
            self.cfg = cfg
            self.render_context = RenderContext([(renderer.cfg, renderer)])
            type(self).current = self

        @classmethod
        def instance(cls):
            return cls.current

        @classmethod
        def clear_instance(cls):
            cls.current = None

        def get_or_create_backend(self, _cfg):
            return renderer

    monkeypatch.setattr(ovxr_session, "SimulationContext", Context)
    # SimulationContext resolves a bare "cuda" device from this setting.
    settings = get_settings_manager()
    previous_cuda_device = settings.get("/physics/cudaDevice")
    settings.set("/physics/cudaDevice", 1)
    try:
        bootstrap = ovxr_session.OvxrSessionBootstrap(SimulationCfg(device="cuda:1", dt=0.01))
        yield bootstrap
        bootstrap.abort()
        assert renderer._render_lease.active is None
    finally:
        settings.set("/physics/cudaDevice", previous_cuda_device)


def _environment(bootstrap, *, num_envs=1, device="cuda:1", dt=0.01):
    return SimpleNamespace(
        sim=bootstrap.simulation_context,
        scene=SimpleNamespace(stage=object()),
        num_envs=num_envs,
        cfg=SimpleNamespace(sim=SimulationCfg(device=device, dt=dt)),
    )


@pytest.mark.parametrize(
    ("num_envs", "device", "dt", "error"),
    [
        (1, "cuda", 0.01, None),
        (2, "cuda:1", 0.01, "exactly one environment"),
        (1, "cuda:0", 0.01, r"sim\.device='cuda:0'"),
        (1, "cuda:1", 0.02, r"sim\.dt=0\.02"),
    ],
)
def test_environment_validation_resolves_devices_and_requires_one_environment(bootstrap, num_envs, device, dt, error):
    """The caller-side check resolves a bare CUDA device like SimulationContext and rejects late worker failures."""
    env = _environment(bootstrap, num_envs=num_envs, device=device, dt=dt)
    if error is None:
        bootstrap.validate_environment(env)
        env.cfg.sim = bootstrap.simulation_context.cfg
        original_dt = env.cfg.sim.dt
        env.cfg.sim.dt = original_dt * 2
        try:
            with pytest.raises(RuntimeError, match=r"sim\.dt"):
                bootstrap.validate_environment(env)
        finally:
            env.cfg.sim.dt = original_dt
    else:
        with pytest.raises(RuntimeError, match=error):
            bootstrap.validate_environment(env)


def test_binding_requires_the_claimed_owner_transfer(bootstrap):
    """Binding before the claim names the missing claim, even when no camera deferred the Stage yet."""
    with pytest.raises(RuntimeError, match="requires a claimed owner transfer"):
        bootstrap.bind_environment(_environment(bootstrap))


def test_bootstrap_rejects_an_ordinary_renderer_configuration(monkeypatch):
    """An ordinary OVRTX configuration selects Isaac Lab's renderer, which the session cannot borrow."""
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_teleop import ovxr_session

    monkeypatch.delattr(builtins, ovxr_session._TERMINAL_MODE_SENTINEL, raising=False)
    monkeypatch.setattr(ovxr_session, "SimulationContext", _UnusedContext)
    with pytest.raises(TypeError, match="OvxrRendererCfg"):
        ovxr_session.OvxrSessionBootstrap(SimulationCfg(device="cpu", dt=0.01), OVRTXRendererCfg())
    assert _UnusedContext.current is None


def test_camera_panels_must_use_the_session_renderer_configuration(bootstrap):
    """A panel with another configuration would get a second renderer that the session never borrows."""
    from isaaclab_ov.renderers import OVRTXRendererCfg

    for renderer_cfg in (OVRTXRendererCfg(), OvxrRendererCfg(log_level="warn")):
        camera = SimpleNamespace(renderer_cfg=renderer_cfg, prim_path="/World/Camera")
        with pytest.raises(ValueError, match="session's renderer configuration"):
            bootstrap.configure_auxiliary_camera(camera)


class _UnusedContext:
    current = None

    def __init__(self, _cfg):
        pytest.fail("created a simulation context")

    @classmethod
    def instance(cls):
        return cls.current

    @classmethod
    def clear_instance(cls):
        cls.current = None

    def get_or_create_backend(self, _cfg):
        pytest.fail("created a renderer")


class _CaptureOwner:
    """Isaac Capture's session owner: closing it drains the worker, which releases the lease on the render owner."""

    def __init__(self, bootstrap, events):
        self.bootstrap = bootstrap
        self.events = events
        self.failure = None
        self.fails_after_retirement = False
        self.closes = 0
        self.retirement_complete = False
        # The worker attached, so a join that times out is a retirement that did not finish.
        self.render_worker_bootstrap_ready = True

    def external_oxr_handles(self):
        return "handles"

    def close(self):
        self.closes += 1
        self.events.append("xr-drain")
        if self.failure is not None and not self.fails_after_retirement:
            raise self.failure
        self.bootstrap.close()
        self.retirement_complete = True
        if self.failure is not None:
            # Isaac Capture raises a latched worker error although its worker retired everything.
            raise self.failure


@pytest.fixture
def ovxr_simulation(monkeypatch):
    """Prepare a real kitless simulation and OVXR session whose leased pair is bound; native endpoints are doubled."""
    import contextlib
    import sys
    from unittest.mock import MagicMock

    pytest.importorskip("isaaccapture.ovxr_session_manager")
    pytest.importorskip("ovphysx")
    import isaaclab_ov.stage as stage_module
    import ovstage
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_teleop import ovxr_scene, ovxr_session
    from isaaclab_teleop.ov_libraries_xr import OvLibrariesXrCfg, OvLibrariesXrSession

    from isaaclab.physics import PhysicsEvent
    from isaaclab.sim import SimulationContext

    monkeypatch.delattr(builtins, ovxr_session._TERMINAL_MODE_SENTINEL, raising=False)
    monkeypatch.setattr(ovxr_session, "_RETAINED_NATIVE_PAIRS", [])
    events = []
    native, stage, paths = MagicMock(), MagicMock(), MagicMock()
    native.destroy.side_effect = lambda: events.append("native-destroy")
    floor = stage.get_attribute_write_floor.return_value.__enter__.return_value
    floor.fetch.return_value = ovstage.ORDINAL_MIN  # A new OVStage is sealed through the minimum ordinal.
    stage.advance_write_floor.side_effect = lambda *, ordinal: SimpleNamespace(
        wait=lambda: setattr(floor.fetch, "return_value", ordinal)
    )

    @contextlib.contextmanager
    def create_ovstage(_name):
        yield stage
        events.append("stage-close")

    monkeypatch.setattr(ovxr_scene, "Renderer", lambda _cfg: native)
    monkeypatch.setattr(ovxr_scene, "redirect_shader_cache", lambda _cfg: None)
    monkeypatch.setattr(stage_module, "create_ovstage", create_ovstage)
    monkeypatch.setattr(stage_module.ovstage, "PathDictionary", lambda _: contextlib.nullcontext(paths))
    monkeypatch.setattr(stage_module.ovstage.population, "open_usd_from_string", lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "ovxr", SimpleNamespace(apply_renderer_settings=lambda _settings: None))

    session = OvLibrariesXrSession.prepare(
        SimulationCfg(physics=OvPhysxCfg(), device="cpu", dt=0.01), OvLibrariesXrCfg()
    )
    bootstrap = session._bootstrap
    sim = bootstrap.simulation_context
    try:
        renderer = sim.get_or_create_backend(OvxrRendererCfg())
        renderer._sdp = SimpleNamespace(backend=SimpleNamespace(transform_paths=[]), get_geometry_points=lambda: {})
        # The session created its own pair; this thread claims and binds it as the render owner.
        lease = bootstrap.claim_render_owner_transfer(bootstrap.offer_render_owner_transfer())
        renderer.prepare_stage(sim.stage, 1)
        lease.validate_live()
        # Camera sensors invalidate their render data at this STOP priority.
        sim.backend_manager.register_callback(
            lambda _payload: events.append("camera-cleanup"), PhysicsEvent.STOP, order=10
        )
        close_render_context = type(sim.render_context).close

        def record_render_context_close(context):
            events.append("render-context-close")
            close_render_context(context)

        monkeypatch.setattr(type(sim.render_context), "close", record_render_context_close)
        owner = _CaptureOwner(bootstrap, events)
        session._owner = owner
        yield SimpleNamespace(
            session=session, bootstrap=bootstrap, sim=sim, owner=owner, native=native, stage=stage, events=events
        )
    finally:
        if SimulationContext.instance() is not None:
            with contextlib.suppress(Exception):
                SimulationContext.clear_instance()
        if hasattr(builtins, ovxr_session._TERMINAL_MODE_SENTINEL):
            delattr(builtins, ovxr_session._TERMINAL_MODE_SENTINEL)


def test_simulation_stop_retires_ovxr_before_cameras_renderers_and_native_backends(ovxr_simulation):
    """Clearing the context drains the XR worker before any Isaac Lab STOP listener or native backend closes."""
    from isaaclab.sim import SimulationContext

    state = ovxr_simulation
    SimulationContext.clear_instance()
    events = state.events
    assert events[0] == "xr-drain"
    order = [events.index(name) for name in ("camera-cleanup", "render-context-close", "native-destroy", "stage-close")]
    assert order == sorted(order)
    assert state.session.retirement_complete
    assert state.bootstrap.closed
    state.native.destroy.assert_called_once_with()


def test_simulation_stop_retirement_runs_once(ovxr_simulation):
    """Repeated STOP events, as a later teardown dispatches, do not repeat the drain."""
    from isaaclab.physics import PhysicsEvent
    from isaaclab.sim import SimulationContext

    state = ovxr_simulation
    state.sim.backend_manager.dispatch_event(PhysicsEvent.STOP)
    state.sim.backend_manager.dispatch_event(PhysicsEvent.STOP)
    SimulationContext.clear_instance()
    assert state.owner.closes == 1
    assert state.events.count("xr-drain") == 1
    state.native.destroy.assert_called_once_with()


def test_simulation_stop_while_running_leaves_the_session_open_and_its_pair_usable(ovxr_simulation):
    """A hard reset with the session open is refused without quarantining the pair or consuming the retirement."""
    from isaaclab.physics import PhysicsEvent
    from isaaclab.sim import SimulationContext

    state = ovxr_simulation
    # A hard Newton reset or an OvPhysX re-warm dispatches STOP while the simulation runs.
    state.sim.play()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="while running"):
            state.sim.backend_manager.dispatch_event(PhysicsEvent.STOP)
    assert state.owner.closes == 0
    assert state.bootstrap._retained_pair == ()
    assert "camera-cleanup" not in state.events
    state.bootstrap._lease.validate_live()
    # The later shutdown still retires the session and destroys the pair normally.
    SimulationContext.clear_instance()
    assert state.owner.closes == 1
    assert state.session.retirement_complete and state.bootstrap.closed
    state.native.destroy.assert_called_once_with()
    assert "stage-close" in state.events


def test_simulation_stop_is_clean_after_a_reported_worker_failure_with_complete_retirement(ovxr_simulation):
    """The worker failure reaches the host once; the teardown that follows a complete retirement reports nothing."""
    from isaaclab.sim import SimulationContext

    state = ovxr_simulation
    failure = state.owner.failure = RuntimeError("OVXR begin_frame failed")
    state.owner.fails_after_retirement = True
    with pytest.raises(RuntimeError) as caught:
        state.session.close()
    assert caught.value is failure
    assert state.session.retirement_complete and state.bootstrap.closed
    state.session.close()
    SimulationContext.clear_instance()
    assert state.owner.closes == 1
    assert state.bootstrap._retained_pair == ()
    state.native.destroy.assert_called_once_with()
    assert "stage-close" in state.events


@pytest.mark.parametrize("refusal", ["worker", "input"])
def test_unproved_simulation_stop_retirement_keeps_the_native_pair(ovxr_simulation, refusal, caplog):
    """A drain that fails or cannot run keeps the leased engine and OVStage alive through teardown."""
    import logging

    from isaaclab_teleop import ovxr_session

    from isaaclab.sim import SimulationContext

    state = ovxr_simulation
    failure = None
    if refusal == "worker":
        failure = state.owner.failure = TimeoutError("OVXR session worker did not stop")
    else:
        # XR input borrowed the session's OpenXR handles and the host never confirmed they stopped.
        assert state.session.external_oxr_handles() == "handles"
    with caplog.at_level(logging.ERROR, logger=ovxr_session.logger.name), pytest.raises(RuntimeError) as caught:
        SimulationContext.clear_instance()
    assert SimulationContext.instance() is None
    if refusal == "worker":
        assert state.owner.closes == 1
        assert caught.value.__cause__.__cause__ is failure
    else:
        assert state.owner.closes == 0
    if refusal == "input":
        assert "input may still borrow" in str(caught.value.__cause__.__cause__)
    assert not state.bootstrap.closed and not state.session.retirement_complete
    state.native.destroy.assert_not_called()
    state.native.detach_ovstage.assert_not_called()
    assert "stage-close" not in state.events
    assert state.native in state.bootstrap._retained_pair and state.stage in state.bootstrap._retained_pair
    assert sum("keeping the leased OVRTX renderer and OVStage alive" in r.getMessage() for r in caplog.records) == 1
    # The process also holds the pair, so it survives the session and its bootstrap being dropped.
    assert any(pair is state.bootstrap._retained_pair for pair in ovxr_session._RETAINED_NATIVE_PAIRS)
