# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-process teleoperation entrypoint, configuration, and control-loop contracts."""

import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Repository root: this file lives at source/isaaclab_teleop/test/<file>.py
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def run_live_workflow(monkeypatch):
    """Run the imported live owner with finite input, renderer, and simulation endpoints."""
    import contextlib

    # These adapter tests require the optional Capture OVXR session module.
    pytest.importorskip("isaaccapture.ovxr_session_manager")
    import isaaclab_teleop
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    def run(env, interface, session, *, step_hz=0.0, refresh=lambda: None, started=True):
        class Device:
            session_cleanup_failed = False
            target_frame_prim_path = None
            xr_render_anchor_path = None
            required_openxr_extensions = ()
            external_anchor_state = None

            def __enter__(self):
                getattr(interface, "enter", lambda: None)()
                return self

            def __exit__(self, *_args):
                return False

            def advance(self):
                return interface.advance()

            def reset(self):
                return interface.reset()

            def notify_reset(self):
                return interface.notify_reset()

            @property
            def last_control_events(self):
                return interface.control_events()

        camera = SimpleNamespace(bind=lambda _env: contextlib.nullcontext(), refresh=refresh, close=lambda: None)
        session.external_oxr_handles = None
        vars(session).setdefault("start_render_worker", lambda *_args, **_kwargs: started)
        vars(session).setdefault("close", lambda: None)
        vars(env).setdefault("close", lambda: None)
        env.unwrapped = env
        env.scene = SimpleNamespace(env_prim_paths=["/World/envs/env_0"])
        env.sim.device = "cpu"
        env.sim.visualizers = ()
        cfg = SimpleNamespace(
            sim=object(),
            isaac_teleop=isaaclab_teleop.IsaacTeleopCfg(),
        )
        monkeypatch.setattr(
            agent,
            "args_cli",
            agent._build_parser().parse_args(["--task", "Test", "--ovxr", "--ovxr_step_hz", str(step_hz)]),
            raising=False,
        )
        monkeypatch.setattr(agent, "has_kit", lambda: False)
        monkeypatch.setattr(agent.gym, "make", lambda *_args, **_kwargs: SimpleNamespace(unwrapped=env))
        monkeypatch.setattr(agent, "create_teleop_device", lambda *_args, **_kwargs: Device())
        monkeypatch.setattr(isaaclab_teleop, "create_haptic_feedback_driver", lambda *_args: None)
        monkeypatch.setattr(isaaclab_teleop.XrCameraFeedSession, "prepare", lambda *_args, **_kwargs: camera)
        monkeypatch.setattr(isaaclab_teleop.OvLibrariesXrSession, "prepare", lambda *_args, **_kwargs: session)
        agent.run_teleoperation(cfg, cfg.isaac_teleop, use_isaac_teleop=True)

    return run


def test_teleop_se3_agent_yields_only_when_ovxr_is_idle(monkeypatch):
    """A paused kitless loop must yield without slowing Kit or active physics frames."""
    from isaaclab_teleop.entrypoints import _ovxr_runtime

    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(time, "perf_counter", lambda: 10.003)
    yield_idle = _ovxr_runtime.pace_ovxr_control_loop

    timing = {"iteration_started_s": 10.0, "action_period_s": 0.02}
    yield_idle(using_ovxr=False, action_applied=False, **timing)
    yield_idle(using_ovxr=True, action_applied=True, **timing)
    assert sleeps == []

    yield_idle(using_ovxr=True, action_applied=False, **timing)
    assert sleeps == pytest.approx([0.017])
    yield_idle(using_ovxr=True, action_applied=False, iteration_started_s=9.9, action_period_s=0.02)
    assert sleeps == pytest.approx([0.017])  # Slow input work needs no added delay.


@pytest.mark.parametrize(
    "active,step_hz,expected_polls",
    [(False, 0.0, 6), (True, 25.0, 3), (None, 25.0, 3)],
)
def test_kitless_loop_respects_control_rate_and_processes_reset(
    monkeypatch, run_live_workflow, active, step_hz, expected_polls
):
    """The real caller preserves idle pacing and applies an explicit active limit."""
    from types import SimpleNamespace

    import torch
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    clock = 0.0
    advances = 0
    resets = []
    reset_notifications = []
    refreshes = []
    publications = []
    applied = []
    sequence = active is None
    anchor_ready = False

    def reset():
        nonlocal anchor_ready
        resets.append(clock)
        anchor_ready = False

    def request(identity, *, force=False):
        nonlocal anchor_ready
        publications.append((identity, {"force": force}))
        anchor_ready = True
        return 1

    def sleep(duration):
        nonlocal clock
        clock += duration

    def advance():
        nonlocal advances, clock
        assert anchor_ready, "Input cannot read an unseeded post-reset anchor"
        advances += 1
        clock += 0.003  # Caller work consumes part of the independent 20ms action budget.
        return torch.ones((1, 36)) if active or (sequence and advances != 4) else None

    def step(_value):
        applied.append(clock)
        return None, None, torch.tensor([False]), torch.tensor([False]), {}

    sim = SimpleNamespace(
        is_running=lambda: clock < (0.199 if sequence else 0.119),
        get_physics_step_count=lambda: 0,
        get_physics_dt=lambda: 0.005,
        render=lambda: None,
    )
    monkeypatch.setattr(agent, "time", SimpleNamespace(perf_counter=lambda: clock, sleep=sleep))
    run_live_workflow(
        SimpleNamespace(sim=sim, reset=reset, step_dt=0.02, num_envs=1, step=step),
        SimpleNamespace(
            advance=advance,
            reset=lambda: None,
            notify_reset=lambda: reset_notifications.append(len(resets)),
            control_events=lambda: SimpleNamespace(
                is_active=advances not in (2, 4) if sequence else active,
                should_reset=advances == (5 if sequence else 3),
            ),
        ),
        SimpleNamespace(
            shutdown_requested=False,
            wait_for_shutdown=sleep,
            request_render=request,
        ),
        step_hz=step_hz,
        refresh=lambda: refreshes.append(clock),
    )

    assert advances == (5 if sequence else expected_polls)
    assert clock == pytest.approx(
        0.20 if sequence else 0.12
    )  # Work time is credited, rather than adding a full-period sleep.
    assert len(applied) == (3 if sequence else advances if active else 0)
    assert len(resets) == 2  # Initial reset plus the public control event.
    assert reset_notifications == [2]  # Notify Capture after the environment reset.
    assert len(refreshes) == 1
    assert sum(kwargs["force"] for _identity, kwargs in publications) == 2
    assert any(kwargs["force"] for _identity, kwargs in publications)


def test_paused_loop_retries_the_latest_unaccepted_publication(monkeypatch, run_live_workflow):
    """A paused iteration retries the rejected latest state without another physics step."""
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    advances = 0
    steps = []
    requests = []
    sleeps = []
    session = SimpleNamespace(shutdown_requested=False)

    def advance():
        nonlocal advances
        advances += 1
        return torch.ones(1)

    def request(identity, *, force):
        requests.append((identity, force))
        if force:
            return 1
        if len(requests) == 2:
            return None
        session.shutdown_requested = True
        return 2

    def sleep(duration):
        sleeps.append((duration, len(requests)))

    session.request_render = request
    session.wait_for_shutdown = sleep
    monkeypatch.setattr(agent, "time", SimpleNamespace(perf_counter=lambda: 0.0, sleep=sleep))
    env = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        reset=lambda: None,
        step=lambda action: (steps.append(action), None, torch.tensor([False]), torch.tensor([False]), {}),
        sim=SimpleNamespace(
            is_running=lambda: advances < 4, get_physics_step_count=lambda: len(steps), get_physics_dt=lambda: 0.005
        ),
    )
    run_live_workflow(
        env,
        SimpleNamespace(
            advance=advance,
            reset=lambda: None,
            control_events=lambda: SimpleNamespace(is_active=advances == 1, should_reset=False),
        ),
        session,
    )
    assert advances == 2 and len(steps) == 1
    assert len(requests) == 3 and requests[0][1] is True
    assert requests[1:] == [((1, 1), False), ((1, 1), False)]
    assert sleeps == [(0.02, 3)]  # Pacing follows the retried publication.


@pytest.mark.parametrize("failure_at", ["startup", "publication"])
def test_live_render_worker_failure_drains_without_hiding_loop_errors(run_live_workflow, failure_at):
    """Startup shutdown skips input; an in-loop publication error escapes after orderly cleanup."""
    events = []
    failure = RuntimeError("publication failed")
    publications = 0

    def publish(*_args, **_kwargs):
        nonlocal publications
        publications += 1
        if publications > 1:
            raise failure
        return publications

    env = SimpleNamespace(
        sim=SimpleNamespace(is_running=lambda: True, get_physics_step_count=lambda: 0, get_physics_dt=lambda: 0.005),
        step_dt=0.02,
        reset=lambda: events.append("env-reset"),
        close=lambda: events.append("env-close"),
    )
    interface = SimpleNamespace(
        advance=lambda: events.append("advance"),
        reset=lambda: None,
        control_events=lambda: SimpleNamespace(is_active=False, should_reset=False),
    )
    session = SimpleNamespace(
        shutdown_requested=False,
        close=lambda: events.append("xr-close"),
        request_render=publish,
        report_host_timing=lambda *_args: None,
    )
    if failure_at == "startup":
        run_live_workflow(env, interface, session, started=False)
        assert events == ["xr-close", "env-close"]
    else:
        with pytest.raises(RuntimeError) as caught:
            run_live_workflow(env, interface, session)
        assert caught.value is failure
        assert publications == 2
        assert events == ["env-reset", "advance", "xr-close", "env-close"]


def test_live_prestart_signal_cancels_input_and_preserves_cleanup(run_live_workflow):
    """Cancellation during input entry prevents worker start; a second signal cannot interrupt drain."""
    from unittest.mock import Mock

    events = []

    def close():
        events.append("xr-close")
        signal.raise_signal(signal.SIGINT)

    env = SimpleNamespace(sim=SimpleNamespace(), close=lambda: events.append("env-close"))
    session = SimpleNamespace(
        shutdown_requested=False,
        start_render_worker=Mock(side_effect=AssertionError("worker started after cancelled input")),
        request_shutdown=lambda: events.append("shutdown"),
        close=close,
    )
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(KeyboardInterrupt):
        run_live_workflow(env, SimpleNamespace(enter=lambda: signal.raise_signal(signal.SIGTERM)), session)
    session.start_render_worker.assert_not_called()
    assert events == ["xr-close", "shutdown", "env-close"]
    assert {sig: signal.getsignal(sig) for sig in original} == original


@pytest.fixture
def run_record_workflow(monkeypatch):
    """Run the imported recorder with ``--ovxr`` and finite input, renderer, and simulation endpoints."""
    import contextlib

    pytest.importorskip("isaaccapture.ovxr_session_manager")
    import isaaclab_teleop
    from isaaclab_teleop.entrypoints import record_demos

    def run(env, session, events, *arguments, make=None, create_device=None):
        camera = SimpleNamespace(
            bind=lambda _env: contextlib.nullcontext(),
            refresh=lambda: None,
            close=lambda: events.append("camera-close"),
        )

        def prepare(*_args, **_kwargs):
            events.append("xr-prepare")
            return session

        def make_env(*_args, **_kwargs):
            events.append("make")
            return SimpleNamespace(unwrapped=env)

        session.external_oxr_handles = lambda: None
        vars(session).setdefault("close", lambda: events.append("xr-close"))
        env.unwrapped = env
        monkeypatch.setattr(
            record_demos,
            "args_cli",
            record_demos._build_parser().parse_args(["--task", "Test", "--ovxr", *arguments]),
            raising=False,
        )
        monkeypatch.setattr(record_demos, "has_kit", lambda: False)
        monkeypatch.setattr(record_demos, "time", SimpleNamespace(perf_counter=lambda: 0.0, time=time.time))
        monkeypatch.setattr(record_demos.gym, "make", make or make_env)
        monkeypatch.setattr(record_demos, "create_teleop_device", create_device)
        monkeypatch.setattr(isaaclab_teleop, "create_haptic_feedback_driver", lambda *_args: None)
        monkeypatch.setattr(isaaclab_teleop.XrCameraFeedSession, "prepare", lambda *_args, **_kwargs: camera)
        monkeypatch.setattr(isaaclab_teleop.OvLibrariesXrSession, "prepare", prepare)
        cfg = SimpleNamespace(sim=object())
        with contextlib.ExitStack() as cleanup:
            record_demos.record_demos(cfg, None, isaaclab_teleop.IsaacTeleopCfg(), cleanup)

    return run


def test_record_ovxr_loop_follows_live_presentation_and_shutdown_order(run_record_workflow):
    """Recording prepares OVXR first, fences input on reset publication, never renders, and drains XR first."""
    from isaaclab_teleop.entrypoints._ovxr_runtime import RENDERER_ATTACH_TIMEOUT_S

    events = []
    frames = []
    steps = []
    reset_attempts = 0
    device_options = {}

    class Device:
        session_cleanup_failed = False
        target_frame_prim_path = "/World/Robot"
        xr_render_anchor_path = "/World/XRAnchor"
        external_anchor_state = None
        required_openxr_extensions = ("XR_ext",)

        def __enter__(self):
            events.append("device-enter")
            return self

        def __exit__(self, *_args):
            events.append("device-exit")
            return False

        def reset(self):
            events.append("device-reset")

        def notify_reset(self):
            events.append("device-notify-reset")

        def advance(self, *, target_T_world):
            frames.append(target_T_world)
            events.append("advance")
            return None if len(frames) == 3 else torch.ones(1, 2)

        @property
        def last_control_events(self):
            return SimpleNamespace(is_active=True, should_reset=len(frames) == 2)

    def create_device(_cfg, **kwargs):
        events.append("device")
        device_options.update(kwargs)
        return Device()

    def request(identity, *, force=False):
        nonlocal reset_attempts
        events.append(("publish", identity, force))
        if force and identity[0] == 2:
            # Decline the operator reset epoch twice: once after the reset, once at the input fence.
            reset_attempts += 1
            if reset_attempts < 3:
                return None
        return 1

    def step(actions):
        steps.append(actions)
        events.append("step")
        done = torch.tensor([len(steps) == 3])
        return {}, None, done, torch.tensor([False]), {}

    session = SimpleNamespace(
        shutdown_requested=False,
        start_render_worker=lambda _env, **kwargs: events.append(("worker-start", kwargs["timeout_s"])) or True,
        request_render=request,
        get_target_frame_T_world=lambda path: (path, len(steps)),
        wait_for_shutdown=lambda seconds: events.append(("wait", seconds)) or False,
    )
    fail = pytest.fail
    env = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        cfg=SimpleNamespace(),
        scene=SimpleNamespace(env_prim_paths=["/World/envs/env_0"], physics_backend="ovphysx"),
        sim=SimpleNamespace(
            visualizers=(),
            is_running=lambda: len(frames) < 4,
            is_stopped=lambda: False,
            get_physics_step_count=lambda: len(steps),
            render=lambda: fail("OVXR owns rendering; the control thread must not render"),
            reset=lambda: fail("A stage leased to OVXR must not be rebuilt"),
        ),
        recorder_manager=SimpleNamespace(
            reset=lambda: events.append("recorder-reset"), exported_successful_episode_count=0
        ),
        reset=lambda: events.append("env-reset"),
        step=step,
        close=lambda: events.append("env-close"),
    )
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    run_record_workflow(env, session, events, "--mcap_record_path", "session.mcap", create_device=create_device)

    assert events == [
        "xr-prepare",
        "make",
        "device",
        "device-enter",
        ("worker-start", RENDERER_ATTACH_TIMEOUT_S),
        "env-reset",
        "device-reset",
        ("publish", (1, 0), True),
        "advance",
        "step",
        ("publish", (1, 1), False),
        "advance",
        "step",
        "recorder-reset",
        "env-reset",
        "device-reset",
        "device-notify-reset",
        ("publish", (2, 2), True),
        ("publish", (2, 2), True),
        ("wait", 0.02),
        ("publish", (2, 2), True),
        "advance",
        ("publish", (2, 2), False),
        ("wait", 0.02),
        "advance",
        "step",
        "device-notify-reset",
        ("publish", (3, 3), True),
        "device-exit",
        "xr-close",
        "camera-close",
        "env-close",
    ]
    assert frames == [("/World/Robot", 0), ("/World/Robot", 1), ("/World/Robot", 2), ("/World/Robot", 2)]
    assert device_options["cloudxr_env_file"] is None and device_options["auto_launch_cloudxr"] is False
    assert device_options["use_kit_xr_bridge"] is False
    assert device_options["external_oxr_handles_provider"] is session.external_oxr_handles
    assert device_options["mcap_record_path"] == "session.mcap"
    assert {sig: signal.getsignal(sig) for sig in original} == original


@pytest.mark.parametrize("failure_at", ["environment", "device", "signal"])
def test_record_ovxr_startup_failure_drains_xr_before_releasing_the_environment(
    run_record_workflow, caplog, failure_at
):
    """Startup failures propagate after cleanup; a pre-start signal cancels the render worker."""
    from unittest.mock import Mock

    events = []
    failure = RuntimeError(f"{failure_at} failed")

    class Device:
        session_cleanup_failed = False

        def __enter__(self):
            events.append("device-enter")
            signal.raise_signal(signal.SIGTERM)

        def __exit__(self, *_args):
            events.append("device-exit")
            return False

    env = SimpleNamespace(
        scene=SimpleNamespace(env_prim_paths=["/World/envs/env_0"]),
        sim=SimpleNamespace(visualizers=()),
        cfg=SimpleNamespace(),
        close=lambda: events.append("env-close"),
    )
    session = SimpleNamespace(
        start_render_worker=Mock(side_effect=AssertionError("worker started after cancelled input")),
        request_shutdown=lambda: events.append("shutdown"),
    )
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    expected = KeyboardInterrupt if failure_at == "signal" else RuntimeError
    with pytest.raises(expected) as caught:
        run_record_workflow(
            env,
            session,
            events,
            make=Mock(side_effect=failure) if failure_at == "environment" else None,
            create_device=Mock(side_effect=failure) if failure_at == "device" else Mock(return_value=Device()),
        )
    if failure_at == "signal":
        session.start_render_worker.assert_not_called()
        assert events == ["xr-prepare", "make", "device-enter", "xr-close", "camera-close", "env-close"]
    else:
        assert caught.value is failure
        assert f"Failed to create {'environment' if failure_at == 'environment' else 'teleop device'}" in caplog.text
        if failure_at == "environment":
            assert events == ["xr-prepare", "xr-close", "camera-close"]
        else:
            assert events == ["xr-prepare", "make", "xr-close", "camera-close", "env-close"]
    assert {sig: signal.getsignal(sig) for sig in original} == original


@pytest.mark.parametrize("selector", [["--physics", "ovphysx"], ["physics=ovphysx"]])
@pytest.mark.parametrize("ovxr", [False, True])
def test_teleop_replay_agent_resolves_typed_presets_without_reapplying_physics(monkeypatch, selector, ovxr):
    """Public CLI aliases reach task resolution without resetting backend settings."""
    import isaaclab_teleop
    from isaaclab_ov.physics import OvPhysxCfg

    from isaaclab.managers import TerminationTermCfg
    from isaaclab.sim import SimulationCfg

    from isaaclab_tasks.utils import parse_cfg

    source_path = ROOT / "source/isaaclab_teleop/isaaclab_teleop/entrypoints/teleop_replay_agent.py"
    argv = [str(source_path), "--task", "Test", "--replay_file", "recording.mcap", *selector]
    if ovxr:
        argv.append("--ovxr")
    monkeypatch.setattr(sys, "argv", argv)
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay

    monkeypatch.setattr(replay, "args_cli", None, raising=False)
    monkeypatch.setattr(replay, "hydra_overrides", None, raising=False)
    namespace = vars(replay)
    replay._parse_args()
    physics = OvPhysxCfg()
    cfg = SimpleNamespace(
        sim=SimulationCfg(physics=physics),
        scene=SimpleNamespace(num_envs=1),
        isaac_teleop=isaaclab_teleop.IsaacTeleopCfg(),
        terminations=SimpleNamespace(
            success=TerminationTermCfg(func=lambda env: False),
            time_out=TerminationTermCfg(func=lambda env: False),
        ),
    )
    physics = cfg.sim.physics
    resolutions = []

    def resolve(task, agent, *, overrides):
        resolutions.append((task, agent, overrides))
        return cfg, None

    monkeypatch.setattr(replay, "ManagerBasedRLEnvCfg", SimpleNamespace)
    monkeypatch.setattr(parse_cfg, "resolve_task_config", resolve)
    # Without --ovxr the replay is headless; standalone OvPhysX needs no Kit either way.
    prepared, success = namespace["_prepare_env_cfg"]("test-task", 1, None, overrides=namespace["hydra_overrides"])
    assert resolutions == [("test-task", None, ["physics=ovphysx"])]
    assert prepared is cfg and success is not None
    assert cfg.sim.physics is physics
    assert not getattr(namespace["args_cli"], "require_kit", False)
    assert not namespace["args_cli"].start_on_replay
    legacy_args = replay._build_parser().parse_args(
        ["--task", "Test", "--replay_file", "recording.mcap", "--start_on_replay"]
    )
    assert legacy_args.start_on_replay
    assert not hasattr(namespace["args_cli"], "physics")


@pytest.mark.parametrize(
    ("kitless", "ovxr", "rejected"), [(True, False, True), (True, True, False), (False, False, False)]
)
def test_kitless_replay_without_ovxr_rejects_a_task_target_frame(monkeypatch, kitless, ovxr, rejected):
    """Only Kit or the OVXR session reads the task's target frame, so a headless kitless replay stops before launch."""
    import isaaclab_teleop
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay

    from isaaclab.sim import SimulationCfg

    from isaaclab_tasks.utils import parse_cfg

    argv = ["teleop_replay_agent.py", "--task", "Test", "--replay_file", "recording.mcap"]
    if ovxr:
        argv.append("--ovxr")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(replay, "args_cli", None, raising=False)
    monkeypatch.setattr(replay, "hydra_overrides", None, raising=False)
    replay._parse_args()
    cfg = SimpleNamespace(
        # An unset physics selection runs Isaac Sim PhysX, which needs Kit.
        sim=SimulationCfg(physics=OvPhysxCfg() if kitless else None),
        scene=SimpleNamespace(num_envs=1),
        isaac_teleop=isaaclab_teleop.IsaacTeleopCfg(target_frame_prim_path="{ENV_REGEX_NS}/Robot/base"),
        terminations=SimpleNamespace(),
    )
    monkeypatch.setattr(replay, "ManagerBasedRLEnvCfg", SimpleNamespace)
    monkeypatch.setattr(parse_cfg, "resolve_task_config", lambda *_args, **_kwargs: (cfg, None))
    if rejected:
        with pytest.raises(ValueError, match=r"'test-task' sets isaac_teleop\.target_frame_prim_path.*--ovxr.*Kit"):
            replay._prepare_env_cfg("test-task", 1, None, overrides=replay.hydra_overrides)
    else:
        prepared, _success = replay._prepare_env_cfg("test-task", 1, None, overrides=replay.hydra_overrides)
        assert prepared is cfg


@pytest.mark.parametrize(
    "backend,viewer,hard_reset",
    [
        ("ovphysx", False, False),
        ("ovphysx", True, False),
        ("physx", False, True),
        (None, False, True),
        ("newton", False, False),
        ("newton", True, False),
    ],
)
def test_recorded_environment_reset_preserves_backend_policy_and_hand_cursor(monkeypatch, backend, viewer, hard_reset):
    """The imported replay loop preserves backend reset order and the input cursor."""
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay

    calls = []
    frame = -1

    class Device:
        target_frame_prim_path = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def reset(self):
            calls.append("device-reset")

        def notify_reset(self):
            calls.append("device-notify-reset")

        def request_start(self):
            calls.append("start")

        def advance(self):
            nonlocal frame
            frame += 1
            return torch.zeros(1)

        @property
        def last_control_events(self):
            return SimpleNamespace(is_active=False, should_reset=frame == 0)

    env = SimpleNamespace(
        scene=SimpleNamespace(physics_backend=backend, env_prim_paths=["/World/envs/env_0"]),
        sim=SimpleNamespace(
            device="cpu", is_running=lambda: frame < 1, render=lambda: None, reset=lambda: calls.append("sim-reset")
        ),
        recorder_manager=SimpleNamespace(reset=lambda: calls.append("recorder-reset")),
        reset=lambda: calls.append("env-reset"),
        step_dt=0,
    )
    presentation = (
        SimpleNamespace(
            session=SimpleNamespace(shutdown_requested=False, wait_for_shutdown=lambda _: None),
            reset_pending=False,
            start=lambda *_args, **_kwargs: True,
            publish=lambda *_args, **kwargs: calls.append("publish-reset") if kwargs.get("reset") else None,
        )
        if viewer
        else None
    )
    monkeypatch.setattr(
        replay,
        "args_cli",
        SimpleNamespace(
            device="cpu",
            replay_file="recording.mcap",
            start_on_replay=True,
            enable_debug_visualization=False,
            max_replay_duration_s=1,
            max_stage_load_wait_s=1,
        ),
        raising=False,
    )
    monkeypatch.setattr(replay, "NvmlGpuStatsProvider", lambda: SimpleNamespace(summary=lambda: {}))
    replay._run_single_replay(env, SimpleNamespace(), None, 1, 2, Device(), presentation)
    initial = ["sim-reset"] if hard_reset else []
    initial += ["env-reset", "device-reset", "start"]
    if viewer:
        initial.append("publish-reset")
    recorded = ["sim-reset"] if hard_reset else []
    # The recorded RESET was already consumed by the control channel, so only the
    # notification resets retargeters and withholds pre-reset actions.
    recorded += ["recorder-reset", "env-reset", "device-reset", "device-notify-reset", "start"]
    if viewer:
        recorded.append("publish-reset")
    assert calls == initial + recorded
    assert frame == 1


def _load_teleop_agent(monkeypatch, *arguments):
    from isaaclab_teleop.entrypoints import teleop_se3_agent as module

    monkeypatch.setattr(sys, "argv", ["teleop_se3_agent.py", *arguments])
    monkeypatch.setattr(module, "args_cli", None, raising=False)
    module._parse_args()
    return module


def test_teleop_launcher_preserves_the_resolved_physics_configuration(monkeypatch):
    """Launch validation must not replace task-specific settings on the selected backend."""
    agent = _load_teleop_agent(monkeypatch, "--task", "Test")
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_teleop import IsaacTeleopCfg

    physics_cfg = NewtonCfg()
    physics_cfg.solver_cfg.nconmax = 4096
    physics_cfg.solver_cfg.njmax = 1536
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=physics_cfg),
        scene=SimpleNamespace(),
        terminations=SimpleNamespace(time_out=None),
        isaac_teleop=IsaacTeleopCfg(),
    )
    monkeypatch.setattr(agent, "ManagerBasedRLEnvCfg", SimpleNamespace)
    monkeypatch.setattr(agent, "resolve_task_config", lambda *_args: (env_cfg, None))
    resolved = agent._resolve_env_cfg()[0]
    assert resolved is env_cfg
    assert not hasattr(agent.args_cli, "physics")
    assert env_cfg.sim.physics is physics_cfg
    assert physics_cfg.solver_cfg.nconmax == 4096
    assert physics_cfg.solver_cfg.njmax == 1536


def test_ovxr_presentation_starts_once_and_adopts_later_input_anchors():
    """A failed start leaves the worker unstarted; a started worker adopts each later device's anchor."""
    from unittest.mock import Mock

    from isaaclab_teleop.entrypoints._ovxr_presentation import OvxrPresentation

    session = Mock()
    session.start_render_worker.return_value = False
    presentation = OvxrPresentation(session)
    presentation.input_device = SimpleNamespace(xr_render_anchor_path="/Anchor", external_anchor_state="first")
    assert presentation.start("env", timeout_s=5.0, required_openxr_extensions=("XR_ext",)) is False
    assert presentation.started is False
    session.start_render_worker.assert_called_once_with(
        "env",
        tracking_anchor_path="/Anchor",
        required_openxr_extensions=("XR_ext",),
        external_anchor_state="first",
        timeout_s=5.0,
    )

    session.start_render_worker.return_value = True
    assert presentation.start("env", timeout_s=5.0)
    presentation.input_device = SimpleNamespace(external_anchor_state="second")
    assert presentation.start("env", timeout_s=5.0)
    assert session.start_render_worker.call_count == 2
    session.set_external_anchor_state.assert_called_once_with("second")


def test_ovxr_presentation_retains_a_forced_reset_until_admitted():
    """A reset publication stays forced across declined requests; later changes publish unforced."""
    from isaaclab_teleop.entrypoints._ovxr_presentation import OvxrPresentation

    requests = []
    admissions = iter([None, 1, None, 2])
    step = 0

    def request(identity, *, force):
        requests.append((identity, force))
        return next(admissions)

    presentation = OvxrPresentation(SimpleNamespace(request_render=request))
    env = SimpleNamespace(sim=SimpleNamespace(get_physics_step_count=lambda: step))
    presentation.mark_changed(env, reset=True)
    assert requests == [] and presentation.reset_pending
    assert not presentation.publish(env)
    assert presentation.publish(env)
    assert not presentation.reset_pending
    step = 1
    assert not presentation.publish(env, changed=True)
    assert presentation.publish(env)
    assert presentation.publish(env)  # Nothing pending, so nothing is requested.
    assert requests == [((1, 0), True), ((1, 0), True), ((1, 1), False), ((1, 1), False)]


@pytest.mark.parametrize(("drained", "shutdown_fails"), [(True, False), (False, False), (False, True)])
def test_ovxr_presentation_closes_only_after_input_drain(drained, shutdown_fails):
    """An undrained input device leaves the session open and only requests shutdown."""
    from unittest.mock import Mock

    from isaaclab_teleop.entrypoints._ovxr_presentation import OvxrPresentation

    session = Mock()
    if shutdown_fails:
        session.request_shutdown.side_effect = RuntimeError("shutdown failed")
    presentation = OvxrPresentation(session)
    presentation.input_device = SimpleNamespace(session_cleanup_failed=not drained)
    if drained:
        presentation.close()
        session.close.assert_called_once_with()
        session.request_shutdown.assert_not_called()
        return
    with pytest.raises(RuntimeError, match="input did not drain") as caught:
        presentation.close()
    session.request_shutdown.assert_called_once_with()
    session.close.assert_not_called()
    assert any("shutdown failed" in note for note in getattr(caught.value, "__notes__", ())) is shutdown_fails


@pytest.mark.parametrize(
    ("configs", "kit_sources", "message"),
    [
        pytest.param((), ("renderer",), "kitless runtime components", id="kit-source"),
        pytest.param(("visualizer",), (), "not currently supported", id="local-visualizer"),
        pytest.param(("kit",), ("visualizer",), "deprecated Kit visualizer", id="kit-visualizer"),
    ],
)
def test_ovxr_rejects_kit_runtimes_and_local_visualizers(configs, kit_sources, message):
    """Live and replay share one rejection for components that compete with the OVXR frame loop."""
    from isaaclab_teleop.entrypoints._ovxr_launch import reject_ovxr_runtime_conflicts

    from isaaclab.visualizers import VisualizerBackendCfg

    configs = tuple(VisualizerBackendCfg(visualizer_type=cfg) for cfg in configs)
    reject_ovxr_runtime_conflicts((object(),), ())
    with pytest.raises(ValueError, match=message):
        reject_ovxr_runtime_conflicts(configs, kit_sources)


_UNSET_PHYSICS_TASK = "IsaacContrib-PickPlace-GR1T2-Abs"
_UNSET_PHYSICS_OVXR_ERROR = rf"task '{_UNSET_PHYSICS_TASK}' leaves sim.physics unset.*Run this task with --xr"


def _unset_task_physics(monkeypatch, module):
    """Resolve the real task configuration, then leave ``sim.physics`` at its default ``None``."""
    real_resolve = module.resolve_task_config

    def resolve(*args, **kwargs):
        env_cfg, agent_cfg = real_resolve(*args, **kwargs)
        env_cfg.sim.physics = None
        return env_cfg, agent_cfg

    monkeypatch.setattr(module, "resolve_task_config", resolve)


def test_live_unset_task_physics_keeps_simulation_default(monkeypatch):
    """``physics=None`` must reach SimulationContext's PhysxCfg default and launch Kit for it."""
    from isaaclab.app.sim_launcher import validate_launch

    agent = _load_teleop_agent(monkeypatch, "--task", _UNSET_PHYSICS_TASK, "--disable_external_cameras")
    _unset_task_physics(monkeypatch, agent)
    env_cfg, _teleop_cfg, _use_isaac_teleop = agent._resolve_env_cfg()
    assert env_cfg.sim.physics is None
    _configs, kit_sources = validate_launch(env_cfg, vars(agent.args_cli).copy())
    assert kit_sources


def test_ovxr_omitted_launch_options_match_documented_defaults(monkeypatch):
    """The shortened ``--ovxr`` command keeps the documented CloudXR profile, launch policy and visualizers."""
    from isaaclab.app.sim_launcher import validate_launch
    from isaaclab.sim import SimulationCfg

    monkeypatch.delenv("LIVESTREAM", raising=False)

    def parse(*arguments):
        return _load_teleop_agent(monkeypatch, *arguments).args_cli

    implicit = parse("--ovxr")
    explicit = parse("--ovxr", "--viz", "none", "--auto_launch_cloudxr", "--cloudxr_env", "cloudxrjs")
    assert implicit.auto_launch_cloudxr is explicit.auto_launch_cloudxr is True
    assert implicit.cloudxr_env == explicit.cloudxr_env
    assert implicit.cloudxr_env.endswith(".env")
    left_cfg, right_cfg = SimulationCfg(), SimulationCfg()
    validate_launch(left_cfg, vars(implicit).copy())
    validate_launch(right_cfg, vars(explicit).copy())
    assert left_cfg.visualizer_cfgs == right_cfg.visualizer_cfgs == []
    assert parse("--ovxr", "--no-auto_launch_cloudxr").auto_launch_cloudxr is False
    assert parse("--ovxr", "--cloudxr_env", "none").cloudxr_env is None


def test_live_ovxr_rejects_unset_task_physics_before_launch(monkeypatch):
    agent = _load_teleop_agent(monkeypatch, "--task", _UNSET_PHYSICS_TASK, "--ovxr")
    _unset_task_physics(monkeypatch, agent)
    with pytest.raises(ValueError, match=_UNSET_PHYSICS_OVXR_ERROR):
        agent._resolve_env_cfg()


@pytest.mark.parametrize("ovxr", [False, True])
def test_replay_unset_task_physics_keeps_simulation_default_or_rejects_ovxr(monkeypatch, ovxr):
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay

    from isaaclab_tasks.utils import parse_cfg

    argv = ["teleop_replay_agent.py", "--task", _UNSET_PHYSICS_TASK, "--replay_file", "recording.mcap"]
    if ovxr:
        argv.append("--ovxr")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(replay, "args_cli", None, raising=False)
    monkeypatch.setattr(replay, "hydra_overrides", None, raising=False)
    replay._parse_args()
    _unset_task_physics(monkeypatch, parse_cfg)
    if ovxr:
        with pytest.raises(ValueError, match=_UNSET_PHYSICS_OVXR_ERROR):
            replay._prepare_env_cfg(_UNSET_PHYSICS_TASK, 1, None, overrides=replay.hydra_overrides)
        return
    env_cfg, _success = replay._prepare_env_cfg(_UNSET_PHYSICS_TASK, 1, None, overrides=replay.hydra_overrides)
    # No physics selection defaults to Isaac Sim PhysX, which launches Kit without a forced requirement.
    assert env_cfg.sim.physics is None
    assert not getattr(replay.args_cli, "require_kit", False)


@pytest.mark.parametrize("close_while_pending", [False, True])
def test_replay_waits_for_initial_and_recorded_reset_publication(monkeypatch, close_while_pending):
    """A denied reset admission cannot consume the next MCAP frame or target pose."""
    from isaaclab_teleop import IsaacTeleopCfg
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay
    from isaaclab_teleop.entrypoints._ovxr_presentation import OvxrPresentation

    frames = []
    resets = []
    admissions = []
    ready = False
    epoch_attempts = {}

    class Device:
        target_frame_prim_path = "/World/Robot"
        xr_render_anchor_path = "/World/XRAnchor"
        external_anchor_state = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def reset(self):
            nonlocal ready
            ready = False

        def notify_reset(self):
            pass

        def request_start(self):
            pass

        def advance(self, *, target_T_world):
            assert ready, "MCAP input advanced before reset publication was admitted"
            frames.append(target_T_world)
            return torch.zeros(1)

        @property
        def last_control_events(self):
            return SimpleNamespace(is_active=False, should_reset=len(frames) == 1)

    def request(identity, *, force=False):
        nonlocal ready
        assert force
        epoch = identity[0]
        epoch_attempts[epoch] = epoch_attempts.get(epoch, 0) + 1
        admissions.append((epoch, epoch_attempts[epoch]))
        if epoch_attempts[epoch] == 1:
            return None
        ready = True
        return len(admissions)

    session = SimpleNamespace(
        shutdown_requested=False,
        start_render_worker=lambda *_args, **_kwargs: True,
        request_render=request,
        report_host_timing=lambda *_args: None,
        get_target_frame_T_world=lambda _path: len(resets),
    )

    def wait(_seconds):
        if close_while_pending:
            session.shutdown_requested = True

    session.wait_for_shutdown = wait
    if close_while_pending:
        session.request_render = lambda *_args, **_kwargs: None
    env = SimpleNamespace(
        scene=SimpleNamespace(physics_backend="ovphysx", env_prim_paths=["/World/envs/env_0"]),
        sim=SimpleNamespace(
            device="cpu",
            is_running=lambda: len(frames) < 2,
            reset=lambda: pytest.fail("A leased OvPhysX stage must not be rebuilt"),
            render=lambda: pytest.fail("The host must not render the leased stage"),
            get_physics_step_count=lambda: 0,
            get_physics_dt=lambda: 0.01,
        ),
        reset=lambda: resets.append("env"),
        recorder_manager=SimpleNamespace(reset=lambda: None),
        step_dt=0.01,
    )
    monkeypatch.setattr(
        replay,
        "args_cli",
        SimpleNamespace(
            device="cpu",
            replay_file="recording.mcap",
            start_on_replay=False,
            enable_debug_visualization=False,
            max_replay_duration_s=1,
            max_stage_load_wait_s=1,
        ),
        raising=False,
    )
    monkeypatch.setattr(replay, "NvmlGpuStatsProvider", lambda: SimpleNamespace(summary=lambda: {}))
    stats = replay._run_single_replay(env, IsaacTeleopCfg(), None, 1, 2, Device(), OvxrPresentation(session))
    if close_while_pending:
        assert frames == []
        assert resets == ["env"]
        assert stats.outcome == "incomplete"
    else:
        assert frames == [1, 2]
        assert resets == ["env", "env"]
        assert admissions == [(1, 1), (1, 2), (2, 1), (2, 2)]


@pytest.mark.parametrize("termination", ["terminated", "truncated"])
def test_live_auto_reset_rebases_dynamic_anchor_before_next_input(monkeypatch, run_live_workflow, termination):
    """Auto-reset changes the anchor epoch without resetting the environment twice."""
    import numpy as np
    from isaaclab_teleop import IsaacTeleopCfg, IsaacTeleopDevice
    from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle
    from isaaclab_teleop.xr_anchor_manager import ExternalXrAnchor
    from isaaclab_teleop.xr_cfg import XrAnchorRotationMode, XrCfg
    from scipy.spatial.transform import Rotation

    from pxr import Usd, UsdGeom

    from isaaclab.sim.utils.stage import use_stage

    stage = Usd.Stage.CreateInMemory()
    with use_stage(stage):
        UsdGeom.Xform.Define(stage, "/World/Robot")
        anchor = ExternalXrAnchor(
            XrCfg(
                anchor_prim_path="/World/Robot",
                fixed_anchor_height=True,
                anchor_rotation_mode=XrAnchorRotationMode.FOLLOW_PRIM,
            )
        )
    lifecycle = TeleopSessionLifecycle(IsaacTeleopCfg(), use_kit_xr_bridge=False)
    initial_pose = np.eye(4)
    initial_pose[:3, 3] = (1, 2, 3)
    reset_pose = np.eye(4)
    reset_pose[:3, :3] = Rotation.from_euler("z", 65, degrees=True).as_matrix()
    reset_pose[:3, 3] = (4, 5, 6)
    pose = initial_pose
    reset_calls = []
    actions = []
    reads = []
    publications = []
    refreshes = []
    rejected_reset = False

    def publish(identity, *, force):
        nonlocal rejected_reset
        publications.append((identity, force))
        if len(actions) == 1 and not rejected_reset:
            rejected_reset = True
            return None
        anchor.commit_external_pose(pose, 0.02, anchor.external_epoch[0], lambda _matrix: None)
        return len(publications)

    def step(action):
        nonlocal pose
        actions.append(action)
        pose = reset_pose
        done = len(actions) == 1
        return (
            None,
            None,
            torch.tensor([done and termination == "terminated"]),
            torch.tensor([done and termination == "truncated"]),
            {},
        )

    def advance():
        reads.append((anchor.external_epoch[0], anchor.get_world_matrix().copy()))
        return torch.ones(1)

    interface = SimpleNamespace(
        advance=advance,
        reset=anchor.reset,
        control_events=lambda: SimpleNamespace(is_active=True, should_reset=False),
        _anchor_manager=anchor,
        _session_lifecycle=lifecycle,
    )
    interface.notify_reset = lambda: IsaacTeleopDevice.notify_reset(interface)
    run_live_workflow(
        SimpleNamespace(
            num_envs=1,
            step_dt=0.02,
            reset=lambda: reset_calls.append("env"),
            step=step,
            sim=SimpleNamespace(
                is_running=lambda: len(reads) < 2,
                get_physics_step_count=lambda: len(actions),
                get_physics_dt=lambda: 0.01,
            ),
        ),
        interface,
        SimpleNamespace(
            shutdown_requested=False,
            request_render=publish,
            report_host_timing=lambda *_args: None,
            wait_for_shutdown=lambda _: None,
        ),
        refresh=lambda: refreshes.append("refresh"),
    )
    assert reset_calls == ["env"]
    assert len(actions) == 2
    assert refreshes == ["refresh"]
    assert reads[1][0] == reads[0][0] + 1
    np.testing.assert_allclose(reads[1][1][:3, 3] - reads[0][1][:3, 3], (3, 3, 3), atol=1e-6)
    # FOLLOW_PRIM rebases the robot's new yaw at reset; it must not rotate the
    # tracking frame by the jump between the two episodes.
    np.testing.assert_allclose(reads[0][1][:3, :3], reads[1][1][:3, :3], atol=1e-6)
    assert publications[:3] == [((1, 0), True), ((2, 1), True), ((2, 1), True)]


@pytest.mark.parametrize("conflict", ["custom-anchor", "kit-visualizer", "kit-visualizer-list"])
def test_ovxr_rejects_task_conflicts_before_generic_launch_validation(monkeypatch, conflict):
    from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

    agent = _load_teleop_agent(
        monkeypatch,
        "--task",
        "IsaacContrib-Stack-Cube-SO101-IK-Abs-v0",
        "--physics",
        "ovphysx",
        "--ovxr",
        *(["--viz", "kit"] if conflict == "kit-visualizer" else []),
    )
    resolve = agent.resolve_task_config

    def resolve_conflict(*args):
        cfg, runner = resolve(*args)
        if conflict == "custom-anchor":
            cfg.isaac_teleop.xr_cfg.anchor_prim_path = "/World/Robot"
            cfg.isaac_teleop.xr_cfg.anchor_rotation_mode = XrAnchorRotationMode.CUSTOM
        elif conflict == "kit-visualizer-list":
            from isaaclab_visualizers.kit import KitBackendCfg

            from isaaclab.visualizers import VisualizerCfg

            cfg.sim.visualizer_cfgs = VisualizerCfg(backend=[KitBackendCfg()])
        return cfg, runner

    monkeypatch.setattr(agent, "resolve_task_config", resolve_conflict)
    monkeypatch.setattr(agent, "validate_launch", lambda *_args: pytest.fail("generic launch validation reached"))
    message = "previous head pose" if conflict == "custom-anchor" else "deprecated Kit visualizer"
    with pytest.raises(ValueError, match=message):
        agent._resolve_env_cfg()
