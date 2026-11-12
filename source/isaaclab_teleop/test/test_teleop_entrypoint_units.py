# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-process teleoperation entrypoint, configuration, and control-loop contracts."""

import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Repository root: this file lives at source/isaaclab_teleop/test/<file>.py
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("kitless", "selection", "expected", "reject"),
    [
        (True, None, None, False),
        (True, " 1 ", " 1 ", False),
        (True, "0", "0", True),
        (True, "invalid", "invalid", True),
        (False, None, None, False),
    ],
)
@pytest.mark.parametrize("with_preparation", [False, True])
def test_teleop_ovxr_validates_external_stage_without_changing_environment(
    monkeypatch, kitless, selection, expected, reject, with_preparation
):
    """The task resolver preserves the process setting and rejects incompatible explicit modes before launch."""
    from types import SimpleNamespace

    from isaaclab_teleop import (
        IsaacTeleopCfg,
        se3_gamepad_teleop_cfg,
        se3_keyboard_teleop_cfg,
        se3_spacemouse_teleop_cfg,
    )
    from isaaclab_teleop.device_selection import select_teleop_cfg

    from isaaclab.devices import DeviceCfg

    from isaaclab_tasks.utils import hydra

    monkeypatch.setattr(hydra, "enumerate_task_presets", lambda _task: None)
    name = "ISAAC_LAB_OVRTX_USE_OVSTAGE"
    if selection is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, selection)
    preparations = []
    from isaaclab_ov.physics import OvPhysxCfg

    config = SimpleNamespace(
        sim=SimpleNamespace(device="cpu", physics=OvPhysxCfg()),
        scene=SimpleNamespace(num_envs=1),
        terminations=SimpleNamespace(time_out=1),
        isaac_teleop=IsaacTeleopCfg(
            prepare_environment=(lambda selected: preparations.append(selected)) if with_preparation else None,
        ),
    )
    scans = []
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    namespace = {
        "__package__": "isaaclab_teleop.entrypoints",
        "IsaacTeleopCfg": IsaacTeleopCfg,
        "DeviceCfg": DeviceCfg,
        "select_teleop_cfg": select_teleop_cfg,
        "se3_keyboard_teleop_cfg": se3_keyboard_teleop_cfg,
        "se3_spacemouse_teleop_cfg": se3_spacemouse_teleop_cfg,
        "se3_gamepad_teleop_cfg": se3_gamepad_teleop_cfg,
        "os": os,
        "sys": sys,
        "ManagerBasedRLEnvCfg": SimpleNamespace,
        "resolve_task_config": lambda *_args, **_kwargs: (config, None),
        "validate_launch": lambda *_args: (scans.append(os.environ.get(name)) or ((), ())),
        "args_cli": SimpleNamespace(
            task="Test",
            device="cpu",
            num_envs=1,
            teleop_device=None,
            sensitivity=1.0,
            xr=False,
            using_ovxr=kitless,
            ovxr_pip_camera=None,
            geometry_publication_device=None,
            ovrtx_active_cuda_gpus=None,
            disable_external_cameras=False,
            physics=None,
        ),
    }
    for key, value in namespace.items():
        monkeypatch.setattr(agent, key, value, raising=False)
    if reject:
        with pytest.raises(ValueError, match="requires.*OVStage"):
            agent._resolve_env_cfg()
        assert scans == []
    else:
        agent._resolve_env_cfg()
        assert scans == [expected]
    assert os.environ.get(name) == expected

    assert preparations == ([config] if with_preparation and not reject else [])


@pytest.fixture
def run_live_workflow(monkeypatch):
    """Run the imported live owner with finite input, renderer, and simulation endpoints."""
    import contextlib

    # The live OVXR owner imports the session adapter, which needs a Capture release with OVXR
    # session support. The Kit ``teleop`` extra selects an older release; skip rather than error there.
    pytest.importorskip("isaaccapture.ovxr_session_manager")
    import isaaclab_teleop
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    def run(
        env, interface, session, *, conditioner=None, validator=None, step_hz=0.0, refresh=lambda: None, started=True
    ):
        class Device:
            target_frame_prim_path = None
            xr_render_anchor_path = None
            required_openxr_extensions = ()
            external_anchor_state = None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def advance(self):
                return interface.advance()

            def reset(self):
                return interface.reset()

            @property
            def last_control_events(self):
                return interface.control_events()

        camera = SimpleNamespace(bind=lambda _env: contextlib.nullcontext(), refresh=refresh, close=lambda: None)
        session.external_oxr_handles = None
        session.start_render_worker = lambda *_args, **_kwargs: started
        vars(session).setdefault("close", lambda: None)
        session.async_worker_owns_environment = False
        vars(env).setdefault("close", lambda: None)
        env.unwrapped = env
        env.scene = SimpleNamespace(env_prim_paths=["/World/envs/env_0"])
        env.sim.device = "cpu"
        env.sim.visualizers = ()
        cfg = SimpleNamespace(
            sim=SimpleNamespace(device="cpu"),
            isaac_teleop=isaaclab_teleop.IsaacTeleopCfg(
                action_conditioner=conditioner,
                action_validator=validator,
            ),
        )
        monkeypatch.setattr(
            agent,
            "args_cli",
            agent._build_parser().parse_args(["--task", "Test", "--ovxr", "--step_hz", str(step_hz)]),
            raising=False,
        )
        monkeypatch.setattr(agent, "has_kit", lambda: False)
        monkeypatch.setattr(agent.gym, "make", lambda *_args, **_kwargs: SimpleNamespace(unwrapped=env))
        monkeypatch.setattr(agent, "create_teleop_device", lambda *_args, **_kwargs: Device())
        monkeypatch.setattr(isaaclab_teleop, "create_haptic_feedback_driver", lambda *_args: None)
        monkeypatch.setattr(agent, "_install_ovxr_shutdown_handlers", lambda *_args: {})
        monkeypatch.setattr(isaaclab_teleop.XrCameraFeedSession, "prepare", lambda *_args, **_kwargs: camera)
        monkeypatch.setattr(isaaclab_teleop.OvLibrariesXrSession, "prepare", lambda *_args, **_kwargs: session)
        agent.run_teleoperation(cfg, cfg.isaac_teleop, True, ())

    return run


def test_teleop_se3_agent_yields_only_when_ovxr_is_idle(monkeypatch):
    """A paused kitless loop must yield without slowing Kit or active physics frames."""
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(time, "perf_counter", lambda: 10.003)
    yield_idle = agent._pace_ovxr_control_loop

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
    refreshes = []
    publications = []
    applied = []
    seeded_from = []
    stepped_values = []
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

    def step(value):
        applied.append(clock)
        stepped_values.append(value.clone())

    def condition(_env, value, state):
        previous = state.get("frames", 0)
        seeded_from.append(previous)
        state["frames"] = previous + 1
        return value + previous + 2

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
            control_events=lambda: SimpleNamespace(
                is_active=advances not in (2, 4) if sequence else active,
                should_reset=advances == (5 if sequence else 3),
            ),
        ),
        SimpleNamespace(
            shutdown_requested=False,
            wait_for_shutdown=sleep,
            request_render=request,
            report_host_timing=lambda *args: None,
        ),
        conditioner=condition,
        validator=lambda value: None,
        step_hz=step_hz,
        refresh=lambda: refreshes.append(clock),
    )

    assert advances == (5 if sequence else expected_polls)
    assert clock == pytest.approx(
        0.20 if sequence else 0.12
    )  # Work time is credited, rather than adding a full-period sleep.
    assert len(applied) == (3 if sequence else advances if active else 0)
    assert len(resets) == 2  # Initial reset plus the public control event.
    assert len(refreshes) == 1
    assert len(publications) == (4 if sequence else advances + 1 if active else 2)
    assert publications[-1][1] == {"force": True}

    if sequence:
        # Both STOP branches clear an established ramp before the next accepted action.
        assert seeded_from == [0, 0, 0]
        for value in stepped_values:
            torch.testing.assert_close(value, torch.full((1, 36), 3.0))
    elif active:
        assert seeded_from == [0, 1, 2]
        for index, value in enumerate(stepped_values):
            torch.testing.assert_close(value, torch.full((1, 36), 3.0 + index))


def test_paused_loop_retries_the_latest_unaccepted_publication(monkeypatch, run_live_workflow):
    """A paused iteration retries the rejected latest state without another physics step."""
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    advances = 0
    steps = []
    requests = []
    sleeps = []
    reports = []
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
    session.report_host_timing = lambda step, dt: reports.append((step, dt, len(requests)))
    session.wait_for_shutdown = sleep
    monkeypatch.setattr(agent, "time", SimpleNamespace(perf_counter=lambda: 0.0, sleep=sleep))
    env = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        reset=lambda: None,
        step=lambda action: steps.append(action),
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
    assert reports == [(1, 0.005, 2), (1, 0.005, 3)]
    assert sleeps == [(0.02, 3)]  # Pacing follows the retried publication.


def test_live_render_worker_startup_failure_skips_input_and_drains(run_live_workflow):
    """A session that shuts down during worker startup runs no input and still closes before the environment."""
    events = []
    env = SimpleNamespace(
        sim=SimpleNamespace(is_running=lambda: True),
        reset=lambda: events.append("env-reset"),
        close=lambda: events.append("env-close"),
    )
    interface = SimpleNamespace(advance=lambda: events.append("advance"), reset=lambda: None)
    session = SimpleNamespace(shutdown_requested=False, close=lambda: events.append("xr-close"))
    run_live_workflow(env, interface, session, started=False)
    assert events == ["xr-close", "env-close"]


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
    monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay

    monkeypatch.setattr(replay, "args_cli", None, raising=False)
    monkeypatch.setattr(replay, "hydra_overrides", None, raising=False)
    namespace = vars(replay)
    replay._parse_args()
    prepared_configs = []
    physics = OvPhysxCfg()
    cfg = SimpleNamespace(
        sim=SimulationCfg(physics=physics),
        scene=SimpleNamespace(num_envs=1),
        isaac_teleop=isaaclab_teleop.IsaacTeleopCfg(
            prepare_environment=lambda selected: prepared_configs.append(selected)
        ),
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
    if not ovxr:
        # Replay without --ovxr starts Kit, which cannot host standalone OvPhysX.
        with pytest.raises(ValueError, match="Invalid backend combination"):
            namespace["_prepare_env_cfg"]("test-task", 1, None, overrides=namespace["hydra_overrides"])
        assert resolutions == [("test-task", None, ["physics=ovphysx"])]
        assert not prepared_configs
        return
    prepared, success, cameras = namespace["_prepare_env_cfg"](
        "test-task", 1, None, overrides=namespace["hydra_overrides"]
    )
    assert resolutions == [("test-task", None, ["physics=ovphysx"])]
    assert prepared is cfg and success is not None and cameras == ()
    assert cfg.sim.physics is physics
    assert namespace["args_cli"].require_kit is False
    assert not namespace["args_cli"].start_on_replay
    legacy_args = replay._build_parser().parse_args(
        ["--task", "Test", "--replay_file", "recording.mcap", "--start_on_replay"]
    )
    assert legacy_args.start_on_replay
    assert not hasattr(namespace["args_cli"], "physics")
    assert prepared_configs == [cfg]
    assert "ISAAC_LAB_OVRTX_USE_OVSTAGE" not in os.environ


@pytest.mark.parametrize(
    "backend,viewer,hard_reset",
    [
        ("ovphysx", False, True),
        ("ovphysx", True, True),
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
            session=SimpleNamespace(shutdown_requested=False),
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
    monkeypatch.setattr(replay, "create_isaac_teleop_device", lambda *_args, **_kwargs: Device())
    monkeypatch.setattr(replay, "NvmlGpuStatsProvider", lambda: SimpleNamespace(summary=lambda: {}))
    replay._run_single_replay(env, SimpleNamespace(), None, 1, 2, presentation)
    initial = ["sim-reset"] if hard_reset else []
    initial += ["env-reset", "device-reset", "start"]
    if viewer:
        initial.append("publish-reset")
    recorded = ["sim-reset"] if hard_reset else []
    recorded += ["recorder-reset", "env-reset", "device-reset", "start"]
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
    agent = _load_teleop_agent(monkeypatch)
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab.app.sim_launcher import validate_launch

    physics_cfg = NewtonCfg()
    physics_cfg.solver_cfg.nconmax = 4096
    physics_cfg.solver_cfg.njmax = 1536
    env_cfg = SimpleNamespace(sim=SimpleNamespace(physics=physics_cfg))
    configs, _ = validate_launch(env_cfg, vars(agent.args_cli).copy())

    assert not hasattr(agent.args_cli, "physics")
    assert env_cfg.sim.physics is physics_cfg
    assert any(cfg is physics_cfg for cfg in configs)
    assert physics_cfg.solver_cfg.nconmax == 4096
    assert physics_cfg.solver_cfg.njmax == 1536


@pytest.mark.parametrize(
    "task_name,physics",
    [
        ("IsaacContrib-PickPlace-GR1T2-Abs", "isaacsim_physx"),
        ("IsaacContrib-PickPlace-GR1T2-Abs", "ovphysx"),
        ("IsaacContrib-PickPlace-GR1T2-Abs", "newton_mjwarp"),
        ("IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "ovphysx"),
        ("IsaacContrib-PickPlace-Locomanipulation-G1-Abs", "newton_mjwarp"),
        ("IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs", "ovphysx"),
        ("IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs", "newton_mjwarp"),
    ],
)
def test_teleop_launcher_prepares_humanoid_backends(monkeypatch, task_name, physics):
    """The entrypoint runs the task's preparation hook for the selected standalone backend."""
    # Standalone backends run under --ovxr, which removes the task cameras before launch validation.
    ovxr = () if physics == "isaacsim_physx" else ("--ovxr",)
    agent = _load_teleop_agent(monkeypatch, "--task", task_name, "--physics", physics, *ovxr)
    env_cfg, _teleop_cfg, _use_isaac_teleop, _selected_cameras = agent._resolve_env_cfg()
    assert env_cfg._prepared_kitless_backend == (None if physics == "isaacsim_physx" else physics)


def test_task_action_hook_validates_before_and_after_conditioning(monkeypatch):
    agent = _load_teleop_agent(monkeypatch)
    calls = []

    def validate(action):
        calls.append(("validate", action))
        return "rejected" if action in (3, 6) else None

    def condition(env, action, state):
        calls.append(("condition", env, action, state))
        state["last"] = action
        return action + 1

    state = {}
    action, issue = agent.prepare_teleop_action("env", 1, condition, validate, state)
    assert (action, issue, state) == (2, None, {"last": 1})
    assert [call[0] for call in calls] == ["validate", "condition", "validate"]

    calls.clear()
    action, issue = agent.prepare_teleop_action("env", 3, condition, validate, state)
    assert (action, issue) == (3, "rejected")
    assert [call[0] for call in calls] == ["validate"]

    # A conditioned action rejected by the second validation leaves the state uncommitted.
    calls.clear()
    action, issue = agent.prepare_teleop_action("env", 5, condition, validate, state)
    assert (action, issue, state) == (6, "rejected", {"last": 1})
    assert [call[0] for call in calls] == ["validate", "condition", "validate"]


def test_teleop_agent_installs_late_graceful_shutdown_handlers(monkeypatch):
    agent = _load_teleop_agent(monkeypatch)
    requests = []
    installed = {}

    class Session:
        def request_shutdown(self):
            requests.append("shutdown")

    def install(signum, handler):
        installed[signum] = handler
        return f"previous-{signum}"

    monkeypatch.setattr(agent.signal, "signal", install)
    previous = agent._install_ovxr_shutdown_handlers(agent.OvxrPresentation(Session()))

    assert set(installed) == {signal.SIGINT, signal.SIGTERM}
    installed[signal.SIGINT](signal.SIGINT, None)
    assert requests == ["shutdown"]

    restored = {}
    monkeypatch.setattr(agent.signal, "signal", lambda signum, handler: restored.update({signum: handler}))
    agent._restore_ovxr_shutdown_handlers(previous)
    assert restored == previous


def test_ovxr_presentation_starts_once_and_adopts_later_input_anchors():
    """A failed start leaves the worker unstarted; a started worker adopts each later device's anchor."""
    from unittest.mock import Mock

    from isaaclab_teleop.entrypoints._ovxr_presentation import OvxrPresentation

    session = Mock()
    session.start_render_worker.return_value = False
    presentation = OvxrPresentation(session, excluded_visual_meshes=("/Hidden",))
    presentation.input_device = SimpleNamespace(xr_render_anchor_path="/Anchor", external_anchor_state="first")
    assert presentation.start("env", timeout_s=5.0, required_openxr_extensions=("XR_ext",)) is False
    assert presentation.started is False
    session.start_render_worker.assert_called_once_with(
        "env",
        tracking_anchor_path="/Anchor",
        required_openxr_extensions=("XR_ext",),
        excluded_visual_meshes=("/Hidden",),
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
    reports = []
    admissions = iter([None, 1, None, 2])
    step = 0

    def request(identity, *, force):
        requests.append((identity, force))
        return next(admissions)

    presentation = OvxrPresentation(
        SimpleNamespace(request_render=request, report_host_timing=lambda step, _dt: reports.append(step))
    )
    env = SimpleNamespace(sim=SimpleNamespace(get_physics_step_count=lambda: step, get_physics_dt=lambda: 0.005))
    presentation.mark_changed(env, reset=True)
    assert requests == [] and presentation.reset_pending
    assert not presentation.publish(env, report_timing=False)
    assert presentation.publish(env, report_timing=False)
    assert not presentation.reset_pending
    step = 1
    assert not presentation.publish(env, changed=True)
    assert presentation.publish(env)
    assert presentation.publish(env)  # Nothing pending, so nothing is requested.
    assert requests == [((1, 0), True), ((1, 0), True), ((1, 1), False), ((1, 1), False)]
    assert reports == [1, 1, 1]  # Only completed host frames report the physics rate.


@pytest.mark.parametrize("shutdown_fails", [False, True])
@pytest.mark.parametrize("drained", [True, False])
def test_ovxr_presentation_closes_only_after_input_drain(drained, shutdown_fails):
    """An undrained input device keeps the session and environment borrowed and only requests shutdown."""
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
    ("configs", "kit_sources"),
    [
        pytest.param((), ("renderer",), id="kit-source"),
        pytest.param(("visualizer",), (), id="local-visualizer"),
    ],
)
def test_ovxr_rejects_kit_runtimes_and_local_visualizers(configs, kit_sources):
    """Live and replay share one rejection for components that compete with the OVXR frame loop."""
    from isaaclab_teleop.entrypoints._ovxr_launch import reject_ovxr_runtime_conflicts

    from isaaclab.visualizers import VisualizerBackendCfg

    configs = tuple(VisualizerBackendCfg() if cfg == "visualizer" else cfg for cfg in configs)
    reject_ovxr_runtime_conflicts((object(),), ())
    with pytest.raises(ValueError, match="no Kit runtime or local visualizer"):
        reject_ovxr_runtime_conflicts(configs, kit_sources)


@pytest.mark.parametrize(
    ("left_active", "right_active", "expected"),
    (
        (False, True, "waiting for initial left hand tracking"),
        (True, False, "waiting for initial right hand tracking"),
    ),
)
def test_kitless_hand_guard_requires_each_hand_once_then_allows_held_dropout(
    monkeypatch, left_active, right_active, expected
):
    agent = _load_teleop_agent(monkeypatch)

    class OptionalHand:
        def __init__(self, active):
            self.is_none = not active

    lifecycle = SimpleNamespace(
        HAND_LEFT_KEY="hand_left",
        HAND_RIGHT_KEY="hand_right",
        last_step_result={
            "hand_left": OptionalHand(left_active),
            "hand_right": OptionalHand(right_active),
        },
    )
    from isaaclab_teleop import IsaacTeleopDevice

    interface = IsaacTeleopDevice.__new__(IsaacTeleopDevice)
    interface._session_lifecycle = lifecycle
    initialized = set()

    assert agent.initial_hand_tracking_issue(interface, initialized) == expected
    lifecycle.last_step_result = {"hand_left": OptionalHand(True), "hand_right": OptionalHand(True)}
    # Numeric zero is not a sentinel: a present, valid fully-open hand frame is accepted.
    assert agent.initial_hand_tracking_issue(interface, initialized) is None
    assert initialized == {"left", "right"}

    lifecycle.last_step_result = {"hand_left": OptionalHand(False), "hand_right": OptionalHand(False)}
    assert agent.initial_hand_tracking_issue(interface, initialized) is None

    initialized.clear()
    assert agent.initial_hand_tracking_issue(interface, initialized) == "waiting for initial left/right hand tracking"


def test_kitless_hand_guard_renders_without_step_and_next_valid_frame_resumes(monkeypatch):
    agent = _load_teleop_agent(monkeypatch)

    class Sim:
        def __init__(self):
            self.render_count = 0

        def render(self):
            self.render_count += 1

    class Env:
        def __init__(self):
            self.num_envs = 1
            self.sim = Sim()
            self.actions = []

        def step(self, action):
            self.actions.append(action.clone())

    env = Env()
    haptics = {"update": 0, "stop": 0}

    class HapticDriver:
        def update(self):
            haptics["update"] += 1

        def stop(self):
            haptics["stop"] += 1

    driver = HapticDriver()
    valid = torch.arange(36, dtype=torch.float32)

    assert agent._apply_teleop_frame(env, valid, None, driver)
    assert not agent._apply_teleop_frame(env, torch.zeros_like(valid), "missing left hand tracking", driver)
    assert agent._apply_teleop_frame(env, valid, None, driver)

    assert env.sim.render_count == 1
    assert len(env.actions) == 2
    torch.testing.assert_close(env.actions[0], env.actions[1])
    assert haptics == {"update": 2, "stop": 1}


@pytest.mark.parametrize("entrypoint", ["live", "replay"])
@pytest.mark.parametrize(
    "flags, expected",
    [
        ([], True),
        (["--headless"], False),
    ],
)
def test_ovxr_mirror_cli_reaches_session_construction(monkeypatch, entrypoint, flags, expected):
    """The real entrypoint parser controls the config passed to XR session preparation."""
    # Session construction needs a Capture release with OVXR session support; the Kit
    # ``teleop`` extra selects an older release, so skip there like ``run_live_workflow``.
    pytest.importorskip("isaaccapture.ovxr_session_manager")
    import isaaclab_teleop
    from isaaclab_teleop.entrypoints import teleop_replay_agent, teleop_se3_agent

    module = teleop_se3_agent if entrypoint == "live" else teleop_replay_agent
    monkeypatch.setattr(sys, "argv", ["teleop-test"])
    monkeypatch.setattr(module, "args_cli", None, raising=False)
    argv = ["--task", "Test", "--ovxr", *flags]
    if entrypoint == "replay":
        argv += ["--replay_file", "recording.mcap"]
        monkeypatch.setattr(module, "hydra_overrides", None, raising=False)
    module._parse_args(argv)
    assert module.args_cli.headless is ("--headless" in flags)
    cfg = SimpleNamespace(sim=SimpleNamespace(device="cpu"), isaac_teleop=isaaclab_teleop.IsaacTeleopCfg())
    received = []

    class PreparationReached(Exception):
        pass

    def prepare(_sim, xr_cfg, *_args, **_kwargs):
        assert isinstance(xr_cfg, isaaclab_teleop.OvLibrariesXrCfg)
        received.append(xr_cfg.desktop_mirror_enabled)
        raise PreparationReached

    monkeypatch.setattr(isaaclab_teleop.OvLibrariesXrSession, "prepare", prepare)
    if entrypoint == "live":
        monkeypatch.setattr(module, "has_kit", lambda: False)
        monkeypatch.setattr(
            isaaclab_teleop.XrCameraFeedSession,
            "prepare",
            lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
        )
        with pytest.raises(PreparationReached):
            module.run_teleoperation(cfg, cfg.isaac_teleop, True, ())
    else:
        with pytest.raises(PreparationReached):
            module._run_replay_batch(cfg, None, ())
    assert received == [expected]
    assert isaaclab_teleop.OvLibrariesXrCfg().desktop_mirror_enabled is False


@pytest.mark.parametrize("entrypoint, flags", [("live", ["--xr", "--headless"]), ("replay", ["--headless"])])
def test_headless_requires_ovxr(monkeypatch, capsys, entrypoint, flags):
    """Kit runs headless by default, so only the OVXR desktop mirror accepts ``--headless``."""
    from isaaclab_teleop.entrypoints import teleop_replay_agent, teleop_se3_agent

    module = teleop_se3_agent if entrypoint == "live" else teleop_replay_agent
    monkeypatch.setattr(sys, "argv", ["teleop-test"])
    monkeypatch.setattr(module, "args_cli", None, raising=False)
    argv = ["--task", "Test", *flags]
    if entrypoint == "replay":
        argv += ["--replay_file", "recording.mcap"]
        monkeypatch.setattr(module, "hydra_overrides", None, raising=False)
    with pytest.raises(SystemExit):
        module._parse_args(argv)
    assert "--headless requires --ovxr" in capsys.readouterr().err


def _run_ovxr_pip_workflow(monkeypatch, entrypoint, layout, flags, calls, *, pip_camera=True):
    """Run an entrypoint's ``main`` with one PiP camera until it prepares the OVXR session.

    Task resolution and launch are stubbed; CLI parsing, pre-launch preparation and session
    construction are production code. ``calls`` records the kept cameras, launch and preparation.
    With ``pip_camera=False`` the task camera exists but no PiP camera is selected.

    Returns:
        The task configuration and the arguments passed to session preparation.
    """
    from contextlib import contextmanager, suppress

    import isaaclab_teleop
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_teleop.entrypoints import _ovxr_launch, teleop_replay_agent, teleop_se3_agent

    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import SimulationContext

    from isaaclab_tasks.utils import parse_cfg

    config = SimpleNamespace(
        sim=SimpleNamespace(device="cpu", physics=OvPhysxCfg()),
        scene=SimpleNamespace(num_envs=1, head_cam=CameraCfg(prim_path="/World/head_cam", width=640, height=480)),
        terminations=SimpleNamespace(success=None, time_out=None),
        isaac_teleop=isaaclab_teleop.IsaacTeleopCfg(
            xr_camera_feeds=[isaaclab_teleop.XrCameraFeedCfg(camera_name="head_cam", offset_m=(0.1, -0.15))],
            xr_camera_feed_layout=layout,
        ),
    )
    prepared = []

    class SessionReached(RuntimeError):
        pass

    @contextmanager
    def launch(selected_cfg, _args, *, cleanup_context):
        assert selected_cfg is config
        calls.append("launch")
        yield

    def prepare(sim_cfg, xr_cfg, renderer_cfg=None):
        assert sim_cfg is config.sim
        calls.append("prepare")
        prepared.append((xr_cfg, renderer_cfg))
        raise SessionReached

    def remove_camera_configs(selected_cfg, *, keep=()):
        calls.append(("keep", keep))
        return selected_cfg

    module = teleop_se3_agent if entrypoint == "live" else teleop_replay_agent
    monkeypatch.setattr(sys, "argv", ["teleop-test"])
    monkeypatch.setattr(module, "args_cli", None, raising=False)
    monkeypatch.setattr(module, "ManagerBasedRLEnvCfg", SimpleNamespace)
    resolver_owner = teleop_se3_agent if entrypoint == "live" else parse_cfg
    monkeypatch.setattr(resolver_owner, "resolve_task_config", lambda *_args, **_kwargs: (config, None))
    monkeypatch.setattr(module, "validate_launch", lambda *_args: ((), ()))
    monkeypatch.setattr(module, "launch_simulation", launch)
    monkeypatch.setattr(_ovxr_launch, "remove_camera_configs", remove_camera_configs)
    # The Kit Capture release has no OVXR session; layout rejection never reaches the session.
    with suppress(ModuleNotFoundError):
        monkeypatch.setattr(isaaclab_teleop.OvLibrariesXrSession, "prepare", prepare)
    monkeypatch.setattr(SimulationContext, "instance", lambda: None)
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    pip = ["--ovxr_pip_camera", "head_cam"] if pip_camera else []
    arguments = ["--task", "Test", "--ovxr", "--cloudxr_env", "none", *pip, *flags]
    if entrypoint == "live":
        monkeypatch.setattr(module, "has_kit", lambda: False)
        monkeypatch.setattr(
            isaaclab_teleop.XrCameraFeedSession,
            "prepare",
            lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
        )
    else:
        monkeypatch.setattr(module, "hydra_overrides", None, raising=False)
        arguments += ["--replay_file", "recording.mcap"]
    module._parse_args(arguments)
    with suppress(SessionReached):
        module.main()
    return config, prepared


@pytest.mark.parametrize("entrypoint", ["live", "replay"])
@pytest.mark.parametrize(
    "placement, flags, position",
    [
        # A head-locked task layout keeps its panel placement without the flag.
        ("head_locked", [], (0.1, -0.15, -0.8)),
        # The flag places a world layout in the VIEW space, at the requested position.
        ("world", ["--ovxr_pip_head_locked", "--ovxr_pip_view_position", "0", "-0.1", "-0.6"], (0.0, -0.1, -0.6)),
    ],
)
def test_ovxr_pip_cli_reaches_session_construction(monkeypatch, entrypoint, placement, flags, position):
    """The PiP and mirror options of both workflows reach the OVXR session configuration."""
    # Session construction needs a Capture release with OVXR session support; the Kit
    # ``teleop`` extra selects an older release, so skip there like ``run_live_workflow``.
    pytest.importorskip("isaaccapture.ovxr_session_manager")
    from isaaclab_ov.renderers import OVRTXRendererCfg
    from isaaclab_teleop import XrCameraFeedLayoutCfg

    layout = XrCameraFeedLayoutCfg(placement=placement, world_position_m=(1.0, 0.0, 1.5))
    rates = ["--ovxr_pip_fps", "15", "--ovxr_desktop_mirror_fps", "20"]
    calls = []
    config, prepared = _run_ovxr_pip_workflow(monkeypatch, entrypoint, layout, flags + rates, calls)

    assert calls == [("keep", ("head_cam",)), "launch", "prepare"]
    [(xr_cfg, renderer_cfg)] = prepared
    assert xr_cfg.desktop_mirror_enabled is True
    assert xr_cfg.desktop_mirror_max_fps == 20.0
    [camera] = xr_cfg.auxiliary_cameras
    # Configclasses copy their fields, so the panel holds a copy of the scene sensor configuration.
    assert camera.name == "head_cam" and camera.camera_cfg.prim_path == config.scene.head_cam.prim_path
    assert camera.camera_cfg.renderer_cfg == config.scene.head_cam.renderer_cfg
    assert camera.max_fps == 15.0
    assert camera.position == pytest.approx(position)
    assert isinstance(renderer_cfg, OVRTXRendererCfg) and renderer_cfg == config.scene.head_cam.renderer_cfg
    # Forcing head-locked placement leaves the task layout unchanged.
    assert config.isaac_teleop.xr_camera_feed_layout.placement == placement


@pytest.mark.parametrize("entrypoint", ["live", "replay"])
@pytest.mark.parametrize("placement", ["viewer_start", "world"])
def test_ovxr_pip_rejects_task_layout_that_is_not_head_locked_before_launch(monkeypatch, entrypoint, placement):
    """Without ``--ovxr_pip_head_locked``, a task layout OVXR cannot present fails before launch."""
    from isaaclab_teleop import XrCameraFeedLayoutCfg

    layout = XrCameraFeedLayoutCfg(placement=placement, world_position_m=(1.0, 0.0, 1.5))
    calls = []
    with pytest.raises(ValueError, match=rf"{placement!r} placement.*--ovxr_pip_head_locked"):
        _run_ovxr_pip_workflow(monkeypatch, entrypoint, layout, [], calls)
    assert calls == []


@pytest.fixture
def physics_cuda_device():
    """Set ``/physics/cudaDevice``, which resolves a bare ``cuda`` simulation device, for one test."""
    from isaaclab.app import get_settings_manager

    settings = get_settings_manager()
    previous = settings.get("/physics/cudaDevice")
    yield lambda index: settings.set("/physics/cudaDevice", index)
    settings.set("/physics/cudaDevice", previous)


@pytest.mark.parametrize("pip_camera", [False, True])
def test_gpu_selection_cli_reaches_the_shared_xr_renderer(monkeypatch, pip_camera):
    """The GPU options reach the one renderer configuration of the stereo session and its PiP camera."""
    # Session construction needs a Capture release with OVXR session support; the Kit
    # ``teleop`` extra selects an older release, so skip there like ``run_live_workflow``.
    pytest.importorskip("isaaccapture.ovxr_session_manager")
    from isaaclab_teleop import XrCameraFeedLayoutCfg

    flags = ["--ovrtx_active_cuda_gpus", "1", "--geometry_publication_device", "cuda:1"]
    layout = XrCameraFeedLayoutCfg(placement="head_locked")
    config, prepared = _run_ovxr_pip_workflow(monkeypatch, "live", layout, flags, [], pip_camera=pip_camera)

    [(xr_cfg, renderer_cfg)] = prepared
    assert renderer_cfg.active_cuda_gpus == "1"
    assert renderer_cfg.geometry_publication_device == "cuda:1"
    # Configclasses copy their fields, so each panel holds a copy of the scene sensor configuration.
    assert [camera.camera_cfg.prim_path for camera in xr_cfg.auxiliary_cameras] == (
        [config.scene.head_cam.prim_path] if pip_camera else []
    )
    assert all(camera.camera_cfg.renderer_cfg == renderer_cfg for camera in xr_cfg.auxiliary_cameras)


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--ovxr", "--ovrtx_active_cuda_gpus", "cuda:0"], "active_cuda_gpus must list CUDA indices"),
        (["--ovxr", "--ovrtx_active_cuda_gpus", "0,0"], "more than once"),
        (["--ovxr", "--geometry_publication_device", "cuda"], "explicit CUDA device"),
        (["--ovrtx_active_cuda_gpus", "0"], "require --ovxr"),
        (["--geometry_publication_device", "cuda:0"], "require --ovxr"),
    ],
)
def test_gpu_selection_cli_rejects_invalid_options_at_parse_time(monkeypatch, capsys, arguments, message):
    with pytest.raises(SystemExit):
        _load_teleop_agent(monkeypatch, *arguments)
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--ovrtx_active_cuda_gpus", "0", "--geometry_publication_device", "cuda:1"], r"rendering GPU \[0\]"),
        (["--device", "cpu", "--geometry_publication_device", "cuda:0"], "requires --ovrtx_active_cuda_gpus"),
    ],
)
def test_geometry_publication_device_must_be_a_rendering_gpu(monkeypatch, flags, message):
    """Copies to a GPU that does not render add work without helping the renderer, so launch never starts."""
    from isaaclab_teleop import XrCameraFeedLayoutCfg

    calls = []
    with pytest.raises(ValueError, match=message):
        _run_ovxr_pip_workflow(monkeypatch, "live", XrCameraFeedLayoutCfg(), flags, calls, pip_camera=False)
    assert calls == []


@pytest.mark.parametrize(
    "device, physics_cuda_index, publication, accepted",
    [
        ("cuda:1", None, "cuda:0", False),
        ("cuda", 1, "cuda:0", False),
        ("cuda", None, "cuda:0", True),
        ("cuda", 1, "cuda:1", True),
    ],
)
def test_geometry_publication_device_follows_the_simulation_gpu(
    monkeypatch, physics_cuda_device, device, physics_cuda_index, publication, accepted
):
    """Without explicit rendering GPUs, publication must target the simulation GPU that launch selects.

    A bare ``cuda`` device follows ``/physics/cudaDevice``, as the launcher does.
    """
    # The simulation GPU resolves through the OVXR session module, which loads the OVRTX renderer.
    pytest.importorskip("isaaclab_teleop.ovxr_session")
    if accepted:
        # Session construction needs a Capture release with OVXR session support.
        pytest.importorskip("isaaccapture.ovxr_session_manager")
    from isaaclab_teleop import XrCameraFeedLayoutCfg

    physics_cuda_device(physics_cuda_index)
    flags = ["--device", device, "--geometry_publication_device", publication]
    calls = []
    if not accepted:
        with pytest.raises(ValueError, match=r"rendering GPU \[1\]"):
            _run_ovxr_pip_workflow(monkeypatch, "live", XrCameraFeedLayoutCfg(), flags, calls, pip_camera=False)
        assert calls == []
        return
    _, prepared = _run_ovxr_pip_workflow(monkeypatch, "live", XrCameraFeedLayoutCfg(), flags, calls, pip_camera=False)
    [(_, renderer_cfg)] = prepared
    assert renderer_cfg.geometry_publication_device == publication


@pytest.mark.parametrize("entrypoint", ["live", "replay"])
def test_task_preparation_precedes_ovxr_simulation_context(monkeypatch, entrypoint):
    """Both actual workflows finalize the selected config before OVXR allocates its context."""
    pytest.importorskip("ovrtx")
    pytest.importorskip("isaaccapture.ovxr_session_manager")
    from contextlib import contextmanager

    import isaaclab_teleop
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_teleop import ovxr_session
    from isaaclab_teleop.entrypoints import teleop_replay_agent, teleop_se3_agent

    from isaaclab.sim import SimulationCfg, SimulationContext

    from isaaclab_tasks.utils import parse_cfg

    calls = []

    def prepare_environment(selected_cfg):
        assert selected_cfg is config
        calls.append("prepare")
        selected_cfg.sim.dt = 0.007

    config = SimpleNamespace(
        sim=SimulationCfg(physics=OvPhysxCfg()),
        scene=SimpleNamespace(num_envs=1),
        terminations=SimpleNamespace(success=None, time_out=None),
        isaac_teleop=isaaclab_teleop.IsaacTeleopCfg(prepare_environment=prepare_environment),
    )

    class ContextReached(RuntimeError):
        pass

    class ContextAllocation:
        def __init__(self, sim_cfg):
            assert sim_cfg is config.sim
            assert sim_cfg.dt == 0.007
            assert calls == ["prepare", "launch"]
            calls.append("context")
            raise ContextReached

        @staticmethod
        def instance():
            return None

    @contextmanager
    def launch(selected_cfg, _args, *, cleanup_context):
        assert selected_cfg is config
        assert not cleanup_context
        calls.append("launch")
        yield

    def validate_launch(selected_cfg, _args):
        assert selected_cfg is config
        return (), ()

    module = teleop_se3_agent if entrypoint == "live" else teleop_replay_agent
    monkeypatch.setattr(sys, "argv", ["teleop-test"])
    monkeypatch.setattr(module, "args_cli", None, raising=False)
    monkeypatch.setattr(module, "ManagerBasedRLEnvCfg", SimpleNamespace)
    resolver_owner = teleop_se3_agent if entrypoint == "live" else parse_cfg
    monkeypatch.setattr(resolver_owner, "resolve_task_config", lambda *_args, **_kwargs: (config, None))
    monkeypatch.setattr(module, "validate_launch", validate_launch)
    monkeypatch.setattr(module, "launch_simulation", launch)
    monkeypatch.setattr(SimulationContext, "instance", lambda: None)
    monkeypatch.setattr(ovxr_session, "SimulationContext", ContextAllocation)
    monkeypatch.setitem(sys.modules, "ovxr", SimpleNamespace(apply_renderer_settings=lambda _settings: None))
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    monkeypatch.delenv("ISAAC_LAB_SHARE_OVSTAGE", raising=False)
    arguments = ["--task", "Test", "--ovxr", "--cloudxr_env", "none"]
    if entrypoint == "live":
        monkeypatch.setattr(module, "has_kit", lambda: False)
        monkeypatch.setattr(
            isaaclab_teleop.XrCameraFeedSession,
            "prepare",
            lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None),
        )
    else:
        monkeypatch.setattr(module, "hydra_overrides", None, raising=False)
        arguments += ["--replay_file", "recording.mcap"]
    module._parse_args(arguments)

    with pytest.raises(ContextReached):
        module.main()
    assert calls == ["prepare", "launch", "context"]


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
    env_cfg = agent._resolve_env_cfg()[0]
    assert env_cfg.sim.physics is None
    _configs, kit_sources = validate_launch(env_cfg, vars(agent.args_cli).copy())
    assert kit_sources


def test_live_ovxr_rejects_unset_task_physics_before_launch(monkeypatch):
    monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
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
    monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    monkeypatch.setattr(replay, "args_cli", None, raising=False)
    monkeypatch.setattr(replay, "hydra_overrides", None, raising=False)
    replay._parse_args()
    _unset_task_physics(monkeypatch, parse_cfg)
    if ovxr:
        with pytest.raises(ValueError, match=_UNSET_PHYSICS_OVXR_ERROR):
            replay._prepare_env_cfg(_UNSET_PHYSICS_TASK, 1, None, overrides=replay.hydra_overrides)
        return
    env_cfg, _success, _cameras = replay._prepare_env_cfg(
        _UNSET_PHYSICS_TASK, 1, None, overrides=replay.hydra_overrides
    )
    assert env_cfg.sim.physics is None
    assert replay.args_cli.require_kit


def test_rejected_ovxr_frame_does_not_render_on_the_control_thread(monkeypatch, run_live_workflow):
    """OVXR owns rendering, so a rejected frame holds the scene without a control-thread render or step."""
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    advances = 0
    renders = []
    steps = []
    requests = []
    waits = []

    def advance():
        nonlocal advances
        advances += 1
        return torch.ones((1, 2))

    def request(identity, *, force):
        requests.append(force)
        return 1

    monkeypatch.setattr(agent, "time", SimpleNamespace(perf_counter=lambda: 0.0, sleep=waits.append))
    env = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        reset=lambda: None,
        step=steps.append,
        sim=SimpleNamespace(
            is_running=lambda: advances < 1,
            get_physics_step_count=lambda: len(steps),
            get_physics_dt=lambda: 0.005,
            render=lambda: renders.append(True),
        ),
    )
    run_live_workflow(
        env,
        SimpleNamespace(
            advance=advance,
            reset=lambda: None,
            control_events=lambda: SimpleNamespace(is_active=True, should_reset=False),
        ),
        SimpleNamespace(
            shutdown_requested=False,
            request_render=request,
            wait_for_shutdown=waits.append,
            report_host_timing=lambda *args: None,
        ),
        validator=lambda _action: "rejected",
    )
    assert advances == 1
    assert renders == [] and steps == []
    assert requests == [True]  # Only the reset publication; the held scene is not republished.
    assert waits == [0.02]  # A rejected frame waits one action period, like a paused one.
