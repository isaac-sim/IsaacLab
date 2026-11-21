# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute the real replay bootstrap without starting a simulator.

Only device/simulator endpoints are isolated; the replay/reset function bodies
and Kit detection are production code. These tests do not qualify MCAP motion.
"""

import builtins
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "kit_active,run_index,backend,viewer,hard_reset",
    [
        (False, 0, "ovphysx", False, False),
        (True, 0, "physx", False, True),
        (True, 1, "physx", False, True),
        (False, 0, "ovphysx", True, False),
    ],
)
def test_replay_bootstrap_waits_only_for_an_active_kit_application(
    monkeypatch, kit_active, run_index, backend, viewer, hard_reset
):
    """A viewer preserves backend reset policy; kitless replay never waits for Kit assets."""
    calls = []

    class Device:
        xr_render_anchor_path = None
        external_anchor_state = None

        def __enter__(self):
            calls.append("device-enter")
            return self

        def reset(self):
            calls.append("device-reset")

        def request_start(self):
            calls.append("device-start")

        def __exit__(self, *args):
            calls.append("device-exit")

    app = ModuleType("omni.kit.app")
    app.get_app = lambda: object()
    if kit_active:
        monkeypatch.setitem(sys.modules, "omni.kit.app", app)
    else:
        monkeypatch.delitem(sys.modules, "omni.kit.app", raising=False)
    wait_module = ModuleType("isaaclab_isaacsim.renderers.isaac_rtx_renderer_utils")
    wait_module.wait_for_stage_load = lambda *args, **kwargs: calls.append("kit-wait")
    monkeypatch.setitem(sys.modules, wait_module.__name__, wait_module)
    original_import = builtins.__import__

    def import_with_kit_guard(name, *args, **kwargs):
        if name == wait_module.__name__ and not kit_active:
            raise AssertionError("Kit-only stage loader imported by a kitless replay")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_kit_guard)

    def render():
        calls.append("render")

    env = SimpleNamespace(
        scene=SimpleNamespace(physics_backend=backend, env_prim_paths=["/Lab/Cells/Cell0"]),
        sim=SimpleNamespace(
            device="cpu",
            render=render,
            reset=lambda: calls.append("sim-reset"),
            is_running=lambda: False,
            get_physics_step_count=lambda: 0,
        ),
        reset=lambda: calls.append("env-reset"),
    )
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay
    from isaacteleop.teleop_session_manager import RetargetingExecutionConfig

    cfg = SimpleNamespace(retargeting_execution=RetargetingExecutionConfig(mode="pipelined"))
    namespace = {
        "args_cli": SimpleNamespace(
            device="cpu",
            replay_file="fixture.mcap",
            enable_debug_visualization=False,
            max_stage_load_wait_s=5,
            replay_start_delay_s=0,
            start_on_replay=False,
            max_replay_duration_s=1,
        ),
        "NvmlGpuStatsProvider": lambda: SimpleNamespace(_available=True, summary=lambda: {}),
        "_RENDERER_SETTLE_FRAMES": 30,
    }
    for name, value in namespace.items():
        monkeypatch.setattr(replay, name, value, raising=False)
    presentation = None
    if viewer:
        from isaaclab_teleop.entrypoints._ovxr_presentation import OvxrPresentation

        session = SimpleNamespace(
            start_render_worker=lambda *_args, **_kwargs: calls.append("viewer-start") or True,
            request_render=lambda identity, *, force: calls.append(("publish", identity, force)) or 1,
        )
        presentation = OvxrPresentation(session)
    replay._run_single_replay(env, cfg, None, run_index, 2, Device(), presentation)
    expected = ["device-enter"]
    if hard_reset:
        expected.append("sim-reset")
    expected.extend(["env-reset", "device-reset"])
    if viewer:
        expected.extend(["viewer-start", ("publish", (1, 0), True)])
    if kit_active and run_index == 0:
        expected.append("kit-wait")
    assert calls == [*expected, "device-exit"]
    assert env.sim.render is render


@pytest.mark.parametrize(
    "stop_at,requested",
    [
        ("startup", 2),
        ("startup-second-signal", 2),
        ("construction-new", 2),
        ("construction-previous", 2),
        ("after-first", 2),
        ("after-first", 1),
        ("native-replaced", 1),
    ],
)
def test_presented_replay_preserves_startup_ownership_and_batch_completion(monkeypatch, stop_at, requested):
    """Failed construction preserves prior owners; a partial batch never reports completion."""
    from isaaclab.sim import SimulationContext

    calls = []
    previous = object()
    context_state = {"current": previous}

    def clear_context():
        calls.append("context-clear")
        context_state["current"] = None

    monkeypatch.setattr(SimulationContext, "instance", lambda: context_state["current"])
    monkeypatch.setattr(SimulationContext, "clear_instance", clear_context)
    handlers = {}
    reports = []
    session = SimpleNamespace(shutdown_requested=False, request_shutdown=lambda: None)

    class Presentation:
        @classmethod
        def prepare(cls, cfg, args, *, application_name):
            assert application_name == "isaaclab-ovxr-replay-task"
            return cls()

        def __init__(self):
            self.session = session
            self.started = False

        def close(self):
            if stop_at == "startup-second-signal":
                handlers[2](2, None)
            calls.append("presentation-close")

    def install_signal(sig, handler):
        handlers[sig] = handler
        return

    def make_env(*args, **kwargs):
        if stop_at == "native-replaced":
            # An embedded native library replaces both dispositions during initialization.
            handlers.update({1: "native", 2: "native"})
        if stop_at.startswith("startup"):
            handlers[1](1, None)
        if stop_at.startswith("construction"):
            if stop_at == "construction-new":
                context_state["current"] = object()
            raise RuntimeError("environment construction failed")
        return SimpleNamespace(unwrapped=env)

    def replay(**kwargs):
        calls.append(
            ("replay", handlers[1] != "native" and handlers[2] != "native")
            if stop_at == "native-replaced"
            else "replay"
        )
        session.shutdown_requested = True
        return SimpleNamespace(outcome="success", to_dict=lambda index: {"outcome": "success"})

    env = SimpleNamespace(
        sim=SimpleNamespace(is_running=lambda: True),
        scene=SimpleNamespace(env_prim_paths=["/Lab/Cells/Cell0"]),
        close=lambda: calls.append("env-close"),
    )
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay_agent

    namespace = {
        "args_cli": SimpleNamespace(
            ovxr=True,
            device="cpu",
            enable_debug_visualization=False,
            require_kit=False,
            task="task",
            replay_file="fixture.mcap",
            num_replays=requested,
            stats_output_file=None,
        ),
        "OvxrPresentation": Presentation,
        "signal": SimpleNamespace(SIGINT=1, SIGTERM=2, signal=install_signal),
        "gym": SimpleNamespace(make=make_env),
        "_run_single_replay": replay,
        "create_isaac_teleop_device": lambda *_args, **_kwargs: SimpleNamespace(session_cleanup_failed=False),
        "_extract_env_perf_cfg": lambda cfg: None,
        "_print_stdout_summary": reports.append,
    }
    for name, value in namespace.items():
        monkeypatch.setattr(replay_agent, name, value, raising=False)
    if stop_at.startswith("startup"):
        with pytest.raises(KeyboardInterrupt):
            replay_agent._run_replay_batch(SimpleNamespace(isaac_teleop=None), None)
        assert calls == ["presentation-close"]
    elif stop_at.startswith("construction"):
        with pytest.raises(RuntimeError, match="environment construction failed"):
            replay_agent._run_replay_batch(SimpleNamespace(isaac_teleop=None), None)
        assert calls == ["presentation-close"] + (["context-clear"] if stop_at == "construction-new" else [])
        assert context_state["current"] is (None if stop_at == "construction-new" else previous)
    else:
        result = replay_agent._run_replay_batch(SimpleNamespace(isaac_teleop=None), None)
        assert result == (0 if requested == 1 else 1)
        assert reports[0]["num_replays"] == 1
        assert reports[0]["requested_num_replays"] == requested
        assert reports[0]["batch_complete"] is (requested == 1)
        # The shutdown handlers are active again before input starts; the environment closes after XR.
        assert calls == [
            ("replay", True) if stop_at == "native-replaced" else "replay",
            "presentation-close",
            "env-close",
        ]
    # Restoration returns the dispositions that preceded the batch.
    assert handlers == {1: None, 2: None}


@pytest.mark.parametrize("primary,drain_failed", [(False, False), (True, False), (True, True)])
def test_native_replay_cleanup_preserves_cause_and_borrowed_owners(monkeypatch, primary, drain_failed):
    """The public replay entry keeps CloudXR for a failed input drain and still closes the environment."""
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay
    from isaacteleop.teleop_session_manager import RetargetingExecutionConfig

    from isaaclab.app import launch_simulation
    from isaaclab.physics import PhysicsCfg
    from isaaclab.sim import SimulationContext

    original = RuntimeError("replay failed")
    cleanup = RuntimeError("environment close failed")
    previous, current = object(), object()
    state = {"context": previous}
    clear = Mock()
    runtime = SimpleNamespace(stop=Mock())
    device = SimpleNamespace(session_cleanup_failed=False)
    env = SimpleNamespace(
        scene=SimpleNamespace(env_prim_paths=["/Lab/Cells/Cell0"]),
        sim=SimpleNamespace(is_running=lambda: True),
        close=Mock(side_effect=cleanup),
    )
    cfg = SimpleNamespace(
        isaac_teleop=SimpleNamespace(retargeting_execution=RetargetingExecutionConfig(mode="pipelined"))
    )

    def make_env(*_args, **_kwargs):
        state["context"] = current
        return SimpleNamespace(unwrapped=env)

    def create_device(selected, **kwargs):
        assert selected.retargeting_execution.mode == "pipelined"
        assert kwargs["env_prim_path"] == "/Lab/Cells/Cell0"
        assert kwargs["cloudxr_env_file"] is None and not kwargs["auto_launch_cloudxr"]
        return device

    def run(**kwargs):
        assert kwargs["teleop_interface"] is device
        device.session_cleanup_failed = drain_failed
        if primary:
            raise original
        return SimpleNamespace(outcome="success")

    monkeypatch.setattr(SimulationContext, "instance", lambda: state["context"])
    monkeypatch.setattr(SimulationContext, "clear_instance", clear)
    monkeypatch.setattr(replay, "_prepare_env_cfg", lambda *_args, **_kwargs: (cfg, None))
    monkeypatch.setattr(replay, "hydra_overrides", [], raising=False)
    monkeypatch.setattr(
        replay,
        "args_cli",
        SimpleNamespace(
            num_replays=1,
            task="task",
            num_envs=1,
            device="cpu",
            ovxr=False,
            require_kit=False,
            cloudxr_env="standalone",
            auto_launch_cloudxr=True,
            replay_file="fixture.mcap",
            enable_debug_visualization=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(replay, "launch_simulation", lambda _cfg, _args: launch_simulation(PhysicsCfg()))
    monkeypatch.setattr(replay.gym, "make", make_env)
    monkeypatch.setattr(replay, "_maybe_launch_cloudxr", lambda *_args: runtime)
    monkeypatch.setattr(replay, "create_isaac_teleop_device", create_device)
    monkeypatch.setattr(replay, "_run_single_replay", run)
    with pytest.raises(RuntimeError) as caught:
        replay.main()
    assert caught.value is (original if primary else cleanup)
    env.close.assert_called_once_with()
    assert runtime.stop.call_count == int(not drain_failed)
    # The environment double leaves the context, which the launcher then releases.
    clear.assert_called_once_with()
    if primary:
        assert any(
            ("drain failed" if drain_failed else "environment close failed") in note for note in original.__notes__
        )


@pytest.mark.parametrize("start_on_replay", [False, True])
def test_replay_control_timing_uses_recorded_events_or_explicit_legacy_start(monkeypatch, start_on_replay):
    """The real control owners gate input, distinguish RESET from STOP, and retain the replay cursor."""
    import torch
    from isaaclab_teleop.control_events import ControlEvents
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay
    from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
    from isaaclab_teleop.isaac_teleop_device import IsaacTeleopDevice
    from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle
    from isaaclab_teleop.teleop_message_processor import TeleopMessageProcessor
    from isaacteleop.teleop_session_manager import DefaultTeleopStateManager

    processor = TeleopMessageProcessor(name="recorded_control")
    state_manager = DefaultTeleopStateManager(name="recorded_state")
    lifecycle = TeleopSessionLifecycle.__new__(TeleopSessionLifecycle)
    lifecycle._message_processor = processor
    lifecycle._last_control_events = ControlEvents()
    lifecycle._last_control_event_frame_id = None
    lifecycle._haptic_forces = {}
    lifecycle._session = SimpleNamespace(last_context=None, last_step_info=None)
    lifecycle._retargeter_reset_lock = threading.Lock()
    lifecycle._retargeter_reset_pending = False
    lifecycle._retargeter_reset_frame_id = None
    lifecycle._retargeter_reset_cutoff_ns = 0
    lifecycle._retargeter_last_submission_ns = 0
    messages = {7: b"reset", 12: b"stop"}
    if not start_on_replay:
        messages.update({3: b"start", 10: b"start"})
    frame = -1
    applied, resets, recorder_resets = [], [], []

    class Signal(list):
        is_none = False

    class RecordedDevice(IsaacTeleopDevice):
        def __init__(self):
            self._session_lifecycle = lifecycle
            self._anchor_manager = SimpleNamespace(reset=lambda: None, cleanup=lambda: None)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def advance(self):
            nonlocal frame
            frame += 1
            payload = messages.get(frame)
            tracked = SimpleNamespace(data=[SimpleNamespace(payload=payload)] if payload is not None else [])
            outputs = {name: Signal([False]) for name in ("kill", "run_toggle", "reset")}
            processor._compute_fn({processor.INPUT_MESSAGES: [tracked]}, outputs, context=None)
            events = state_manager._compute_execution_events(
                {
                    state_manager.INPUT_KILL: outputs["kill"],
                    state_manager.INPUT_RUN_TOGGLE: outputs["run_toggle"],
                    state_manager.INPUT_RESET: outputs["reset"],
                },
                context=None,
            )
            lifecycle._session.last_context = SimpleNamespace(execution_events=events)
            lifecycle._session.last_step_info = SimpleNamespace(returned_frame_id=frame, submitted_frame_id=frame)
            lifecycle._update_control_events()
            return torch.zeros(1)

    env = SimpleNamespace(
        num_envs=1,
        scene=SimpleNamespace(physics_backend="newton_mjwarp", env_prim_paths=["/World/envs/env_0"]),
        sim=SimpleNamespace(device="cpu", render=lambda: None, is_running=lambda: frame < 15),
        recorder_manager=SimpleNamespace(reset=lambda: recorder_resets.append(frame)),
        reset=lambda: resets.append(frame),
    )

    def step(_actions):
        applied.append(frame)
        env.sim.render()
        return None, None, torch.tensor([False]), torch.tensor([False]), {}

    env.step = step
    monkeypatch.setattr(
        replay,
        "NvmlGpuStatsProvider",
        lambda: SimpleNamespace(sample=lambda: None, summary=lambda: {}),
    )
    monkeypatch.setattr(
        replay,
        "args_cli",
        SimpleNamespace(
            device="cpu",
            replay_file="recording.mcap",
            start_on_replay=start_on_replay,
            enable_debug_visualization=False,
            max_replay_duration_s=10.0,
            num_success_steps=1,
        ),
        raising=False,
    )
    cfg = IsaacTeleopCfg()
    stats = replay._run_single_replay(env, cfg, None, run_index=1, total_runs=2, teleop_interface=RecordedDevice())
    assert applied == ([2, 3, 4, 5, 6, 9, 10, 11] if start_on_replay else [5, 6, 10, 11])
    assert resets == [-1, 0, 7]
    assert recorder_resets == [0, 7]
    assert frame == 12  # STOP terminates; RESET retains the cursor and permits the later START.
    # The control channel already consumed the recorded RESET; the notification still resets retargeters.
    assert lifecycle._retargeter_reset_pending
    assert lifecycle._retargeter_reset_frame_id == 7
    assert stats.active_iterations == len(applied)
    assert stats.outcome == "incomplete"


@pytest.mark.parametrize("kit_active", [False, True])
def test_unrendered_replay_reports_its_outcome_only_without_kit(monkeypatch, kit_active):
    """A kitless replay has no application render loop to time; under Kit an unmeasured run still fails."""
    import torch
    from isaaclab_teleop import IsaacTeleopCfg
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay

    class Device:
        last_control_events = SimpleNamespace(is_active=True, should_reset=False)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def reset(self):
            pass

        def advance(self):
            return torch.zeros(1)

    steps = []

    def step(_actions):
        steps.append(len(steps))
        return None, None, torch.tensor([False]), torch.tensor([False]), {}

    env = SimpleNamespace(
        num_envs=1,
        scene=SimpleNamespace(physics_backend="ovphysx", env_prim_paths=["/World/envs/env_0"]),
        sim=SimpleNamespace(device="cpu", render=lambda: None, is_running=lambda: len(steps) < 3),
        reset=lambda: None,
        step=step,
    )
    monkeypatch.setattr(replay, "has_kit", lambda: kit_active)
    monkeypatch.setattr(
        replay, "NvmlGpuStatsProvider", lambda: SimpleNamespace(sample=lambda: None, summary=lambda: {})
    )
    monkeypatch.setattr(
        replay,
        "args_cli",
        SimpleNamespace(
            device="cpu",
            replay_file="recording.mcap",
            start_on_replay=False,
            enable_debug_visualization=False,
            max_replay_duration_s=10.0,
            num_success_steps=1,
        ),
        raising=False,
    )

    def run():
        return replay._run_single_replay(
            env, IsaacTeleopCfg(), None, run_index=1, total_runs=2, teleop_interface=Device()
        )

    if kit_active:
        with pytest.raises(RuntimeError, match="no usable frame intervals"):
            run()
    else:
        stats = run()
        assert stats.active_iterations == 3
        assert not stats.active_frame_times_ms


def test_replay_renderer_startup_shares_the_cold_start_bound():
    """Replay waits for renderer attachment as long as live teleoperation and recording do."""
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay
    from isaaclab_teleop.entrypoints._ovxr_runtime import RENDERER_ATTACH_TIMEOUT_S

    waits = []
    presentation = SimpleNamespace(
        start=lambda _env, *, timeout_s: waits.append(timeout_s) or True,
        publish=lambda _env, *, reset: None,
    )
    replay._start_replay_presentation("env", presentation)
    assert waits == [RENDERER_ATTACH_TIMEOUT_S]
