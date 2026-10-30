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
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def replay_presentation_type():
    from isaaclab_teleop.entrypoints.teleop_replay_agent import _ReplayPresentation

    return _ReplayPresentation


def test_replay_failed_input_drain_requests_stop_without_releasing_borrowers(replay_presentation_type):
    presentation = replay_presentation_type.__new__(replay_presentation_type)
    presentation.device = SimpleNamespace(session_cleanup_failed=True)
    presentation.session = Mock()
    with pytest.raises(RuntimeError, match="Replay input did not drain"):
        presentation.close()
    presentation.session.request_shutdown.assert_called_once_with()
    presentation.session.close.assert_not_called()


@pytest.mark.parametrize("layout_mode", ["horizontal", "manual", None])
def test_replay_presentation_preserves_camera_layout(monkeypatch, replay_presentation_type, layout_mode):
    """The actual viewer constructor changes placement without discarding task packing."""
    import isaaclab_teleop
    from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg, XrCameraFeedCfg, XrCameraFeedLayoutCfg

    from isaaclab.sensors import CameraCfg

    names = ["left", "right"]
    config = SimpleNamespace(
        scene=SimpleNamespace(**{name: CameraCfg(prim_path=f"/World/{name}", width=640, height=480) for name in names}),
        sim=object(),
        isaac_teleop=IsaacTeleopCfg(
            xr_camera_feeds=[XrCameraFeedCfg(camera_name=name, panel_width_m=0.4) for name in names],
            xr_camera_feed_layout=(
                XrCameraFeedLayoutCfg(
                    mode=layout_mode,
                    placement="viewer_start",
                    center_offset_m=(0.1, -0.2),
                    panel_gap_m=0.1,
                    distance_m=1.2,
                )
                if layout_mode is not None
                else None
            ),
        ),
    )
    original_layout = config.isaac_teleop.xr_camera_feed_layout
    original_layout_fields = original_layout.to_dict() if original_layout is not None else None
    original_feeds = [feed.to_dict() for feed in config.isaac_teleop.xr_camera_feeds]
    monkeypatch.setitem(
        replay_presentation_type.__init__.__globals__,
        "args_cli",
        SimpleNamespace(
            ovxr_pip_camera=names,
            task="Test",
            ovxr_session_mode="ar",
            cloudxr_env="none",
            auto_launch_cloudxr=False,
            ovxr_desktop_mirror=False,
        ),
    )
    prepare = Mock()
    monkeypatch.setattr(isaaclab_teleop.OvLibrariesXrSession, "prepare", prepare)
    presentation = replay_presentation_type(config)
    assert presentation.session is prepare.return_value
    session_config = prepare.call_args.args[1]
    expected = (
        [(-0.15, -0.2, -1.2), (0.35, -0.2, -1.2)]
        if layout_mode == "horizontal"
        else [(0.0, 0.0, -0.8), (0.0, 0.0, -0.8)]
    )
    for camera, position in zip(session_config.auxiliary_cameras, expected, strict=True):
        assert camera.position == pytest.approx(position)
    assert config.isaac_teleop.xr_camera_feed_layout is original_layout
    if original_layout is not None:
        assert original_layout.to_dict() == original_layout_fields
    assert [feed.to_dict() for feed in config.isaac_teleop.xr_camera_feeds] == original_feeds


@pytest.mark.parametrize(
    "kit_active,run_index,backend,viewer,hard_reset",
    [
        (False, 0, "ovphysx", False, True),
        (True, 0, "ovphysx", False, True),
        (True, 1, "ovphysx", False, True),
        (False, 0, "ovphysx", True, True),
        (False, 0, "newton_mjwarp", True, False),
    ],
)
def test_replay_bootstrap_waits_only_for_an_active_kit_application(
    monkeypatch, kit_active, run_index, backend, viewer, hard_reset
):
    """A viewer preserves backend reset policy; kitless replay never waits for Kit assets."""
    calls = []

    class Device:
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
    wait_module = ModuleType("isaaclab_physx.renderers.isaac_rtx_renderer_utils")
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
            device="cpu", render=render, reset=lambda: calls.append("sim-reset"), is_running=lambda: False
        ),
        reset=lambda: calls.append("env-reset"),
    )
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay
    from isaacteleop.teleop_session_manager import RetargetingExecutionConfig

    cfg = SimpleNamespace(retargeting_execution=RetargetingExecutionConfig(mode="pipelined"))
    constructed_modes = []

    def create_device(selected, **kwargs):
        assert kwargs["env_prim_path"] == "/Lab/Cells/Cell0"
        constructed_modes.append(selected.retargeting_execution.mode)
        assert selected.retargeting_execution.mode == "pipelined"
        return Device()

    namespace = {
        "args_cli": SimpleNamespace(
            replay_file="fixture.mcap",
            enable_debug_visualization=False,
            max_stage_load_wait_s=5,
            replay_start_delay_s=0,
            start_on_replay=False,
            max_replay_duration_s=1,
        ),
        "NvmlGpuStatsProvider": lambda: SimpleNamespace(_available=True, summary=lambda: {}),
        "create_isaac_teleop_device": create_device,
        "_RENDERER_SETTLE_FRAMES": 30,
    }
    for name, value in namespace.items():
        monkeypatch.setattr(replay, name, value, raising=False)
    presentation = SimpleNamespace(start=lambda *_args: calls.append("viewer-start")) if viewer else None
    replay._run_single_replay(env, cfg, None, run_index, 2, presentation)
    expected = ["device-enter"]
    if hard_reset:
        expected.append("sim-reset")
    expected.extend(["env-reset", "device-reset"])
    if viewer:
        expected.append("viewer-start")
    if kit_active and run_index == 0:
        expected.append("kit-wait")
    assert calls == [*expected, "device-exit"]
    assert env.sim.render is render
    assert constructed_modes == ["pipelined"]


@pytest.mark.parametrize(
    "stop_at,requested",
    [
        ("startup", 2),
        ("startup-second-signal", 2),
        ("construction-new", 2),
        ("construction-previous", 2),
        ("after-first", 2),
        ("after-first", 1),
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
    session = SimpleNamespace(
        shutdown_requested=False,
        async_worker_owns_environment=not stop_at.startswith("construction"),
        request_shutdown=lambda: None,
    )

    class Presentation:
        def __init__(self, cfg):
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
        if stop_at.startswith("startup"):
            handlers[1](1, None)
        if stop_at.startswith("construction"):
            if stop_at == "construction-new":
                context_state["current"] = object()
            raise RuntimeError("environment construction failed")
        return SimpleNamespace(unwrapped=env)

    def replay(**kwargs):
        calls.append("replay")
        session.shutdown_requested = True
        return SimpleNamespace(outcome="success", to_dict=lambda index: {"outcome": "success"})

    env = SimpleNamespace(sim=SimpleNamespace(is_running=lambda: True))
    from isaaclab_teleop.entrypoints import teleop_replay_agent as replay_agent

    namespace = {
        "args_cli": SimpleNamespace(
            ovxr=True,
            require_kit=False,
            task="task",
            replay_file="fixture.mcap",
            num_replays=requested,
            stats_output_file=None,
        ),
        "_ReplayPresentation": Presentation,
        "signal": SimpleNamespace(SIGINT=1, SIGTERM=2, signal=install_signal),
        "gym": SimpleNamespace(make=make_env),
        "_run_single_replay": replay,
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
        assert calls == ["replay", "presentation-close"]
    assert handlers == {1: None, 2: None}


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
            lifecycle._session.last_step_info = SimpleNamespace(returned_frame_id=frame)
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
    monkeypatch.setattr(replay, "create_isaac_teleop_device", lambda *_args, **_kwargs: RecordedDevice())
    monkeypatch.setattr(
        replay,
        "NvmlGpuStatsProvider",
        lambda: SimpleNamespace(sample=lambda: None, summary=lambda: {}),
    )
    monkeypatch.setattr(
        replay,
        "args_cli",
        SimpleNamespace(
            replay_file="recording.mcap",
            start_on_replay=start_on_replay,
            enable_debug_visualization=False,
            max_replay_duration_s=10.0,
            num_success_steps=1,
        ),
        raising=False,
    )
    conditioning_frames = []

    def condition(_env, action, state):
        previous = state.get("frames", 0)
        conditioning_frames.append(previous)
        state["frames"] = previous + 1
        return action

    cfg = IsaacTeleopCfg(action_conditioner=condition)
    stats = replay._run_single_replay(env, cfg, None, run_index=1, total_runs=2)
    assert applied == ([2, 3, 4, 5, 6, 9, 10, 11] if start_on_replay else [5, 6, 10, 11])
    assert resets == [-1, 0, 7]
    assert recorder_resets == [0, 7]
    assert frame == 12  # STOP terminates; RESET retains the cursor and permits the later START.
    assert conditioning_frames == ([0, 1, 2, 3, 4, 0, 1, 2] if start_on_replay else [0, 1, 0, 1])
    assert stats.active_iterations == len(applied)
    assert stats.outcome == "incomplete"
