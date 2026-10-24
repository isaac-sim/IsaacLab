# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute the real replay bootstrap without starting a simulator.

Only device/simulator endpoints are isolated; the replay/reset function bodies
and Kit detection are production code. These tests do not qualify MCAP motion.
"""

import __future__

import ast
import builtins
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from isaaclab.utils.version import has_kit

_ROOT = Path(__file__).resolve().parents[3]


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
        scene=SimpleNamespace(physics_backend=backend),
        sim=SimpleNamespace(
            device="cpu", render=render, reset=lambda: calls.append("sim-reset"), is_running=lambda: False
        ),
        reset=lambda: calls.append("env-reset"),
    )
    source_path = _ROOT / "scripts/environments/teleoperation/teleop_replay_agent.py"
    names = {
        "_run_single_replay",
        "_warmup_replay",
        "_advance_replay_input",
        "_render_replay_idle",
        "_replay_requires_hard_sim_reset",
        "_reset_replay_environment",
        "_assert_run_measured",
    }
    selected = [
        node
        for node in ast.parse(source_path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in selected} == names
    from isaacteleop.teleop_session_manager import RetargetingExecutionConfig

    cfg = SimpleNamespace(retargeting_execution=RetargetingExecutionConfig(mode="pipelined"))
    constructed_modes = []

    def create_device(selected, **kwargs):
        constructed_modes.append(selected.retargeting_execution.mode)
        assert selected.retargeting_execution.mode == "pipelined"
        return Device()

    namespace = {
        "args_cli": SimpleNamespace(
            replay_file="fixture.mcap",
            enable_debug_visualization=False,
            max_stage_load_wait_s=5,
            replay_start_delay_s=0,
            max_replay_duration_s=1,
        ),
        "_RunStats": lambda: SimpleNamespace(active_iterations=0, active_frame_times_ms=[], active_render_calls=0),
        "NvmlGpuStatsProvider": lambda: SimpleNamespace(_available=True, summary=lambda: {}),
        "create_isaac_teleop_device": create_device,
        "_RENDERER_SETTLE_FRAMES": 30,
        "contextmanager": contextmanager,
        "has_kit": has_kit,
        "time": time,
    }
    # Avoid executing the script's CLI/application startup during this unit test.
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(source_path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    presentation = SimpleNamespace(start=lambda *_args: calls.append("viewer-start")) if viewer else None
    namespace["_run_single_replay"](env, cfg, None, run_index, 2, presentation)
    expected = ["device-enter"]
    if hard_reset:
        expected.append("sim-reset")
    expected.extend(["env-reset", "device-reset", "device-start"])
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
    source_path = _ROOT / "scripts/environments/teleoperation/teleop_replay_agent.py"
    names = {"_run_replay_batch", "_exit_code_for_outcomes", "_build_report", "_aggregate_runs"}
    selected = [
        node
        for node in ast.parse(source_path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
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
        "_STATS_SCHEMA_VERSION": 1,
        "statistics": statistics,
        "_extract_env_perf_cfg": lambda cfg: None,
        "_print_stdout_summary": reports.append,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(source_path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    if stop_at.startswith("startup"):
        with pytest.raises(KeyboardInterrupt):
            namespace["_run_replay_batch"](SimpleNamespace(isaac_teleop=None), None)
        assert calls == ["presentation-close"]
    elif stop_at.startswith("construction"):
        with pytest.raises(RuntimeError, match="environment construction failed"):
            namespace["_run_replay_batch"](SimpleNamespace(isaac_teleop=None), None)
        assert calls == ["presentation-close"] + (["context-clear"] if stop_at == "construction-new" else [])
        assert context_state["current"] is (None if stop_at == "construction-new" else previous)
    else:
        result = namespace["_run_replay_batch"](SimpleNamespace(isaac_teleop=None), None)
        assert result == (0 if requested == 1 else 1)
        assert reports[0]["num_replays"] == 1
        assert reports[0]["requested_num_replays"] == requested
        assert reports[0]["batch_complete"] is (requested == 1)
        assert calls == ["replay", "presentation-close"]
    assert handlers == {1: None, 2: None}
