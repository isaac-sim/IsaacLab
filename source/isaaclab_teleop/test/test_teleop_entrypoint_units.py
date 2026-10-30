# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-process teleoperation entrypoint, configuration, and control-loop contracts."""

import ast
import os
import re
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Repository root: this file lives at source/isaaclab_teleop/test/<file>.py
ROOT = Path(__file__).resolve().parents[3]

# XR experience file whose dependencies decide whether ``--xr`` can start offline.
_XR_EXPERIENCE = ROOT / "apps" / "isaaclab.python.xr.openxr.kit"

# ``.kit`` files repeat section headers, so they cannot be parsed as TOML.
_SECTION_RE = re.compile(r"^\s*\[+(?P<name>[^\[\]]+)\]+\s*$")
_DEPENDENCY_RE = re.compile(r'^\s*"(?P<name>[^"]+)"\s*=')


def _kit_dependencies(experience: Path) -> set[str]:
    """Collect the extension names declared in an experience file's dependency sections.

    Scoped to ``[dependencies]`` tables so quoted keys elsewhere in the file (settings tables
    use them too) are not mistaken for extensions.
    """
    dependencies: set[str] = set()
    in_dependencies = False

    for line in experience.read_text(encoding="utf-8").splitlines():
        section = _SECTION_RE.match(line)
        if section is not None:
            in_dependencies = section.group("name").strip() == "dependencies"
            continue
        if in_dependencies:
            dependency = _DEPENDENCY_RE.match(line)
            if dependency is not None:
                dependencies.add(dependency.group("name"))

    return dependencies


def test_xr_experience_declares_only_shipped_extensions():
    """The XR experience depends on XR extensions Isaac Sim ships, not on a registry-only bundle.

    ``omni.kit.xr.bundle.generic`` is not shipped and resolves only from the Kit SDK registry, so
    depending on it aborted every ``--xr`` run that could not reach the registry.

    Static on purpose: a machine that reaches the registry caches the bundle on first use and
    loads it locally forever after, so a launch test cannot catch this once that cache is warm.
    """
    dependencies = _kit_dependencies(_XR_EXPERIENCE)

    assert "omni.kit.xr.system.openxr" in dependencies
    assert "omni.kit.xr.ui.window.profile" in dependencies, "omni.kit.xr.profile.ar's Kit 110 successor is missing"
    assert not [name for name in dependencies if name.startswith("omni.kit.xr.bundle.")], (
        "XR bundle meta-extensions are not shipped with Isaac Sim and resolve only from the Kit SDK registry"
    )


def test_teleop_readme_programmatic_config_returns_pipeline_not_factory_tuple():
    """Execute the documented public config, without entering an XR session."""
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner

    readme = ROOT / "source/isaaclab_teleop/docs/README.md"
    section = readme.read_text().split("### 4. Programmatic Usage", 1)[1]
    example = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert example is not None
    tree = ast.parse(example.group(1))
    config_statements = []
    for statement in tree.body:
        if isinstance(statement, ast.With):
            break
        config_statements.append(statement)
    source = ControllersSource(name="documentation_controllers")
    pipeline = OutputCombiner({"action": source.output(ControllersSource.RIGHT)})
    retargeters = []
    namespace = {"my_pipeline_builder": lambda: (pipeline, retargeters)}
    exec(compile(ast.Module(body=config_statements, type_ignores=[]), str(readme), "exec"), namespace)
    cfg = namespace["cfg"]
    assert cfg.pipeline_builder() is pipeline
    assert cfg.pipeline_builder().output("action") is not None
    assert cfg.retargeters_to_tune() is retargeters


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

    import isaaclab_teleop
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
    monkeypatch.setattr(isaaclab_teleop, "remove_camera_configs", lambda cfg, **_kwargs: cfg)
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

    import isaaclab_teleop
    from isaaclab_teleop.entrypoints import teleop_se3_agent as agent

    def run(env, interface, session, *, conditioner=None, validator=None, step_hz=0.0, refresh=lambda: None):
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
        session.start_render_worker = lambda *_args, **_kwargs: True
        session.close = lambda: None
        session.async_worker_owns_environment = False
        env.close = lambda: None
        env.scene = SimpleNamespace(env_prim_paths=["/World/envs/env_0"])
        env.sim.device = "cpu"
        env.sim.visualizers = ()
        cfg = SimpleNamespace(
            sim=object(),
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
        monkeypatch.setattr(agent, "_make_haptic_io", lambda *_args: (lambda: None, lambda: None))
        monkeypatch.setattr(agent, "_install_ovxr_shutdown_handlers", lambda *_args: {})
        monkeypatch.setattr(isaaclab_teleop.XrCameraFeedSession, "prepare", lambda *_args, **_kwargs: camera)
        monkeypatch.setattr(isaaclab_teleop.OvLibrariesXrSession, "prepare", lambda *_args, **_kwargs: session)
        agent.run_teleoperation(cfg, cfg.isaac_teleop, True)

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
    assert requests[1:] == [(1, False), (1, False)]
    assert sleeps == [(0.02, 3)]  # Pacing follows the retried publication.


@pytest.mark.parametrize("selector", [["--physics", "ovphysx"], ["physics=ovphysx"]])
@pytest.mark.parametrize("ovxr", [False, True])
def test_teleop_replay_agent_resolves_typed_presets_without_reapplying_physics(monkeypatch, selector, ovxr):
    """Public CLI aliases reach task resolution without resetting backend settings."""
    import isaaclab_teleop
    from isaaclab_ov.physics import OvPhysxCfg

    from isaaclab.managers import TerminationTermCfg
    from isaaclab.sim import SimulationCfg

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
    monkeypatch.setattr(replay, "resolve_task_config", resolve)
    monkeypatch.setattr(isaaclab_teleop, "remove_camera_configs", lambda cfg, **_kwargs: cfg)
    monkeypatch.setattr(replay, "remove_camera_configs", isaaclab_teleop.remove_camera_configs)
    prepared, success = namespace["_prepare_env_cfg"]("test-task", 1, None, overrides=namespace["hydra_overrides"])
    assert resolutions == [("test-task", "", ["physics=ovphysx"])]
    assert prepared is cfg and success is not None
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
    [("ovphysx", False, True), ("ovphysx", True, True), ("newton_mjwarp", True, False)],
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
            start=lambda *_args: None,
            publish=lambda *_args, **kwargs: calls.append("publish-reset") if kwargs.get("reset") else None,
        )
        if viewer
        else None
    )
    monkeypatch.setattr(
        replay,
        "args_cli",
        SimpleNamespace(
            replay_file="recording.mcap",
            start_on_replay=True,
            enable_debug_visualization=False,
            max_replay_duration_s=1,
        ),
        raising=False,
    )
    monkeypatch.setattr(replay, "create_isaac_teleop_device", lambda *_args, **_kwargs: Device())
    monkeypatch.setattr(replay, "NvmlGpuStatsProvider", lambda: SimpleNamespace(summary=lambda: {}))
    replay._run_single_replay(env, SimpleNamespace(), None, 1, 2, presentation)
    initial = ["sim-reset"] if hard_reset else []
    initial += ["env-reset", "device-reset", "start"]
    recorded = ["sim-reset"] if hard_reset else []
    recorded += ["recorder-reset", "env-reset", "device-reset", "start"]
    if viewer:
        recorded.append("publish-reset")
    assert calls == initial + recorded
    assert frame == 1


def test_teleop_replay_reset_policy_skips_hard_reset_only_for_newton():
    """Newton keeps its constructed solver while OVPhysX retains the PR-5507 hard reset."""
    from isaaclab_teleop.entrypoints.teleop_replay_agent import _replay_requires_hard_sim_reset as requires_hard_reset

    assert requires_hard_reset("newton_mjwarp") is False
    assert requires_hard_reset("newtonmanager") is False
    assert requires_hard_reset("ovphysx") is True
    assert requires_hard_reset(None) is True


def test_teleop_replay_reset_sequence_preserves_backend_owned_order():
    """The backend policy changes only the optional simulator reset in the reset sequence."""
    from isaaclab_teleop.entrypoints.teleop_replay_agent import _reset_replay_environment as reset_environment

    calls: list[str] = []

    class _Resetter:
        def __init__(self, name: str):
            self._name = name

        def reset(self):
            calls.append(self._name)

    class _Environment:
        sim = _Resetter("sim")
        recorder_manager = _Resetter("recorder")

        def reset(self):
            calls.append("env")

    env = _Environment()
    reset_environment(env, hard_sim_reset=False, reset_recorder=True)
    assert calls == ["recorder", "env"]

    calls.clear()
    reset_environment(env, hard_sim_reset=True, reset_recorder=True)
    assert calls == ["sim", "recorder", "env"]

    calls.clear()
    reset_environment(env, hard_sim_reset=False, reset_recorder=False)
    assert calls == ["env"]


def _load_teleop_agent(monkeypatch, *arguments):
    from isaaclab_teleop.entrypoints import teleop_se3_agent as module

    monkeypatch.setattr(sys, "argv", ["teleop_se3_agent.py", *arguments])
    monkeypatch.setattr(module, "args_cli", None, raising=False)
    module._parse_args()
    return module


def test_product_xr_default_resolves_cloudxrjs_and_autolaunch(monkeypatch):
    agent = _load_teleop_agent(monkeypatch, "--ovxr")
    from isaaclab_teleop import CLOUDXR_JS_ENV

    assert agent.args_cli.cloudxr_env == CLOUDXR_JS_ENV
    assert agent.args_cli.auto_launch_cloudxr is True


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
def test_teleop_launcher_preserves_the_resolved_physics_configuration(monkeypatch, task_name, physics):
    """Task preparation must reach launch validation without losing its backend settings."""
    from isaaclab.app.sim_launcher import validate_launch

    from isaaclab_tasks.utils import parse_env_cfg

    agent = _load_teleop_agent(monkeypatch, "--task", task_name, "--physics", physics)
    env_cfg = parse_env_cfg(task_name, device="cpu", num_envs=1, overrides=[f"physics={physics}"])
    physics_cfg = env_cfg.sim.physics
    env_cfg.isaac_teleop.prepare_environment(env_cfg)
    if physics == "newton_mjwarp":
        if "Locomanipulation-G1" in task_name:
            assert physics_cfg.num_substeps == 2
            assert physics_cfg.solver_cfg.integrator == "implicitfast"
        else:
            assert physics_cfg.solver_cfg.nconmax == 4096
            assert physics_cfg.solver_cfg.njmax == 1536
    else:
        from isaaclab_tasks.contrib.packing_table import PackingTableUsdFileCfg

        assert isinstance(env_cfg.scene.packing_table.spawn, PackingTableUsdFileCfg) == (physics == "ovphysx")

    configs, _ = validate_launch(SimpleNamespace(sim=SimpleNamespace(physics=physics_cfg)), vars(agent.args_cli).copy())
    assert not hasattr(agent.args_cli, "physics")
    assert env_cfg.sim.physics is physics_cfg
    assert any(cfg is physics_cfg for cfg in configs)


def test_task_action_hook_validates_before_and_after_conditioning(monkeypatch):
    agent = _load_teleop_agent(monkeypatch)
    calls = []

    def validate(action):
        calls.append(("validate", action))
        return "rejected" if action == 3 else None

    def condition(env, action, state):
        calls.append(("condition", env, action, state))
        state["seen"] = True
        return action + 1

    state = {}
    action, issue = agent.prepare_teleop_action("env", 1, condition, validate, state)
    assert (action, issue, state) == (2, None, {"seen": True})
    assert [call[0] for call in calls] == ["validate", "condition", "validate"]

    calls.clear()
    action, issue = agent.prepare_teleop_action("env", 3, condition, validate, state)
    assert (action, issue) == (3, "rejected")
    assert [call[0] for call in calls] == ["validate"]


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
    previous = agent._install_ovxr_shutdown_handlers(Session())

    assert set(installed) == {signal.SIGINT, signal.SIGTERM}
    installed[signal.SIGINT](signal.SIGINT, None)
    assert requests == ["shutdown"]

    restored = {}
    monkeypatch.setattr(agent.signal, "signal", lambda signum, handler: restored.update({signum: handler}))
    agent._restore_ovxr_shutdown_handlers(previous)
    assert restored == previous


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

    assert agent._required_hand_tracking_issue(interface, initialized) == expected
    lifecycle.last_step_result = {"hand_left": OptionalHand(True), "hand_right": OptionalHand(True)}
    # Numeric zero is not a sentinel: a present, valid fully-open hand frame is accepted.
    assert agent._required_hand_tracking_issue(interface, initialized) is None
    assert initialized == {"left", "right"}

    lifecycle.last_step_result = {"hand_left": OptionalHand(False), "hand_right": OptionalHand(False)}
    assert agent._required_hand_tracking_issue(interface, initialized) is None

    initialized.clear()
    assert agent._required_hand_tracking_issue(interface, initialized) == (
        "waiting for initial left/right hand tracking"
    )


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

    def update():
        haptics["update"] += 1

    def stop():
        haptics["stop"] += 1

    valid = torch.arange(36, dtype=torch.float32)

    assert agent._apply_teleop_frame(env, valid, None, update, stop)
    assert not agent._apply_teleop_frame(env, torch.zeros_like(valid), "missing left hand tracking", update, stop)
    assert agent._apply_teleop_frame(env, valid, None, update, stop)

    assert env.sim.render_count == 1
    assert len(env.actions) == 2
    torch.testing.assert_close(env.actions[0], env.actions[1])
    assert haptics == {"update": 2, "stop": 1}


@pytest.mark.parametrize("entrypoint", ["live", "replay"])
def test_task_preparation_precedes_ovxr_simulation_context(monkeypatch, entrypoint):
    """Both actual workflows finalize the selected config before OVXR allocates its context."""
    from contextlib import contextmanager

    import isaaclab_teleop
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_teleop import ovxr_session
    from isaaclab_teleop.entrypoints import teleop_replay_agent, teleop_se3_agent

    from isaaclab.sim import SimulationCfg, SimulationContext

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
    monkeypatch.setattr(module, "resolve_task_config", lambda *_args, **_kwargs: (config, None))
    monkeypatch.setattr(module, "validate_launch", validate_launch)
    monkeypatch.setattr(module, "launch_simulation", launch)
    monkeypatch.setattr(isaaclab_teleop, "remove_camera_configs", lambda cfg, **_kwargs: cfg)
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
        monkeypatch.setattr(module, "remove_camera_configs", isaaclab_teleop.remove_camera_configs)
        arguments += ["--replay_file", "recording.mcap"]
    module._parse_args(arguments)

    with pytest.raises(ContextReached):
        module.main()
    assert calls == ["prepare", "launch", "context"]
