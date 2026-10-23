# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch smoke tests for the three teleoperation entry-point scripts.

Each test launches a teleop script end-to-end (spawning Isaac Sim / Kit as a subprocess) and
asserts it starts up past the regression introduced by the render-flag cleanup in #6656:

* ``apply_isaac_rtx_global_settings`` used to crash with
  ``ModuleNotFoundError: No module named 'omni.replicator'`` when the selected Kit experience
  did not preload the replicator extension, and
* ``record_demos.py`` read the removed ``args_cli.enable_cameras`` attribute, raising
  ``AttributeError``.

They also cover the XR startup regression, where a dependency on the unshipped
``omni.kit.xr.bundle.generic`` aborted ``--xr`` wherever the Kit SDK registry was unreachable.

These follow the Isaac Lab subprocess-integration test convention (see
``scripts/benchmarks/test``) rather than the isaacsim-CI methodology, so they run in the Isaac
Lab CI ``integration`` lane. They live under the ``isaaclab_teleop`` package tests so the CI
teleop filter picks them up, but exercise the entry-point scripts under ``scripts/``. The teleop
scripts run an unbounded interaction loop, so each is launched under ``PYTHONUNBUFFERED=1`` (the
startup prints are otherwise buffered), watched until a known startup marker appears, then
terminated via its process group.
"""

import ast
import importlib.util
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from isaaclab_teleop.action_policy import prepare_teleop_action, prepare_teleop_backend

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


# Lightweight Franka task that configures an Isaac Capture pipeline (controllers), so omitting
# ``--teleop_device`` drives the scripts through the Isaac Capture stack rather than the legacy
# native devices.
_ISAAC_TELEOP_TASK = "IsaacContrib-Stack-Cube-Franka-IK-Abs"

# Exact regression signatures that must never appear in a script's output.
_REGRESSION_SIGNATURES = (
    # #6656 render-flag cleanup fallout
    "No module named 'omni.replicator'",
    "has no attribute 'enable_cameras'",
    "has no attribute 'headless'",
    # Success termination removed while a reward still references it
    "Not all regular expressions are matched",
    # Registry-only XR dependency: the app exits before any marker is printed
    "Exiting app because of dependency solver failure",
)

# Logged by ``isaaclab_teleop`` when it builds the Isaac Capture device + retargeting pipeline.
# This runs *after* the RTX-apply site that regressed, and before the OpenXR/CloudXR session is
# entered -- which needs a headset/CloudXR runtime (``NV_CXR_RUNTIME_DIR``) not present in headless
# CI. Reaching it proves the RTX/camera startup and the Isaac Capture factory both succeed.
_ISAAC_TELEOP_MARKER = "Using Isaac Capture stack for teleoperation"

# Marker logged by ManagerBasedEnv creation, which runs *after* the RTX-apply site that regressed.
_ENV_CREATED_MARKER = "Base environment:"

# Marker logged after every manager has initialized successfully.
_ENV_READY_MARKER = "Completed setting up the environment"

# Disable CloudXR: no headset connects in CI, and auto-launching the runtime is unnecessary to
# exercise the Isaac Capture device factory (the live session is out of scope for a headless smoke).
_NO_CLOUDXR = ["--cloudxr_env", "none", "--no-auto_launch_cloudxr"]

# Generous ceiling for a cold Isaac Sim launch + Franka env creation on CI hardware. On a warm
# shader/asset cache the marker appears in ~25s, but the first launch on a cold runner can take
# ~500s+ while caches populate, so the ceiling is sized well above that to avoid spurious timeouts.
# The watcher returns as soon as the marker is seen, so this costs nothing on warm runs.
_STARTUP_TIMEOUT_S = 900.0


def _launch_until_marker(argv: list[str], markers: list[str], log_path: Path) -> str:
    """Launch a teleop script and return its output once a marker appears (or it exits / times out).

    The child is started in its own process group and unbuffered so both ``print`` startup markers
    and ``[INFO]`` logs surface promptly; the whole group is terminated before returning.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, str(ROOT / argv[0]), *argv[1:]]
    # Keep the write handle open for the child's whole lifetime (the poll loop reads progress back
    # through a separate read handle); it is closed only when this ``with`` block exits after teardown.
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            deadline = time.time() + _STARTUP_TIMEOUT_S
            while time.time() < deadline:
                output = log_path.read_text(errors="replace")
                if any(marker in output for marker in markers):
                    break
                if any(sig in output for sig in _REGRESSION_SIGNATURES):
                    break
                if proc.poll() is not None:
                    break
                time.sleep(2.0)
        finally:
            _terminate_group(proc)
    return log_path.read_text(errors="replace")


def _terminate_group(proc: subprocess.Popen) -> None:
    """Terminate the child's whole process group (Isaac Sim spawns helper processes)."""
    if proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            continue


def _assert_started_cleanly(output: str, markers: list[str], script: str) -> None:
    """Assert the run cleared known regressions and reached a post-startup marker."""
    tail = output[-4000:]
    for sig in _REGRESSION_SIGNATURES:
        assert sig not in output, f"{script}: regression signature present ({sig!r})\n---\n{tail}"
    assert any(marker in output for marker in markers), (
        f"{script}: did not reach a startup marker {markers} within {_STARTUP_TIMEOUT_S:.0f}s\n---\n{tail}"
    )


@pytest.mark.integration
@pytest.mark.rendering
def test_teleop_se3_agent_starts(tmp_path):
    """teleop_se3_agent.py builds the Isaac Capture device (past the #6656 RTX/camera regression).

    Omitting ``--teleop_device`` selects the Isaac Capture stack for the Isaac Capture-configured task.
    The run reaches the Isaac Capture device factory (``_ISAAC_TELEOP_MARKER``) and then stops when it
    tries to enter the OpenXR/CloudXR session (no headset in CI); reaching the marker is success.
    """
    argv = [
        "scripts/environments/teleoperation/teleop_se3_agent.py",
        "--task",
        _ISAAC_TELEOP_TASK,
        "physics=isaacsim_physx",
        *_NO_CLOUDXR,
    ]
    output = _launch_until_marker(argv, [_ISAAC_TELEOP_MARKER], tmp_path / "se3_agent.log")
    _assert_started_cleanly(output, [_ISAAC_TELEOP_MARKER], "teleop_se3_agent.py")


@pytest.mark.integration
@pytest.mark.rendering
def test_record_demos_starts(tmp_path):
    """record_demos.py builds the Isaac Capture device without AttributeError-ing on enable_cameras.

    Same Isaac Capture path as ``teleop_se3_agent``: reaching ``_ISAAC_TELEOP_MARKER`` proves the RTX
    apply and the ``args_cli.enable_cameras`` read both succeed before the CloudXR session boundary.
    """
    argv = [
        "scripts/tools/record_demos.py",
        "--task",
        _ISAAC_TELEOP_TASK,
        "physics=isaacsim_physx",
        "--dataset_file",
        str(tmp_path / "dataset.hdf5"),
        *_NO_CLOUDXR,
    ]
    output = _launch_until_marker(argv, [_ISAAC_TELEOP_MARKER], tmp_path / "record_demos.log")
    _assert_started_cleanly(output, [_ISAAC_TELEOP_MARKER], "record_demos.py")


@pytest.mark.integration
@pytest.mark.rendering
def test_record_demos_starts_when_reward_references_success(tmp_path):
    """record_demos.py preserves success-term references while disabling automatic resets."""
    argv = [
        "scripts/tools/record_demos.py",
        "--task",
        "Isaac-Reach-Franka",
        "--teleop_device",
        "keyboard",
        "--dataset_file",
        str(tmp_path / "dataset.hdf5"),
        "physics=isaacsim_physx",
        "presets=diffik",
    ]
    output = _launch_until_marker(argv, [_ENV_READY_MARKER], tmp_path / "record_demos_success_reward.log")
    _assert_started_cleanly(output, [_ENV_READY_MARKER], "record_demos.py with a success reward")


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


@pytest.mark.integration
@pytest.mark.rendering
def test_record_demos_starts_headless_xr(tmp_path):
    """record_demos.py starts under ``--xr`` with no ``--viz kit`` (the QA-reported failure).

    ``--xr`` without an explicit windowed visualizer resolves to headless and selects
    ``isaaclab.python.xr.openxr.headless.kit``, whose dependency chain aborted the app before any
    Python ran. Either marker is accepted: env creation already clears the dependency solver, which
    is what this gates on. The run stops short of the OpenXR session, which needs a headset.

    The registry is disabled so extensions must resolve from what Isaac Sim ships, keeping the test
    hermetic. See :func:`test_xr_experience_declares_only_shipped_extensions` for the warm-cache
    caveat.
    """
    argv = [
        "scripts/tools/record_demos.py",
        "--task",
        _ISAAC_TELEOP_TASK,
        "--dataset_file",
        str(tmp_path / "dataset.hdf5"),
        "--xr",
        *_NO_CLOUDXR,
        "--kit_args",
        "--/app/extensions/registryEnabled=false",
    ]
    markers = [_ISAAC_TELEOP_MARKER, _ENV_CREATED_MARKER]
    output = _launch_until_marker(argv, markers, tmp_path / "record_demos_xr.log")
    _assert_started_cleanly(output, markers, "record_demos.py --xr")


@pytest.mark.integration
@pytest.mark.rendering
def test_teleop_replay_agent_applies_rtx_without_replicator_crash(tmp_path):
    """teleop_replay_agent.py clears the RTX-apply site (create env) without the replicator crash.

    No MCAP fixture ships with the repo, so a non-existent ``--replay_file`` lets the run exercise
    ``_prepare_env_cfg`` (the regressed RTX-apply call) and env creation, then fail on the missing
    file. Reaching env creation with no replicator ``ModuleNotFoundError`` proves the fix.
    """
    argv = [
        "scripts/environments/teleoperation/teleop_replay_agent.py",
        "--task",
        _ISAAC_TELEOP_TASK,
        "--replay_file",
        str(tmp_path / "missing.mcap"),
        "--cloudxr_env",
        "none",
    ]
    output = _launch_until_marker(argv, [_ENV_CREATED_MARKER], tmp_path / "replay_agent.log")
    _assert_started_cleanly(output, [_ENV_CREATED_MARKER], "teleop_replay_agent.py")


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
        (True, None, "1", False),
        (True, " 1 ", " 1 ", False),
        (True, "0", "0", True),
        (True, "invalid", "invalid", True),
        (False, None, None, False),
    ],
)
@pytest.mark.parametrize("has_ovphysx", [False, True])
def test_teleop_ovxr_selects_external_stage_before_runtime(
    monkeypatch, kitless, selection, expected, reject, has_ovphysx
):
    """The actual task resolver selects OVStage, while incompatible explicit modes fail before launch validation."""
    from types import SimpleNamespace

    import isaaclab_teleop

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
        sim=SimpleNamespace(device="cpu", physics=OvPhysxCfg() if has_ovphysx else object()),
        scene=SimpleNamespace(num_envs=1),
        terminations=SimpleNamespace(time_out=1),
        isaac_teleop=object(),
        stabilize_for_kitless_ovphysx=lambda: preparations.append("ovphysx"),
    )
    scans = []
    source = ROOT / "scripts/environments/teleoperation/teleop_se3_agent.py"
    tree = ast.parse(source.read_text())
    resolve = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_env_cfg")
    namespace = {
        "prepare_teleop_backend": prepare_teleop_backend,
        "prepare_teleop_action": prepare_teleop_action,
        "action_state": {},
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
            using_ovxr=kitless,
            ovxr_pip_camera=None,
            geometry_publication_device=None,
            ovrtx_active_cuda_gpus=None,
            disable_external_cameras=False,
            physics=None,
        ),
    }
    exec(compile(ast.Module(body=[resolve], type_ignores=[]), str(source), "exec"), namespace)
    if reject:
        with pytest.raises(ValueError, match="requires.*OVStage"):
            namespace["_resolve_env_cfg"]()
        assert scans == []
    else:
        namespace["_resolve_env_cfg"]()
        assert scans == [expected]
    assert os.environ.get(name) == expected

    assert preparations == (["ovphysx"] if has_ovphysx and not reject else [])


def test_teleop_se3_agent_yields_only_when_ovxr_is_idle(monkeypatch):
    """A paused kitless loop must yield without slowing Kit or active physics frames."""
    source_path = ROOT / "scripts/environments/teleoperation/teleop_se3_agent.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    selected = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_pace_ovxr_control_loop"
    ]
    namespace = {"time": time, "Callable": Callable}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)

    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    monkeypatch.setattr(time, "perf_counter", lambda: 10.003)
    yield_idle = namespace["_pace_ovxr_control_loop"]

    timing = {"iteration_started_s": 10.0, "action_period_s": 0.02}
    yield_idle(using_ovxr=False, action_applied=False, **timing)
    yield_idle(using_ovxr=True, action_applied=True, **timing)
    assert sleeps == []

    yield_idle(using_ovxr=True, action_applied=False, **timing)
    assert sleeps == pytest.approx([0.017])
    yield_idle(using_ovxr=True, action_applied=False, iteration_started_s=9.9, action_period_s=0.02)
    assert sleeps == pytest.approx([0.017])  # Slow input work needs no added delay.


@pytest.mark.parametrize("active,step_hz,expected_polls", [(False, 0.0, 6), (True, 25.0, 3), (None, 25.0, 3)])
def test_kitless_loop_respects_control_rate_and_processes_reset(active, step_hz, expected_polls):
    """The real caller preserves idle pacing and applies an explicit active limit."""
    from types import SimpleNamespace

    import torch

    source_path = ROOT / "scripts/environments/teleoperation/teleop_se3_agent.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_pace_ovxr_control_loop",
            "_ovxr_publication_identity",
            "_apply_teleop_frame",
        }
    ]
    loop = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "run_loop")
    # Isolate the executable loop from CLI/GPU initialization, retaining its body.
    loop.body = [
        ast.copy_location(ast.Global(names=node.names), node) if isinstance(node, ast.Nonlocal) else node
        for node in loop.body
    ]
    selected.append(loop)
    clock = 0.0
    advances = 0
    resets = []
    refreshes = []
    publications = []
    applied = []
    seeded_from = []
    stepped_values = []
    sequence = active is None

    def sleep(duration):
        nonlocal clock
        clock += duration

    def advance():
        nonlocal advances, clock
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
    namespace = {
        "prepare_teleop_backend": prepare_teleop_backend,
        "prepare_teleop_action": prepare_teleop_action,
        "action_state": {},
        "time": SimpleNamespace(perf_counter=lambda: clock, sleep=sleep),
        "torch": torch,
        "env": SimpleNamespace(sim=sim, reset=lambda: resets.append(clock), step_dt=0.02, num_envs=1, step=step),
        "env_cfg": SimpleNamespace(isaac_teleop=SimpleNamespace(target_frame_prim_path=None)),
        "args_cli": SimpleNamespace(xr=True, using_ovxr=True, step_hz=step_hz),
        "use_isaac_teleop": True,
        "teleop_interface": SimpleNamespace(advance=advance, reset=lambda: None),
        "poll_control_events": lambda _: SimpleNamespace(
            is_active=advances not in (2, 4) if sequence else active, should_reset=advances == (5 if sequence else 3)
        ),
        "ov_libraries_xr_session": SimpleNamespace(
            shutdown_requested=False,
            wait_for_shutdown=lambda duration: sleep(duration),
            request_render=lambda identity, **kwargs: publications.append((identity, kwargs)) or 1,
            report_host_timing=lambda *args: None,
        ),
        "camera_feed_session": SimpleNamespace(refresh=lambda: refreshes.append(clock)),
        "initialized_hand_tracking": set(),
        "require_initial_bimanual_tracking": False,
        "action_conditioner": condition,
        "action_validator": lambda value: None,
        "Callable": Callable,
        "haptic_update": lambda: None,
        "haptic_stop": lambda: None,
        "should_reset_recording_instance": False,
        "skipped_tracking_frames": 0,
        "teleoperation_active": False,
        "reset_publication_generation": 0,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    namespace["run_loop"]()

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


def test_paused_loop_retries_the_latest_unaccepted_publication():
    """Run the real publication block: no new simulation step is needed to flush B."""
    from types import SimpleNamespace

    source_path = ROOT / "scripts/environments/teleoperation/teleop_se3_agent.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "state_changed" in ast.unparse(node)
        and "ov_libraries_xr_session.request_render" in ast.unparse(node)
    ]
    block = min(candidates, key=lambda node: len(ast.unparse(node)))
    calls = []
    admissions = iter((None, 12))

    def request(identity, *, force):
        calls.append((identity, force))
        return next(admissions)

    namespace = {
        "prepare_teleop_backend": prepare_teleop_backend,
        "prepare_teleop_action": prepare_teleop_action,
        "action_state": {},
        "ov_libraries_xr_session": SimpleNamespace(request_render=request),
        "state_changed": True,
        "publication_identity": "B",
        "pending_publication_identity": None,
        "force_publication": False,
    }
    code = compile(ast.Module(body=[block], type_ignores=[]), str(source_path), "exec")
    exec(code, namespace)
    assert namespace["pending_publication_identity"] == "B"
    namespace["state_changed"] = False
    exec(code, namespace)
    assert calls == [("B", False), ("B", False)]
    assert namespace["pending_publication_identity"] is None


def test_teleop_se3_agent_observes_ovxr_session_shutdown():
    """Closing the optional desktop mirror must end the application loop."""
    source = (ROOT / "scripts/environments/teleoperation/teleop_se3_agent.py").read_text(encoding="utf-8")
    loop_index = source.index("while env.sim.is_running()")
    loop_header = source[loop_index : source.index(":", loop_index)]

    assert "ov_libraries_xr_session.shutdown_requested" in loop_header
    assert "signal.SIGINT" in source
    assert "signal.SIGTERM" in source
    assert "ov_libraries_xr_session.request_shutdown()" in source


def test_teleop_se3_agent_defaults_ovxr_to_ar_and_routes_the_override():
    """Kitless XR defaults to AR while retaining an explicit opaque-VR override."""
    source = (ROOT / "scripts/environments/teleoperation/teleop_se3_agent.py").read_text(encoding="utf-8")

    assert '"--ovxr_session_mode"' in source
    assert 'choices=("ar", "vr")' in source
    assert 'default="ar"' in source
    assert "session_mode=args_cli.ovxr_session_mode" in source


@pytest.mark.parametrize("selector", [["--physics", "ovphysx"], ["physics=ovphysx"]])
@pytest.mark.parametrize("ovxr", [False, True])
def test_teleop_replay_agent_resolves_typed_presets_without_reapplying_physics(monkeypatch, selector, ovxr):
    """Public CLI aliases reach task resolution without resetting backend settings."""
    import runpy

    import isaaclab_teleop
    from isaaclab_ov.physics import OvPhysxCfg

    from isaaclab.managers import TerminationTermCfg
    from isaaclab.sim import SimulationCfg

    source_path = ROOT / "scripts/environments/teleoperation/teleop_replay_agent.py"
    argv = [str(source_path), "--task", "Test", "--replay_file", "recording.mcap", *selector]
    if ovxr:
        argv.append("--ovxr")
    monkeypatch.setattr(sys, "argv", argv)
    namespace = runpy.run_path(str(source_path))["_prepare_env_cfg"].__globals__
    stabilization = []
    physics = OvPhysxCfg()
    cfg = SimpleNamespace(
        sim=SimulationCfg(physics=physics),
        scene=SimpleNamespace(num_envs=1),
        terminations=SimpleNamespace(
            success=TerminationTermCfg(func=lambda env: False),
            time_out=TerminationTermCfg(func=lambda env: False),
        ),
        stabilize_for_kitless_ovphysx=lambda: stabilization.append("ovphysx"),
    )
    physics = cfg.sim.physics
    resolutions = []

    def resolve(task, agent, *, overrides):
        resolutions.append((task, agent, overrides))
        return cfg, None

    namespace["ManagerBasedRLEnvCfg"] = SimpleNamespace
    namespace["resolve_task_config"] = resolve
    monkeypatch.setattr(isaaclab_teleop, "remove_camera_configs", lambda cfg, **_kwargs: cfg)
    namespace["remove_camera_configs"] = isaaclab_teleop.remove_camera_configs
    prepared, success = namespace["_prepare_env_cfg"]("test-task", 1, None, overrides=namespace["hydra_overrides"])
    assert resolutions == [("test-task", "", ["physics=ovphysx"])]
    assert prepared is cfg and success is not None
    assert cfg.sim.physics is physics
    assert namespace["args_cli"].require_kit is False
    assert not hasattr(namespace["args_cli"], "physics")
    assert stabilization == ["ovphysx"]


@pytest.mark.parametrize(
    "backend,viewer,hard_reset",
    [("ovphysx", False, True), ("ovphysx", True, True), ("newton_mjwarp", True, False)],
)
def test_recorded_environment_reset_preserves_backend_policy_and_hand_cursor(backend, viewer, hard_reset):
    """Recorded RESET preserves physics ordering and the MCAP cursor with an optional viewer."""
    from types import SimpleNamespace

    source_path = ROOT / "scripts/environments/teleoperation/teleop_replay_agent.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    blocks = [
        node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.unparse(node.test) == "ctrl.should_reset"
    ]
    assert len(blocks) == 1
    body = [node for node in blocks[0].body if not isinstance(node, ast.Continue)]
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_handle_reset",
            "_reset_replay_environment",
            "_replay_requires_hard_sim_reset",
        }
    ]
    calls = []
    namespace = {
        "prepare_teleop_backend": prepare_teleop_backend,
        "prepare_teleop_action": prepare_teleop_action,
        "action_state": {"seed": True},
        "initialized_hand_tracking": {"left", "right"},
        "gym": SimpleNamespace(Env=object),
        "IsaacTeleopDevice": object,
        "env": SimpleNamespace(
            scene=SimpleNamespace(physics_backend=backend),
            sim=SimpleNamespace(reset=lambda: calls.append("sim-reset")),
            recorder_manager=SimpleNamespace(reset=lambda: calls.append("recorder-reset")),
            reset=lambda: calls.append("env-reset"),
        ),
        "teleop_interface": SimpleNamespace(
            reset=lambda: calls.append("device-reset"), request_start=lambda: calls.append("start")
        ),
        "presentation": SimpleNamespace(publish=lambda *_args, **_kwargs: calls.append("publish-reset"))
        if viewer
        else None,
    }
    exec(compile(ast.Module(body=[*helpers, *body], type_ignores=[]), str(source_path), "exec"), namespace)
    expected = ["sim-reset"] if hard_reset else []
    expected.extend(["recorder-reset", "env-reset", "device-reset", "start"])
    if viewer:
        expected.append("publish-reset")
    assert calls == expected

    assert namespace["action_state"] == {}
    assert namespace["initialized_hand_tracking"] == set()


def _load_replay_reset_helpers():
    """Load the replay reset helpers without importing the application launcher."""
    source = (ROOT / "scripts/environments/teleoperation/teleop_replay_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_names = {"_replay_requires_hard_sim_reset", "_reset_replay_environment"}
    helper_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    assert {node.name for node in helper_nodes} == helper_names
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), "<replay-reset-helpers>", "exec"), namespace)
    return namespace


def test_teleop_replay_reset_policy_skips_hard_reset_only_for_newton():
    """Newton keeps its constructed solver while OVPhysX retains the PR-5507 hard reset."""
    requires_hard_reset = _load_replay_reset_helpers()["_replay_requires_hard_sim_reset"]

    assert requires_hard_reset("newton_mjwarp") is False
    assert requires_hard_reset("newtonmanager") is False
    assert requires_hard_reset("ovphysx") is True
    assert requires_hard_reset(None) is True


def test_teleop_replay_reset_sequence_preserves_backend_owned_order():
    """The backend policy changes only the optional simulator reset in the reset sequence."""
    reset_environment = _load_replay_reset_helpers()["_reset_replay_environment"]
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


def test_teleop_replay_routes_initial_and_recorded_resets_through_backend_policy():
    """Both replay reset sites must use the same backend-aware hard-reset decision."""
    source = (ROOT / "scripts/environments/teleoperation/teleop_replay_agent.py").read_text(encoding="utf-8")

    assert source.count("hard_sim_reset=_replay_requires_hard_sim_reset(env.scene.physics_backend)") == 2
    assert "env.sim.reset()" not in source[source.index("def _handle_reset") :]


@pytest.mark.parametrize("use_isaac_teleop", [False, True])
def test_teleop_se3_agent_reads_action_policy_only_for_isaac_teleop(use_isaac_teleop):
    """Exercise the real policy expressions without launching Kit or devices."""
    from types import SimpleNamespace

    source_path = ROOT / "scripts/environments/teleoperation/teleop_se3_agent.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {"require_initial_bimanual_tracking", "action_conditioner", "action_validator"}
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ]
    assert len(assignments) == 3
    policy = SimpleNamespace(
        require_initial_bimanual_hand_tracking=True, action_conditioner=object(), action_validator=object()
    )
    namespace = {
        "prepare_teleop_backend": prepare_teleop_backend,
        "prepare_teleop_action": prepare_teleop_action,
        "action_state": {},
        "use_isaac_teleop": use_isaac_teleop,
        "env_cfg": SimpleNamespace(isaac_teleop=policy) if use_isaac_teleop else SimpleNamespace(),
    }
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(source_path), "exec"), namespace)
    assert namespace["require_initial_bimanual_tracking"] is use_isaac_teleop
    assert namespace["action_conditioner"] is (policy.action_conditioner if use_isaac_teleop else None)
    assert namespace["action_validator"] is (policy.action_validator if use_isaac_teleop else None)


def _load_teleop_agent(monkeypatch, *arguments):
    path = ROOT / "scripts" / "environments" / "teleoperation" / "teleop_se3_agent.py"
    monkeypatch.setattr(sys, "argv", [str(path), *arguments])
    spec = importlib.util.spec_from_file_location("_gr1t2_teleop_agent_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
    prepare_teleop_backend(env_cfg, physics_cfg)
    if physics == "newton_mjwarp":
        if "Locomanipulation-G1" in task_name:
            assert physics_cfg.num_substeps == 2
            assert physics_cfg.solver_cfg.integrator == "implicitfast"
        else:
            assert physics_cfg.solver_cfg.nconmax == 4096
            assert physics_cfg.solver_cfg.njmax == 1536
    else:
        assert hasattr(env_cfg.scene.packing_table.spawn, "_packing_table_collision_prepared") == (physics == "ovphysx")

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
    interface = SimpleNamespace(_session_lifecycle=lifecycle)
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
