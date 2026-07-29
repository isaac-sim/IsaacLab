# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robustness smoke tests for standalone demo and tutorial scripts.

Set ``ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS=1`` to enable the simulator launch
matrix. GUI cases additionally require ``DISPLAY`` or ``WAYLAND_DISPLAY``.
``ISAACLAB_STANDALONE_SOAK_TIME`` and ``ISAACLAB_STANDALONE_STARTUP_TIMEOUT``
may be used to tune the default five-second soak and five-minute startup limit.
Set ``ISAACLAB_STANDALONE_VISUALIZER`` to run one visualizer slice of the matrix.
Set ``ISAACLAB_STANDALONE_SCRIPT_RUNTIME_GROUP`` to ``kit`` or ``non-kit`` to
run the corresponding backend-runtime group.
``ISAACLAB_STANDALONE_SCREENSHOT_DELAY`` controls when visual evidence is captured.
"""

import ast
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import standalone_script_cases as script_cases
from standalone_script_cases import (
    OVERRIDES,
    SmokeResult,
    assert_smoke_passed,
    backend_is_available,
    build_cases,
    discover_specs,
    gui_is_available,
    run_until_ready,
    select_runtime_group,
    select_script_scope,
    visualizer_is_available,
)

SPECS = discover_specs()
SCOPE = os.environ.get("ISAACLAB_STANDALONE_SCRIPT_SCOPE", "all")
SELECTED_SPECS = select_script_scope(SPECS, SCOPE)
CASES = build_cases(SELECTED_SPECS)
VISUALIZER = os.environ.get("ISAACLAB_STANDALONE_VISUALIZER")
if VISUALIZER:
    if VISUALIZER not in script_cases.VISUALIZERS:
        raise ValueError(f"unsupported ISAACLAB_STANDALONE_VISUALIZER: {VISUALIZER!r}")
    CASES = [case for case in CASES if case.visualizer == VISUALIZER]
RUNTIME_GROUP = os.environ.get("ISAACLAB_STANDALONE_SCRIPT_RUNTIME_GROUP")
if RUNTIME_GROUP:
    CASES = select_runtime_group(CASES, RUNTIME_GROUP)
MULTI_MESH_RAYCASTER_CASES = [
    case
    for case in CASES
    if case.spec.relative_path == "scripts/demos/sensors/multi_mesh_raycaster.py" and case.visualizer == "none"
]
RUN_LAUNCH_MATRIX = os.environ.get("ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS") == "1"
SCREENSHOT_DIR = os.environ.get("ISAACLAB_STANDALONE_SCREENSHOT_DIR")
SOAK_TIME = float(os.environ.get("ISAACLAB_STANDALONE_SOAK_TIME", "5"))
STARTUP_TIMEOUT = float(os.environ.get("ISAACLAB_STANDALONE_STARTUP_TIMEOUT", "300"))
SCREENSHOT_DELAY = float(os.environ.get("ISAACLAB_STANDALONE_SCREENSHOT_DELAY", "3"))


def test_every_standalone_script_has_a_readiness_contract_or_exemption():
    """Every executable demo/tutorial must be runnable or explicitly exempted."""
    missing = [spec.relative_path for spec in SPECS if spec.readiness_pattern is None and spec.skip_reason is None]
    assert not missing, f"standalone scripts need a readiness marker or OVERRIDES exemption: {missing}"


def test_overrides_only_reference_discovered_standalone_scripts():
    """Stale override entries must not silently survive script removal or renaming."""
    discovered = {spec.relative_path for spec in SPECS}
    stale = sorted(set(OVERRIDES) - discovered)
    assert not stale, f"stale standalone script overrides: {stale}"


def test_launch_matrix_covers_declared_backends_and_visualizers():
    """Each script must expand across every declared backend and visualizer."""
    for spec in SPECS:
        spec_cases = build_cases([spec])
        assert {case.physics_backend for case in spec_cases} == {backend for _, backend in spec.physics_backends}
        assert {case.renderer_backend for case in spec_cases} == {backend for _, backend in spec.rendering_backends}
        assert {case.visualizer for case in spec_cases} == set(spec.visualizers)
        assert len(spec_cases) == len(spec.physics_backends) * len(spec.rendering_backends) * len(spec.visualizers)


def test_runtime_groups_partition_matrix_without_overlap():
    """Kit and non-Kit groups must cover every launch case exactly once."""
    cases = build_cases(SPECS)
    groups = [select_runtime_group(cases, runtime_group) for runtime_group in ("kit", "non-kit")]
    grouped_ids = [case.id for group in groups for case in group]
    assert sorted(grouped_ids) == sorted(case.id for case in cases)
    assert len(grouped_ids) == len(set(grouped_ids))
    assert all(
        case.physics_backend == "isaacsim_physx" or case.renderer_backend == "isaac_rtx" or case.visualizer == "kit"
        for case in groups[0]
    )
    assert all(
        case.physics_backend != "isaacsim_physx" and case.renderer_backend != "isaac_rtx" and case.visualizer != "kit"
        for case in groups[1]
    )

    with pytest.raises(ValueError, match="runtime group"):
        select_runtime_group(cases, "invalid")


def test_script_scope_rejects_empty_selection():
    """A stale or misspelled scope must not produce a vacuously green launch matrix."""
    assert select_script_scope(SPECS, "all") is SPECS
    assert all(spec.relative_path.startswith("scripts/demos/mpm/") for spec in select_script_scope(SPECS, "demos/mpm"))
    with pytest.raises(ValueError, match="selected no scripts"):
        select_script_scope(SPECS, "missing")


def test_showroom_documents_options_for_each_mentioned_demo():
    """Every demo showcased in the showroom must list its supported launch options."""

    def documented_values(entry: str, label: str) -> set[str]:
        match = re.search(rf"(?ms)^   \*\*{label}:\*\*[ \t]*(.+?)(?=\n\n|\Z)", entry)
        assert match is not None, f"showroom entry does not list {label.lower()} options"
        return set(re.findall(r"``([^`]+)``", match.group(1)))

    showroom = (script_cases.ROOT / "docs/source/overview/showroom.rst").read_text(encoding="utf-8")
    entries = re.findall(r"(?ms)^-  .*?(?=^-  |\Z)", showroom)
    documented_entries = {}
    for entry in entries:
        paths = set(re.findall(r"scripts/demos/[A-Za-z0-9_./-]+\.py", entry))
        assert len(paths) <= 1, f"showroom entry references multiple demos: {paths}"
        if paths:
            documented_entries[paths.pop()] = entry

    referenced_paths = set(re.findall(r"scripts/demos/[A-Za-z0-9_./-]+\.py", showroom))
    assert documented_entries.keys() == referenced_paths

    demo_specs = {
        spec.relative_path: spec
        for spec in SPECS
        if spec.relative_path.startswith("scripts/demos/") and spec.relative_path in referenced_paths
    }
    assert demo_specs.keys() == referenced_paths
    for path, spec in demo_specs.items():
        entry = documented_entries[path]
        expected_physics = {backend for _, backend in spec.physics_backends}
        expected_visualizers = set(spec.visualizers)
        assert documented_values(entry, "Physics") == expected_physics, f"{path} documents incorrect physics options"
        assert documented_values(entry, "Visualizer") == expected_visualizers, (
            f"{path} documents incorrect visualizer options"
        )

        selectable_renderers = {backend for option, backend in spec.rendering_backends if option is not None}
        if selectable_renderers:
            assert documented_values(entry, "Renderer") == selectable_renderers, (
                f"{path} documents incorrect renderer options"
            )
        else:
            assert "**Renderer:**" not in entry, f"{path} advertises a renderer option that it does not expose"


def test_commands_respect_script_launcher_capabilities():
    """Commands must enable cameras and avoid unsupported launcher arguments."""
    camera_case = next(
        case
        for case in build_cases(SPECS)
        if case.spec.relative_path == "scripts/demos/sensors/cameras.py" and case.visualizer == "none"
    )
    assert camera_case.command()[3:5] == ["--num_envs", "1"]
    assert camera_case.command()[-2:] == ["--visualizer", "none"]

    kitless_case = next(
        case for case in build_cases(SPECS) if case.spec.relative_path == "scripts/demos/sensors/ppisp_camera_ovrtx.py"
    )
    assert kitless_case.command()[-2:] == ["--viz", "none"]

    renderer_case = next(
        case
        for case in build_cases(SPECS)
        if case.spec.relative_path == "scripts/demos/sensors/ppisp_camera.py" and case.renderer_backend == "isaac_rtx"
    )
    assert renderer_case.command()[-4:] == ["--renderer", "isaac_rtx", "--visualizer", "none"]

    newton_renderer_case = next(
        case
        for case in build_cases(SPECS)
        if case.spec.relative_path == "scripts/demos/sensors/ppisp_camera.py"
        and case.renderer_backend == "newton_renderer"
    )
    assert newton_renderer_case.command()[-4:] == ["--renderer", "newton_renderer", "--visualizer", "none"]

    physics_case = next(
        case
        for case in build_cases(SPECS)
        if case.spec.relative_path == "scripts/demos/bin_packing.py"
        and case.physics_backend == "newton_mjwarp"
        and case.visualizer == "none"
    )
    assert "--num_envs" in physics_case.command()
    num_envs_index = physics_case.command().index("--num_envs")
    assert physics_case.command()[num_envs_index + 1] == "2"
    assert physics_case.command()[-4:] == ["--physics", "newton_mjwarp", "--visualizer", "none"]

    multi_asset_case = next(
        case
        for case in build_cases(SPECS)
        if case.spec.relative_path == "scripts/demos/multi_asset.py"
        and case.physics_backend == "newton_mjwarp"
        and case.visualizer == "none"
    )
    assert multi_asset_case.command()[3:5] == ["--num_envs", "4"]

    surface_gripper_case = next(
        case
        for case in build_cases(SPECS)
        if case.spec.relative_path == "scripts/tutorials/01_assets/run_surface_gripper.py" and case.visualizer == "none"
    )
    assert "--device" in surface_gripper_case.command()
    assert "cpu" in surface_gripper_case.command()

    ray_camera_case = next(
        case
        for case in build_cases(SPECS)
        if case.spec.relative_path == "scripts/tutorials/04_sensors/run_ray_caster_camera.py"
        and case.visualizer == "none"
    )
    assert "--enable_cameras" in ray_camera_case.command()


def test_launch_case_reports_script_and_combination_exemptions():
    """Whole-script and individual-combination exemptions must remain distinguishable."""
    skipped_script = next(spec for spec in SPECS if spec.relative_path == "scripts/demos/h1_locomotion.py")
    assert build_cases([skipped_script])[0].skip_reason == skipped_script.skip_reason
    spec = next(spec for spec in SPECS if spec.skip_reason is None)
    case = build_cases([spec])[0]
    reason = "unsupported combination"
    key = (case.physics_backend, case.renderer_backend, case.visualizer)
    case = replace(case, spec=replace(spec, case_skip_reasons={key: reason}))
    assert case.skip_reason == reason


def test_subprocess_supervisor_soaks_then_stops_process_group():
    """The supervisor must recognize readiness and bound an infinite script."""
    command = [sys.executable, "-u", "-c", "import time; print('READY'); time.sleep(30)"]
    result = run_until_ready(command, r"READY", startup_timeout=2.0, soak_time=0.05)
    assert result.ready
    assert result.stopped_after_soak
    assert result.elapsed < 2.0


def test_subprocess_supervisor_accepts_clean_exit_after_readiness():
    """A finite script may exit successfully immediately after becoming ready."""
    result = run_until_ready([sys.executable, "-u", "-c", "print('READY')"], r"READY", startup_timeout=2.0)
    assert result.ready
    assert result.returncode == 0
    assert not result.stopped_after_soak


def test_subprocess_supervisor_rejects_exit_before_readiness():
    """A successful process exit is insufficient without its readiness contract."""
    result = run_until_ready([sys.executable, "-u", "-c", "print('not ready')"], r"READY", startup_timeout=2.0)
    assert not result.ready
    assert result.returncode == 0


def test_subprocess_supervisor_bounds_startup_time():
    """A process that never becomes ready must be terminated at the startup deadline."""
    command = [sys.executable, "-u", "-c", "import time; time.sleep(30)"]
    result = run_until_ready(command, r"READY", startup_timeout=0.05)
    assert not result.ready
    assert result.elapsed < 2.0
    assert result.returncode is not None


def test_subprocess_supervisor_retains_fatal_state_when_output_is_truncated(monkeypatch):
    """Fatal output must remain detectable after the bounded output tail rolls over."""
    monkeypatch.setattr(script_cases, "MAX_OUTPUT_BYTES", 64)
    source = (
        "print('Traceback (most recent call last):'); "
        "print('Number of Newton contacts (10) exceeded MJWarp limit (2). Increase nconmax.'); "
        "print('nefc overflow - please increase njmax to 10'); "
        "print('x' * 1024); print('READY')"
    )
    result = run_until_ready([sys.executable, "-u", "-c", source], r"READY", startup_timeout=2.0)
    assert result.ready
    assert len(result.output.encode()) <= 64
    assert "Traceback (most recent call last):" in result.fatal_patterns
    assert "exceeded MJWarp limit" in result.fatal_patterns
    assert "nefc overflow" in result.fatal_patterns


def test_subprocess_supervisor_captures_requested_screenshot(monkeypatch, tmp_path):
    """A healthy visual launch must trigger one screenshot during its soak."""
    captured = []
    monkeypatch.setattr(script_cases, "_capture_screenshot", captured.append)
    screenshot_path = tmp_path / "launch.png"
    command = [sys.executable, "-u", "-c", "import time; print('READY'); time.sleep(30)"]
    result = run_until_ready(
        command,
        r"READY",
        startup_timeout=2.0,
        soak_time=0.1,
        screenshot_path=screenshot_path,
    )
    assert result.ready
    assert captured == [screenshot_path]


def test_screenshot_capture_reports_external_tool_failure(monkeypatch, tmp_path):
    """Screenshot failures must be surfaced instead of producing missing visual evidence."""
    completed = subprocess.CompletedProcess([], 1, "", "display unavailable")
    monkeypatch.setattr(script_cases.subprocess, "run", lambda *args, **kwargs: completed)
    with pytest.raises(RuntimeError, match="display unavailable"):
        script_cases._capture_screenshot(tmp_path / "capture.png")

    completed = subprocess.CompletedProcess([], 0, "", "")
    monkeypatch.setattr(script_cases.subprocess, "run", lambda *args, **kwargs: completed)
    with pytest.raises(RuntimeError, match="did not create a non-empty image"):
        script_cases._capture_screenshot(tmp_path / "missing.png")


def test_screenshot_capture_invokes_imagemagick(monkeypatch, tmp_path):
    """Screenshot capture must create its destination and target the root window."""
    calls = []
    completed = subprocess.CompletedProcess([], 0, "", "")
    path = tmp_path / "nested" / "capture.png"

    def run_and_create_image(*args, **kwargs):
        calls.append((args, kwargs))
        path.write_bytes(b"image")
        return completed

    monkeypatch.setattr(script_cases.subprocess, "run", run_and_create_image)
    script_cases._capture_screenshot(path)
    assert path.parent.is_dir()
    assert calls[0][0][0] == ["import", "-window", "root", str(path)]


def test_smoke_assertion_rejects_each_failure_mode():
    """The result assertion must reject fatal output, missing readiness, and early crashes."""
    case = build_cases([SPECS[0]])[0]
    with pytest.raises(AssertionError, match="fatal output"):
        assert_smoke_passed(
            SmokeResult(True, 0, "tail", 0.1, False, ("Fatal Python error:",)),
            case,
        )
    with pytest.raises(AssertionError, match="did not reach"):
        assert_smoke_passed(SmokeResult(False, 0, "tail", 0.1, False), case)
    with pytest.raises(AssertionError, match="exited with 2"):
        assert_smoke_passed(SmokeResult(True, 2, "tail", 0.1, False), case)


def test_ast_discovery_recognizes_main_guards_and_literal_choices():
    """Static discovery must distinguish executable scripts and preserve literal choices."""
    tree = ast.parse(
        """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--physics', choices=['physx', 'newton_mjwarp'])
parser.add_argument('--viz', choices=('none',))
parser.add_argument('positional')
if __name__ == '__main__':
    pass
"""
    )
    assert script_cases._has_main_guard(tree)
    assert script_cases._literal_cli_options(tree) == {
        "--physics": ("physx", "newton_mjwarp"),
        "--viz": ("none",),
    }
    assert not script_cases._has_main_guard(ast.parse("print('library module')"))


@pytest.mark.parametrize(
    ("backend", "package"),
    [("isaacsim_physx", "isaaclab_physx"), ("newton_mjwarp", "isaaclab_newton")],
)
def test_backend_availability_resolves_implementation_package(monkeypatch, backend, package):
    """Backend gating must query the package that implements each declared backend."""
    queried = []
    monkeypatch.setattr(script_cases.importlib.util, "find_spec", lambda name: queried.append(name) or object())
    assert backend_is_available(backend)
    assert queried == [package]


def test_builtin_backend_availability_handles_default_and_isaac_rtx(monkeypatch):
    """Built-in backend gates must not require an extension package lookup."""
    monkeypatch.setattr(script_cases.importlib.util, "find_spec", lambda name: None)
    assert backend_is_available("default")
    assert backend_is_available("isaac_rtx") == (script_cases.ROOT / "_isaac_sim").exists()


def test_visualizer_availability_requires_shared_and_backend_packages(monkeypatch):
    """External visualizers require both the visualizer extension and selected implementation."""
    available = {"isaaclab_visualizers", "isaaclab_newton", "rerun"}
    monkeypatch.setattr(script_cases.importlib.util, "find_spec", lambda name: object() if name in available else None)
    assert visualizer_is_available("newton")
    assert visualizer_is_available("rerun")
    assert not visualizer_is_available("viser")

    monkeypatch.setattr(script_cases.importlib.util, "find_spec", lambda name: None)
    assert not visualizer_is_available("rerun")


def test_builtin_visualizer_availability_handles_none_and_kit(monkeypatch):
    """Built-in visualizer gates must recognize headless mode and local Isaac Sim."""
    monkeypatch.setattr(script_cases.importlib.util, "find_spec", lambda name: None)
    assert visualizer_is_available("none")
    assert visualizer_is_available("kit") == (script_cases.ROOT / "_isaac_sim").exists()


def test_gui_availability_accepts_x11_or_wayland(monkeypatch):
    """GUI gating must work with either supported Linux display protocol."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert not gui_is_available()
    monkeypatch.setenv("DISPLAY", ":1")
    assert gui_is_available()
    monkeypatch.delenv("DISPLAY")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert gui_is_available()


def test_process_group_termination_handles_exited_and_stubborn_processes(monkeypatch):
    """Termination must no-op after exit and escalate a process that ignores SIGTERM."""
    exited = type("Exited", (), {"poll": lambda self: 0})()
    script_cases._terminate_process_group(exited)

    class Stubborn:
        pid = 123
        waits = 0

        def poll(self):
            return None

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("demo", timeout)
            return -9

    signals = []
    monkeypatch.setattr(script_cases.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    stubborn = Stubborn()
    script_cases._terminate_process_group(stubborn)
    assert signals == [(123, script_cases.signal.SIGTERM), (123, script_cases.signal.SIGKILL)]


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.smoke
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_standalone_script_remains_healthy_after_startup(case):
    """Each supported script launch must initialize and survive a short soak."""
    if not RUN_LAUNCH_MATRIX:
        pytest.skip("set ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS=1 to run the launch matrix")
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    missing_modules = [module for module in case.spec.required_modules if not script_cases.module_is_available(module)]
    if missing_modules:
        pytest.skip(f"required runtime module(s) not installed: {', '.join(missing_modules)}")
    if case.visualizer in {"kit", "newton"} and not gui_is_available():
        pytest.skip("GUI smoke test requires DISPLAY or WAYLAND_DISPLAY")
    if not backend_is_available(case.physics_backend):
        pytest.skip(f"physics backend package for {case.physics_backend!r} is not installed")
    if not backend_is_available(case.renderer_backend):
        pytest.skip(f"renderer backend package for {case.renderer_backend!r} is not installed")
    if not visualizer_is_available(case.visualizer):
        pytest.skip(f"visualizer package for {case.visualizer!r} is not installed")

    assert case.spec.readiness_pattern is not None
    screenshot_path = None
    if SCREENSHOT_DIR and case.visualizer != "none" and gui_is_available():
        screenshot_path = Path(SCREENSHOT_DIR) / f"{case.id}.png"
    result = run_until_ready(
        case.command(),
        case.spec.readiness_pattern,
        startup_timeout=max(STARTUP_TIMEOUT, case.spec.startup_timeout or 0.0),
        soak_time=SOAK_TIME,
        screenshot_path=screenshot_path,
        screenshot_delay=SCREENSHOT_DELAY,
    )
    assert_smoke_passed(result, case)


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.smoke
@pytest.mark.parametrize("asset_type", ("anymal_d", "objects"))
@pytest.mark.parametrize("case", MULTI_MESH_RAYCASTER_CASES, ids=lambda case: case.physics_backend)
def test_multi_mesh_raycaster_supports_each_asset_type(case, asset_type):
    """The multi-mesh raycaster must support every non-default asset path on each physics backend."""
    if not RUN_LAUNCH_MATRIX:
        pytest.skip("set ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS=1 to run the launch matrix")
    if not backend_is_available(case.physics_backend):
        pytest.skip(f"physics backend package for {case.physics_backend!r} is not installed")

    command = [*case.command(), "--asset_type", asset_type]
    result = run_until_ready(
        command,
        case.spec.readiness_pattern,
        startup_timeout=max(STARTUP_TIMEOUT, case.spec.startup_timeout or 0.0),
        soak_time=SOAK_TIME,
    )
    assert_smoke_passed(result, case)


def test_launch_matrix_skips_declared_exemption(monkeypatch):
    """A declared script exemption must win before runtime capability checks."""
    monkeypatch.setattr(sys.modules[__name__], "RUN_LAUNCH_MATRIX", True)
    case = build_cases([next(spec for spec in SPECS if spec.skip_reason)])[0]
    with pytest.raises(pytest.skip.Exception, match=case.skip_reason):
        test_standalone_script_remains_healthy_after_startup(case)


def test_launch_matrix_skips_unavailable_runtime_capabilities(monkeypatch):
    """Display, physics, renderer, and visualizer gates must report distinct reasons."""
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "RUN_LAUNCH_MATRIX", True)
    base_case = next(case for case in build_cases(SPECS) if case.skip_reason is None and case.visualizer == "none")

    missing_module_case = replace(base_case, spec=replace(base_case.spec, required_modules=("missing_runtime",)))
    monkeypatch.setattr(script_cases, "module_is_available", lambda module_name: False)
    with pytest.raises(pytest.skip.Exception, match="required runtime module"):
        test_standalone_script_remains_healthy_after_startup(missing_module_case)

    monkeypatch.setattr(module, "gui_is_available", lambda: False)
    with pytest.raises(pytest.skip.Exception, match="GUI smoke test"):
        test_standalone_script_remains_healthy_after_startup(replace(base_case, visualizer="kit"))

    monkeypatch.setattr(module, "backend_is_available", lambda backend: backend != "missing")
    with pytest.raises(pytest.skip.Exception, match="physics backend package"):
        test_standalone_script_remains_healthy_after_startup(replace(base_case, physics_backend="missing"))
    with pytest.raises(pytest.skip.Exception, match="renderer backend package"):
        test_standalone_script_remains_healthy_after_startup(replace(base_case, renderer_backend="missing"))

    monkeypatch.setattr(module, "visualizer_is_available", lambda visualizer: False)
    with pytest.raises(pytest.skip.Exception, match="visualizer package"):
        test_standalone_script_remains_healthy_after_startup(base_case)


def test_launch_matrix_runs_supported_case_with_screenshot(monkeypatch, tmp_path):
    """A supported visual case must forward supervision and screenshot settings."""
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "RUN_LAUNCH_MATRIX", True)
    monkeypatch.setattr(module, "SCREENSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(module, "SCREENSHOT_DELAY", 3.0)
    monkeypatch.setattr(module, "gui_is_available", lambda: True)
    monkeypatch.setattr(module, "backend_is_available", lambda backend: True)
    monkeypatch.setattr(module, "visualizer_is_available", lambda visualizer: True)
    case = replace(next(case for case in build_cases(SPECS) if case.skip_reason is None), visualizer="newton")
    case = replace(case, spec=replace(case.spec, startup_timeout=600.0))
    calls = []

    def _run(*args, **kwargs):
        calls.append((args, kwargs))
        return SmokeResult(True, 0, "ready", 0.1, False)

    monkeypatch.setattr(module, "run_until_ready", _run)
    test_standalone_script_remains_healthy_after_startup(case)

    assert calls[0][0] == (case.command(), case.spec.readiness_pattern)
    assert calls[0][1]["startup_timeout"] == 600.0
    assert calls[0][1]["screenshot_path"] == tmp_path / f"{case.id}.png"
    assert calls[0][1]["screenshot_delay"] == 3.0


def test_launch_matrix_runs_web_visualizer_without_display(monkeypatch):
    """Web visualizers must remain testable on headless CI workers."""
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "RUN_LAUNCH_MATRIX", True)
    monkeypatch.setattr(module, "gui_is_available", lambda: False)
    monkeypatch.setattr(module, "backend_is_available", lambda backend: True)
    monkeypatch.setattr(module, "visualizer_is_available", lambda visualizer: True)
    monkeypatch.setattr(module, "run_until_ready", lambda *args, **kwargs: SmokeResult(True, 0, "ready", 0.1, False))
    case = replace(next(case for case in build_cases(SPECS) if case.skip_reason is None), visualizer="rerun")
    test_standalone_script_remains_healthy_after_startup(case)
