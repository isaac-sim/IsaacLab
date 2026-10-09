# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robustness smoke tests for packaged programs and standalone tutorial scripts.

Set ``ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS=1`` to enable the simulator launch
matrix. Scripts with a ``--max_steps`` option run a few steps and must exit
cleanly; other scripts must print their readiness marker and survive a soak.
GUI cases additionally require ``DISPLAY`` or ``WAYLAND_DISPLAY``.
``ISAACLAB_STANDALONE_SOAK_TIME`` and ``ISAACLAB_STANDALONE_STARTUP_TIMEOUT``
may be used to tune the default five-second soak and five-minute time limit.
Set ``ISAACLAB_STANDALONE_VISUALIZER`` to run one visualizer slice of the matrix, or ``default`` for the
launches without ``--viz``.
Set ``ISAACLAB_STANDALONE_SCRIPT_RUNTIME_GROUP`` to ``kit`` or ``non-kit`` to
run the corresponding backend-runtime group.
``ISAACLAB_STANDALONE_SCREENSHOT_DIR`` captures a screenshot of soaked visual
launches; ``ISAACLAB_STANDALONE_SCREENSHOT_DELAY`` controls when.
"""

import ast
import os
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import standalone_script_cases as script_cases
from scipy.spatial.transform import Rotation
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
    selected_visualizer = None if VISUALIZER == "default" else VISUALIZER
    if selected_visualizer not in script_cases.VISUALIZERS:
        raise ValueError(f"unsupported ISAACLAB_STANDALONE_VISUALIZER: {VISUALIZER!r}")
    CASES = [case for case in CASES if case.visualizer == selected_visualizer]
RUNTIME_GROUP = os.environ.get("ISAACLAB_STANDALONE_SCRIPT_RUNTIME_GROUP")
if RUNTIME_GROUP:
    CASES = select_runtime_group(CASES, RUNTIME_GROUP)
MULTI_MESH_RAYCASTER_CASES = [
    case
    for case in CASES
    if case.spec.relative_path == "examples/sensors/multi_mesh_raycaster.py" and case.visualizer is None
]
NEWTON_RAYCAST_CASES = [
    case
    for case in CASES
    if case.spec.relative_path == "examples/sensors/newton_raycast.py" and case.visualizer is None
]
RUN_LAUNCH_MATRIX = os.environ.get("ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS") == "1"
SCREENSHOT_DIR = os.environ.get("ISAACLAB_STANDALONE_SCREENSHOT_DIR")
SOAK_TIME = float(os.environ.get("ISAACLAB_STANDALONE_SOAK_TIME", "5"))
STARTUP_TIMEOUT = float(os.environ.get("ISAACLAB_STANDALONE_STARTUP_TIMEOUT", "300"))
SCREENSHOT_DELAY = float(os.environ.get("ISAACLAB_STANDALONE_SCREENSHOT_DELAY", "3"))

_READY_THEN_SLEEP = "import time; print('READY'); time.sleep(30)"


def _python(source: str) -> list[str]:
    """Return a command that runs *source* in an unbuffered Python child process."""
    return [sys.executable, "-u", "-c", source]


# Script contracts.


def test_every_standalone_script_has_a_launch_contract_or_exemption():
    """Packaged programs must stop themselves; other scripts must be runnable or explicitly exempted."""
    not_finite = [spec.relative_path for spec in SPECS if spec.program is not None and not spec.finite]
    assert not not_finite, f"packaged programs need a --max_steps option: {not_finite}"
    missing = [
        spec.relative_path
        for spec in SPECS
        if not spec.finite and spec.readiness_pattern is None and spec.skip_reason is None
    ]
    assert not missing, f"standalone scripts need --max_steps, a readiness marker, or an OVERRIDES exemption: {missing}"


def test_overrides_only_reference_discovered_standalone_scripts():
    """Stale override entries must not silently survive script removal or renaming."""
    stale = sorted(set(OVERRIDES) - {spec.relative_path for spec in SPECS})
    assert not stale, f"stale standalone script overrides: {stale}"


def test_every_packaged_script_is_registered_for_cli_launch():
    """Every executable script in the root examples tree must have one CLI entry."""
    packaged_specs = [spec for spec in SPECS if spec.path.is_relative_to(script_cases.EXAMPLE_ROOT)]
    assert {spec.path for spec in packaged_specs} == set(script_cases.PROGRAMS_BY_PATH)


def test_demo_browser_documents_options_for_each_demo():
    """Every demo card must match a demo and offer exactly the options the demo accepts."""
    docs_source = script_cases.ROOT / "docs/source"
    demos_page = (docs_source / "setup/demos.rst").read_text(encoding="utf-8")
    documented_entries = {}
    for card in re.findall(r'(?s)<button[^>]+data-demo-id="[^"]+"[^>]*>', demos_page):
        attributes = dict(re.findall(r'data-demo-([\w-]+)="([^"]*)"', card))
        demo_name = attributes.pop("id")
        assert demo_name not in documented_entries, f"demo browser contains a duplicate card for {demo_name}"
        documented_entries[demo_name] = attributes
    missing_images = [
        path for path in re.findall(r'<img src="../../([^"]+)"', demos_page) if not (docs_source / path).is_file()
    ]
    assert not missing_images, f"demo browser references missing images: {missing_images}"

    demo_specs = {spec.program[1]: spec for spec in SPECS if spec.program is not None and spec.program[0] == "demo"}
    assert demo_specs.keys() == documented_entries.keys()
    for demo_name, spec in demo_specs.items():
        entry = documented_entries[demo_name]
        assert set(entry["physics"].split(",")) == {backend for _, backend in spec.physics_backends}, (
            f"{demo_name} documents incorrect physics options"
        )
        # the demo browser offers the visualizer types; omitting --viz is not a menu entry
        assert set(entry["visualizers"].split(",")) == set(spec.visualizers) - {None}, (
            f"{demo_name} documents incorrect visualizer options"
        )


@pytest.fixture(scope="module")
def factory_k10_demo():
    return runpy.run_path(str(script_cases.ROOT / "examples/demos/factory_k10.py"))


def test_factory_k10_demo_does_not_depend_on_training_packages():
    """The exported-policy demo must remain usable without task registration or an RL runner."""
    path = script_cases.ROOT / "examples/demos/factory_k10.py"
    imports = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith(("isaaclab_tasks", "isaaclab_rl", "rsl_rl")) for name in imports)


def test_factory_k10_headless_launch_disables_configured_renderer(factory_k10_demo, monkeypatch):
    """Headless CLI selection must remove the demo's default RTX renderer before building the scene."""
    demo = factory_k10_demo

    class ConfigInspected(Exception):
        pass

    def inspect_config(cfg):
        assert cfg.sim.visualizer_cfgs == []
        assert cfg.sim.device == "cpu"
        assert cfg.scene.num_envs == 2
        raise ConfigInspected

    monkeypatch.setattr(demo["torch"].jit, "load", Mock())
    monkeypatch.setitem(demo["main"].__globals__, "ManagerBasedEnv", inspect_config)
    monkeypatch.setattr(
        sys,
        "argv",
        ["factory_k10.py", "--policy", "actor.pt", "--device", "cpu", "--visualizer", "none", "--num_envs", "2"],
    )
    with pytest.raises(ConfigInspected):
        demo["main"]()


def test_factory_k10_resets_only_finished_boards_before_history_update(factory_k10_demo):
    """Success, invalid state, nonfinite robot state, and timeout preserve continuing episodes."""
    demo = factory_k10_demo
    torch = demo["torch"]
    assembled = torch.zeros((6, 10), dtype=torch.bool)
    assembled[[0, 2]] = True
    geometry = Mock(assembled=assembled, invalid=torch.tensor([False, True, False, False, False, False]))
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.zeros((6, 9))),
            joint_vel=SimpleNamespace(torch=torch.zeros((6, 9))),
        )
    )
    robot.data.joint_vel.torch[2, 0] = float("nan")

    class Scene(dict):
        write_data_to_sim = Mock()

    observations = SimpleNamespace(
        cfg={"slots": SimpleNamespace(geometry=SimpleNamespace(func=geometry, params={}))},
        compute=Mock(side_effect=AssertionError("History must update once, after interval events.")),
    )
    env = SimpleNamespace(
        num_envs=6,
        device="cpu",
        step_dt=0.1,
        scene=Scene(robot=robot),
        observation_manager=observations,
        sim=Mock(),
        reset=Mock(side_effect=AssertionError("Public reset advances continuing histories.")),
    )
    term = demo["EpisodeReset"](demo["EventTermCfg"](), env)
    env._reset_idx = Mock(side_effect=term.reset)
    term.age[:] = torch.tensor([1, 3, 7, 9, 4, 2])
    term(env, None, horizon=1.0)

    assert term.completed == 4 and term.successes == 1
    assert env._reset_idx.call_args.args[0].tolist() == [0, 1, 2, 3]
    assert term.age.tolist() == [0, 0, 0, 0, 5, 3]
    env.scene.write_data_to_sim.assert_called_once_with()
    env.sim.forward.assert_called_once_with()
    env.reset.assert_not_called()
    observations.compute.assert_not_called()

    geometry.assembled.zero_()
    geometry.invalid.zero_()
    robot.data.joint_vel.torch.zero_()
    term(env, None, horizon=1.0)
    assert term.age.tolist() == [1, 1, 1, 1, 6, 4]
    assert term.completed == 4 and term.successes == 1
    assert env._reset_idx.call_count == 1


def test_factory_k10_reset_ranges_and_shared_fixture_geometry(factory_k10_demo):
    """Random resets retain canonical geometry and identities, including gears sharing one fixture."""
    demo = factory_k10_demo
    torch = demo["torch"]
    assemblies = tuple(reversed(demo["ASSEMBLIES"][:10]))
    num_envs, env_ids = 64, torch.arange(1, 64, 2)
    origins = torch.stack((torch.arange(num_envs) % 8, torch.arange(num_envs) // 8, torch.zeros(num_envs)), dim=1) * 2
    board_default = torch.tensor([0.452824, 0.19145, 0.0206, 0.0, 1.0, 0.0, 0.0])

    class Scene(dict):
        env_origins = origins

    scene = Scene()
    names = {"nistboard", *(f"fixed_{part.socket}" for part in assemblies), *(f"held_{slot:02d}" for slot in range(10))}
    for name in names:
        default = board_default if name == "nistboard" else torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        pose = torch.full((num_envs, 7), -99.0)
        scene[name] = SimpleNamespace(
            data=SimpleNamespace(
                default_root_pose=SimpleNamespace(torch=default.repeat(num_envs, 1)),
                root_link_pose_w=SimpleNamespace(torch=pose),
            ),
            write_root_link_pose_to_sim_index=lambda root_pose, env_ids, target=pose: target.__setitem__(
                env_ids, root_pose
            ),
            write_root_com_velocity_to_sim_index=Mock(),
        )
    root = torch.zeros((num_envs, 7))
    root[:, :3], root[:, 6] = origins, 1
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos=SimpleNamespace(torch=torch.zeros((num_envs, 9))),
            root_link_pose_w=SimpleNamespace(torch=root),
            body_link_pose_w=SimpleNamespace(torch=root[:, None]),
        ),
        write_joint_state_to_sim_index=Mock(),
        find_bodies=Mock(return_value=([0], ["panda_hand"])),
    )
    env = SimpleNamespace(scene=scene, num_envs=num_envs, device="cpu")
    torch.manual_seed(1701)
    demo["reset_parts"](env, slice(1, None, 2), assemblies)
    board = scene["nistboard"].data.root_link_pose_w.torch[env_ids].clone()
    board[:, :3] -= origins[env_ids]
    board_rotation = Rotation.from_quat(board[:, 3:].numpy())
    center = board[:, :3].numpy() + board_rotation.apply([0.19232847446867662, -0.191768517928605, 0.0])
    assert np.all(center[:, :2] >= np.array([0.4, -0.03]) - 2e-6)
    assert np.all(center[:, :2] <= np.array([0.5, 0.03]) + 2e-6)
    assert center[:, :2].std(axis=0).min() > 0.008
    np.testing.assert_allclose(center[:, 2], 0.0206, atol=2e-6)
    assert np.ptp((Rotation.from_quat(board_default[3:]).inv() * board_rotation).as_rotvec()[:, 2]) > 4.0

    held = torch.stack([scene[f"held_{slot:02d}"].data.root_link_pose_w.torch[env_ids] for slot in range(10)], dim=1)
    positions = held[..., :3] - origins[env_ids, None]
    assert (positions >= torch.tensor([0.1, -0.5, 0.12]) - 2e-6).all()
    assert (positions <= torch.tensor([0.5, 0.5, 0.14]) + 2e-6).all()
    assert (positions[..., 1].abs() >= 0.3 - 2e-6).all()
    assert ((positions[..., 1] > 0).sum(dim=1) == 5).all()
    assert (positions[..., 1].min(dim=0).values < 0).all() and (positions[..., 1].max(dim=0).values > 0).all()
    distances = torch.cdist(positions[..., :2], positions[..., :2])
    distances[:, torch.arange(10), torch.arange(10)] = float("inf")
    assert distances.min() >= 0.1 - 2e-6
    angles = Rotation.from_quat(held[..., 3:].reshape(-1, 4).numpy()).as_euler("xyz")
    assert np.all(np.abs(angles) <= np.array([1.57, 1.57, 3.14]) + 2e-6)

    first = {name: scene[name].data.root_link_pose_w.torch.clone() for name in names}
    demo["reset_parts"](env, env_ids, assemblies)
    for name, initial in first.items():
        current = scene[name].data.root_link_pose_w.torch
        assert not torch.equal(current[env_ids], initial[env_ids])
        assert (current[::2] == -99).all()
    torch.manual_seed(1701)
    demo["reset_parts"](env, env_ids, assemblies)
    for name, initial in first.items():
        torch.testing.assert_close(scene[name].data.root_link_pose_w.torch, initial, atol=0, rtol=0)

    for slot, part in enumerate(assemblies):
        fixed = scene[f"fixed_{part.socket}"].data.root_link_pose_w.torch[env_ids].clone()
        fixed[:, :3] -= origins[env_ids]
        np.testing.assert_allclose(
            board_rotation.inv().apply((fixed[:, :3] - board[:, :3]).numpy()),
            np.tile(part.board_pose[:3], (len(env_ids), 1)),
            atol=2e-6,
        )
        fixed_rotation = Rotation.from_quat(fixed[:, 3:].numpy())
        relative_rotation = board_rotation.inv() * fixed_rotation
        assert (relative_rotation.inv() * Rotation.from_quat(part.board_pose[3:])).magnitude().max() < 2e-6
        goal = fixed[:, :3].numpy() + fixed_rotation.apply(np.array(part.seated) - part.held_align)
        assert np.all(goal >= np.array(demo["WORKSPACE_LOWER"])) and np.all(goal <= np.array(demo["WORKSPACE_UPPER"]))
        pose = scene[f"held_{slot:02d}"].data.root_link_pose_w.torch
        pose[env_ids, :3] = torch.from_numpy(goal).float() + origins[env_ids]
        pose[env_ids, 3:] = fixed[:, 3:]
    geometry = demo["SlotGeometry"](demo["ObservationTermCfg"](params={"assemblies": assemblies}), env)
    slots = geometry(env, assemblies)
    assert slots.shape == (num_envs, 10, 40)
    assert slots[env_ids, :, 21:].argmax(dim=-1).tolist() == [[part.identity for part in assemblies]] * len(env_ids)
    assert geometry.assembled[env_ids].all()


# Launch matrix selection.


def test_runtime_groups_partition_matrix_without_overlap():
    """Kit and non-Kit groups must cover every launch case exactly once."""
    cases = build_cases(SPECS)
    grouped_ids = [
        case.id for runtime_group in ("kit", "non-kit") for case in select_runtime_group(cases, runtime_group)
    ]
    assert sorted(grouped_ids) == sorted(case.id for case in cases)
    with pytest.raises(ValueError, match="runtime group"):
        select_runtime_group(cases, "invalid")


def test_script_scope_selects_matching_scripts_and_rejects_empty_selection():
    """Scopes select their scripts, and a stale or misspelled scope must not produce a vacuously green matrix."""
    assert {spec.path for spec in select_script_scope(SPECS, "demos")} == set(script_cases.PROGRAMS_BY_PATH)
    assert {spec.path for spec in select_script_scope(SPECS, "examples/demos")} == {
        spec.path for spec in SPECS if spec.program is not None and spec.program[0] == "demo"
    }
    assert all(spec.relative_path.startswith("examples/mpm/") for spec in select_script_scope(SPECS, "examples/mpm"))
    with pytest.raises(ValueError, match="selected no scripts"):
        select_script_scope(SPECS, "missing")


def test_ast_discovery_recognizes_main_guards_and_literal_choices():
    """Static discovery must distinguish executable scripts and preserve literal choices."""
    tree = ast.parse(
        """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--physics', choices=['physx', 'newton_mjwarp'])
parser.add_argument('--viz', choices=('kit',))
parser.add_argument('positional')
if __name__ == '__main__':
    pass
"""
    )
    assert script_cases._has_main_guard(tree)
    assert script_cases._literal_cli_options(tree) == {"--physics": ("physx", "newton_mjwarp"), "--viz": ("kit",)}
    assert not script_cases._has_main_guard(ast.parse("print('library module')"))


# Process supervision.


def test_supervisor_soaks_then_stops_process_group():
    """A script that stays healthy after readiness is stopped once the soak ends."""
    result = run_until_ready(_python(_READY_THEN_SLEEP), r"READY", startup_timeout=2.0, soak_time=0.05)
    assert result.ready
    assert result.stopped_after_soak
    assert result.elapsed < 2.0


def test_supervisor_ignores_fatal_output_caused_by_its_own_teardown():
    """Errors printed while the supervisor stops a healthy script must not fail the launch."""
    source = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: (print('Traceback (most recent call last):'), sys.exit(1)))\n"
        "print('READY')\n"
        "time.sleep(30)\n"
    )
    result = run_until_ready(_python(source), r"READY", startup_timeout=5.0, soak_time=0.2)
    assert result.stopped_after_soak
    assert "Traceback (most recent call last):" in result.output
    assert not result.fatal_patterns


def test_supervisor_requires_readiness_before_exit():
    """A successful exit is insufficient for a soaked script that never printed its readiness marker."""
    result = run_until_ready(_python("print('not ready')"), r"READY", startup_timeout=2.0)
    assert not result.ready
    assert result.returncode == 0


def test_supervisor_runs_finite_scripts_to_exit_and_reports_shutdown_errors():
    """Without a readiness marker the script runs to completion, and errors during its shutdown are fatal."""
    clean = run_until_ready(_python("print('stepping')"), None, startup_timeout=5.0)
    assert clean.returncode == 0
    assert not clean.fatal_patterns

    crashing_shutdown = run_until_ready(
        _python("print('stepping'); print('Fatal Python error: during teardown')"), None, startup_timeout=5.0
    )
    assert crashing_shutdown.returncode == 0
    assert "Fatal Python error:" in crashing_shutdown.fatal_patterns


@pytest.mark.parametrize("readiness_pattern", [r"READY", None], ids=["soaked", "finite"])
def test_supervisor_bounds_run_time(readiness_pattern):
    """A script that never becomes ready or never exits is terminated at the time limit."""
    result = run_until_ready(_python("import time; time.sleep(30)"), readiness_pattern, startup_timeout=0.05)
    assert not result.ready
    assert result.elapsed < 2.0
    assert result.returncode not in (None, 0)


def test_supervisor_retains_fatal_state_when_output_is_truncated(monkeypatch):
    """Fatal output must remain detectable after the bounded output tail rolls over."""
    monkeypatch.setattr(script_cases, "MAX_OUTPUT_BYTES", 64)
    source = (
        "print('Traceback (most recent call last):'); "
        "print('Number of Newton contacts (10) exceeded MJWarp limit (2). Increase nconmax.'); "
        "print('nefc overflow - please increase njmax to 10'); "
        "print('x' * 1024); print('READY')"
    )
    result = run_until_ready(_python(source), r"READY", startup_timeout=2.0)
    assert result.ready
    assert len(result.output.encode()) <= 64
    assert {"Traceback (most recent call last):", "exceeded MJWarp limit", "nefc overflow"} <= set(
        result.fatal_patterns
    )


def test_supervisor_captures_one_requested_screenshot(monkeypatch, tmp_path):
    """A healthy visual launch triggers exactly one screenshot during its soak."""
    captured = []
    monkeypatch.setattr(script_cases, "_capture_screenshot", captured.append)
    screenshot_path = tmp_path / "launch.png"
    result = run_until_ready(
        _python(_READY_THEN_SLEEP), r"READY", startup_timeout=2.0, soak_time=0.1, screenshot_path=screenshot_path
    )
    assert result.ready
    assert captured == [screenshot_path]


def test_smoke_assertion_rejects_each_failure_mode():
    """Fatal output, missing readiness, and early or unclean exits must all fail a launch."""
    soaked = next(case for case in build_cases(SPECS) if not case.spec.finite and case.spec.readiness_pattern)
    finite = next(case for case in build_cases(SPECS) if case.spec.finite)
    for case in (soaked, finite):
        with pytest.raises(AssertionError, match="fatal output"):
            assert_smoke_passed(SmokeResult(True, 0, "tail", 0.1, False, ("Fatal Python error:",)), case)
    with pytest.raises(AssertionError, match="did not reach"):
        assert_smoke_passed(SmokeResult(False, 0, "tail", 0.1, False), soaked)
    with pytest.raises(AssertionError, match="exited with 2"):
        assert_smoke_passed(SmokeResult(True, 2, "tail", 0.1, False), soaked)
    with pytest.raises(AssertionError, match="did not exit cleanly"):
        assert_smoke_passed(SmokeResult(False, -15, "tail", 300.0, False), finite)
    assert_smoke_passed(SmokeResult(False, 0, "tail", 0.1, False), finite)


# Simulator launches.


def _skip_unsupported(case) -> None:
    """Skip a launch the current machine or the script's declared contract cannot run."""
    if not RUN_LAUNCH_MATRIX:
        pytest.skip("set ISAACLAB_RUN_STANDALONE_SCRIPT_TESTS=1 to run the launch matrix")
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    missing_modules = [module for module in case.spec.required_modules if not script_cases.module_is_available(module)]
    if missing_modules:
        pytest.skip(f"required runtime module(s) not installed: {', '.join(missing_modules)}")
    if case.visualizer in {"kit", "newton", "newton_gl", "newton_rtx"} and not gui_is_available():
        pytest.skip("GUI smoke test requires DISPLAY or WAYLAND_DISPLAY")
    if not backend_is_available(case.physics_backend):
        pytest.skip(f"physics backend package for {case.physics_backend!r} is not installed")
    if not backend_is_available(case.renderer_backend):
        pytest.skip(f"renderer backend package for {case.renderer_backend!r} is not installed")
    if not visualizer_is_available(case.visualizer):
        pytest.skip(f"visualizer package for {case.visualizer!r} is not installed")


def _launch(case, *extra_args: str, screenshot_path: Path | None = None) -> None:
    """Launch *case* and assert it ran without a fatal error."""
    result = run_until_ready(
        [*case.command(), *extra_args],
        None if case.spec.finite else case.spec.readiness_pattern,
        startup_timeout=max(STARTUP_TIMEOUT, case.spec.startup_timeout or 0.0),
        soak_time=SOAK_TIME,
        screenshot_path=screenshot_path,
        screenshot_delay=SCREENSHOT_DELAY,
    )
    assert_smoke_passed(result, case)


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.smoke
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_standalone_script_runs_cleanly(case):
    """Each supported script launch must initialize, step, and stop without a fatal error."""
    _skip_unsupported(case)
    screenshot_path = None
    if SCREENSHOT_DIR and not case.spec.finite and case.visualizer is not None and gui_is_available():
        screenshot_path = Path(SCREENSHOT_DIR) / f"{case.id}.png"
    _launch(case, screenshot_path=screenshot_path)


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.smoke
@pytest.mark.parametrize("asset_type", ("anymal_d", "objects"))
@pytest.mark.parametrize("case", MULTI_MESH_RAYCASTER_CASES, ids=lambda case: case.physics_backend)
def test_multi_mesh_raycaster_supports_each_asset_type(case, asset_type):
    """The multi-mesh raycaster must support every non-default asset path on each physics backend."""
    _skip_unsupported(case)
    _launch(case, "--asset_type", asset_type)


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.smoke
@pytest.mark.parametrize("case", NEWTON_RAYCAST_CASES, ids=lambda case: case.physics_backend)
def test_newton_raycast_supports_moving_geometry(case):
    """The consolidated Newton ray-cast example must launch its non-default scene."""
    _skip_unsupported(case)
    _launch(case, "--scene", "moving-geometry")
