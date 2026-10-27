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
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

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


def test_haply_scene_config_stays_lazy_until_scene_construction():
    """The Haply module must configure its robot without constructing an articulation."""
    script = script_cases.ROOT / "examples" / "haply_teleoperation.py"
    code = """
import runpy
import sys

from isaaclab.assets import ArticulationCfg

script = sys.argv[1]
sys.argv = [script, '--physics', 'newton_mjwarp', '--visualizer', 'none']
namespace = runpy.run_path(script)
scene_cfg = namespace['FrankaHaplySceneCfg'](num_envs=1, env_spacing=2.0)
assert isinstance(scene_cfg.robot, ArticulationCfg)
assert scene_cfg.robot.init_state.pos == (-0.02, 0.0, 1.05)
assert scene_cfg.robot.spawn.activate_contact_sensors is True
assert scene_cfg.robot.spawn.variants['Physics'] == 'mujoco'
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "script_path",
    [
        "scripts/tutorials/08_mpm/material_parameters.py",
        "scripts/tutorials/08_mpm/rigid_body_equivalence.py",
        "scripts/tutorials/08_mpm/surface_reconstruction.py",
        "examples/demos/snowball_smash.py",
        "examples/mpm/newton_mpm_granular.py",
        "examples/mpm/newton_mpm_twoway_coupling.py",
        "examples/mpm/g1_coupling.py",
    ],
)
def test_mpm_programs_preserve_concurrent_visualizer_selection(script_path):
    """One shared declaration selects both Newton viewers before launching the runtime."""
    pytest.importorskip("isaaclab_newton")
    pytest.importorskip("isaaclab_visualizers")
    pytest.importorskip("isaaclab_tasks")
    if script_path.endswith("g1_coupling.py"):
        pytest.importorskip("isaaclab_contrib")
    script = script_cases.ROOT / script_path
    code = """
import runpy
import sys

import isaaclab.app
from isaaclab.sim import SimulationCfg
from isaaclab.visualizers import VisualizerCfg
from isaaclab_tasks.utils.hydra import collect_presets

def check_launch(cfg, args):
    assert not collect_presets(cfg), "Preset composition must finish before core launch"
    simulation = cfg if isinstance(cfg, SimulationCfg) else cfg.sim
    shared = simulation.visualizer_cfgs
    if isinstance(shared, list):
        shared, = shared
    assert isinstance(shared, VisualizerCfg)
    assert [backend.visualizer_type for backend in shared.backend] == ['newton_gl', 'newton_rtx']
    configs, _ = isaaclab.app.validate_launch(cfg, args)
    viewers = [cfg for cfg in configs if isinstance(cfg, VisualizerCfg)]
    assert [cfg.backend.visualizer_type for cfg in viewers] == ['newton_gl', 'newton_rtx']
    assert all(viewer.eye == shared.eye and viewer.focal_length == shared.focal_length for viewer in viewers)
    assert viewers[0].backend.show_particles
    if script.endswith('surface_reconstruction.py'):
        assert viewers[1].backend.render_settings['omni:rtx:quality'] == ('Int', 2)
    raise SystemExit(0)

isaaclab.app.launch_simulation = check_launch
script = sys.argv[1]
sys.argv = [script, '--visualizer', 'newton_gl,newton_rtx']
if script.endswith('surface_reconstruction.py'):
    sys.argv.extend(['--fluid_render_mode', 'both', '--rtx_quality', '2'])
else:
    sys.argv.extend(['--voxel_size', '0.1'])
runpy.run_path(script, run_name='__main__')
raise AssertionError('The program did not reach launch_simulation')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_teapot_uses_only_newton_gl():
    """The GL-only demo rejects other backends before launch and needs no task presets."""
    script = script_cases.ROOT / "examples/demos/teapot_fill.py"
    code = """
import runpy
import sys

import isaaclab.app
from isaaclab_visualizers.newton import NewtonGLBackendCfg

def check_launch(cfg, args):
    assert isinstance(cfg.visualizer_cfgs.backend, NewtonGLBackendCfg)
    assert cfg.visualizer_cfgs.backend.show_particles
    assert 'isaaclab_tasks' not in sys.modules
    isaaclab.app.validate_launch(cfg, args)
    assert [view.backend.visualizer_type for view in cfg.visualizer_cfgs] == ['newton_gl']
    raise SystemExit(0)

isaaclab.app.launch_simulation = check_launch
script = sys.argv[1]
sys.argv = [script, '--visualizer', 'newton_gl', '--fluid_render_mode', 'both']
namespace = runpy.run_path(script)
parser = namespace['parser']
for option, value in (
    ('--visualizer', 'newton_rtx'), ('--viz', 'kit'), ('--viz', 'newton_gl,rerun'), ('--viz', 'none')
):
    try:
        parser.parse_args([option, value])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError(f'Teapot accepted unsupported visualizer: {value}')
assert parser.parse_args([]).visualizer == 'newton_gl'
assert parser.parse_args(['--viz', 'newton_gl']).visualizer == 'newton_gl'
namespace['main']()
raise AssertionError('The demo did not reach launch_simulation')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mpm_surface_reconstruction_defaults_to_newton_gl():
    """The surface example must launch with base Newton dependencies by default."""
    script = script_cases.ROOT / "scripts" / "tutorials" / "08_mpm" / "surface_reconstruction.py"
    code = """
import runpy
import sys

script = sys.argv[1]
sys.argv = [script]
namespace = runpy.run_path(script)
assert namespace['args_cli'].visualizer == ['newton_gl']
sys.argv = [script, '--visualizer', 'newton_rtx']
namespace = runpy.run_path(script)
assert namespace['args_cli'].visualizer == ['newton_rtx']
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "script_path",
    [
        "examples/mpm/newton_mpm_granular.py",
        "scripts/tutorials/08_mpm/material_parameters.py",
        "scripts/tutorials/08_mpm/surface_reconstruction.py",
    ],
)
def test_mpm_demos_use_solver_contact_response(script_path):
    """Physical-response examples must not opt into post-step particle projection."""
    pytest.importorskip("isaaclab_newton")
    script = script_cases.ROOT / script_path
    code = """
import runpy
import sys

script = sys.argv[1]
sys.argv = [script, '--visualizer', 'none']
if script.endswith('surface_reconstruction.py'):
    sys.argv.extend(['--fluid_render_mode', 'particles'])
namespace = runpy.run_path(script)
config = namespace['create_sim_cfg']()
assert config.physics.solver_cfg.project_outside_colliders is False
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


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


@pytest.mark.parametrize(
    ("extra_args", "expected_visualizers", "expected_overrides"),
    [
        ((), ["newton_gl"], ["presets=platforms"]),
        (("physics=isaacsim_physx",), ["kit"], ["presets=platforms", "physics=isaacsim_physx"]),
        (("--viz", "none"), [], ["presets=platforms", "visualizer=none"]),
    ],
    ids=("newton-default", "physx-default", "explicit-none"),
)
def test_g1_terrain_forwards_visualizer_selection(monkeypatch, extra_args, expected_visualizers, expected_overrides):
    """The G1 example must choose its backend default without overriding explicit visualizer intent."""
    script = script_cases.ROOT / "examples" / "g1_terrain.py"
    namespace = runpy.run_path(str(script))
    main = namespace["main"]
    resolve_config = main.__globals__["resolve_task_config"]
    captured = {}

    def resolve_task_config(_task, _agent, *, overrides):
        captured["overrides"] = overrides
        return resolve_config(_task, _agent, overrides=overrides)

    class LaunchReached(Exception):
        pass

    @contextmanager
    def launch_simulation(_cfg, launcher_args):
        captured["visualizers"] = [
            backend.visualizer_type
            for viewer in _cfg.sim.visualizer_cfgs
            for backend in (viewer.backend if isinstance(viewer.backend, list) else [viewer.backend])
        ]
        raise LaunchReached
        yield

    monkeypatch.setitem(main.__globals__, "resolve_task_config", resolve_task_config)
    monkeypatch.setitem(main.__globals__, "launch_simulation", launch_simulation)
    monkeypatch.setattr(main.__globals__["torch"], "set_num_threads", lambda _count: None)
    monkeypatch.setattr(sys, "argv", [str(script), "--task", "platforms", *extra_args])

    with pytest.raises(LaunchReached):
        main()

    assert captured == {"overrides": expected_overrides, "visualizers": expected_visualizers}


@pytest.mark.parametrize(
    ("extra_args", "expected_visualizers", "expected_overrides", "expected_video_source"),
    [
        ((), [], ["presets=granular"], "viz:kit"),
        (("--viz", "newton_gl"), ["newton_gl"], ["presets=granular", "visualizer=newton_gl"], None),
        (("--viz", "none"), [], ["presets=granular", "visualizer=none"], None),
        (("--viz", "rerun", "--video"), ["rerun"], ["presets=granular", "visualizer=rerun"], "viz"),
        (("physics=isaacsim_physx",), ["kit"], ["presets=granular", "physics=isaacsim_physx"], None),
        (
            ("--physics", "isaacsim_physx", "--video"),
            ["kit"],
            ["presets=granular", "physics=isaacsim_physx"],
            "viz:kit",
        ),
    ],
    ids=(
        "newton-default",
        "explicit-gl",
        "explicit-none",
        "streaming-viz-video",
        "physx-default",
        "physx-default-video",
    ),
)
def test_sonic_defaults_to_offscreen_recording_and_preserves_explicit_visualizers(
    monkeypatch, extra_args, expected_visualizers, expected_overrides, expected_video_source
):
    """Newton records with Kit by default; explicit viewers and PhysX keep recording opt-in."""
    script = script_cases.ROOT / "examples" / "demos" / "sonic.py"
    namespace = runpy.run_path(str(script))
    main = namespace["main"]
    resolve_config = main.__globals__["resolve_task_config"]
    captured = {}

    def resolve_task_config(_task, _agent, *, overrides):
        captured["overrides"] = overrides
        return resolve_config(_task, _agent, overrides=overrides)

    class LaunchReached(Exception):
        pass

    @contextmanager
    def launch_simulation(_cfg, launcher_args):
        captured["visualizers"] = [
            backend.visualizer_type
            for viewer in _cfg.sim.visualizer_cfgs
            for backend in (viewer.backend if isinstance(viewer.backend, list) else [viewer.backend])
        ]
        captured["launch_physics"] = getattr(launcher_args, "physics", None)
        captured["video_sources"] = [recorder.source for recorder in _cfg.video_recorders]
        raise LaunchReached
        yield

    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = lambda **_kwargs: "models"
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setitem(main.__globals__, "resolve_task_config", resolve_task_config)
    monkeypatch.setitem(main.__globals__, "launch_simulation", launch_simulation)
    monkeypatch.setattr(main.__globals__["torch"], "set_num_threads", lambda _count: None)
    monkeypatch.setattr(sys, "argv", [str(script), *extra_args])

    with pytest.raises(LaunchReached):
        main()

    expected_video_sources = [] if expected_video_source is None else [expected_video_source]
    assert captured == {
        "overrides": expected_overrides,
        "visualizers": expected_visualizers,
        "launch_physics": None,
        "video_sources": expected_video_sources,
    }


def test_demo_browser_documents_options_for_each_demo():
    """Every demo card must match a demo and offer exactly the options the demo accepts."""
    docs_source = script_cases.ROOT / "docs/source"
    demos_page = (docs_source / "isaaclab/demos.rst").read_text(encoding="utf-8")
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
def nist_demo():
    return runpy.run_path(str(script_cases.ROOT / "examples/demos/nist.py"))


def test_nist_demo_does_not_depend_on_training_packages():
    """The demo keeps one composition root and leaves episode resets to the RL environment."""
    path = script_cases.ROOT / "examples/demos/nist.py"
    nodes = list(ast.walk(ast.parse(path.read_text(encoding="utf-8"))))
    imports = []
    for node in nodes:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith(("isaaclab_tasks", "isaaclab_rl", "rsl_rl")) for name in imports)
    assert not any(isinstance(node, ast.Attribute) and node.attr == "_reset_idx" for node in nodes)


def test_nist_reset_samples_separated_loose_parts_in_square(nist_demo):
    """Partial resets place separated parts across a square in their own environment."""
    demo = nist_demo
    torch = demo["torch"]
    assemblies = demo["ASSEMBLIES"][:10]
    num_envs, env_ids = 64, torch.arange(1, 64, 2)

    class Scene(dict):
        env_origins = torch.arange(num_envs)[:, None].expand(-1, 3) * 2.0

    names = {"nistboard", *(f"fixed_{part.socket}" for part in assemblies), *(f"held_{slot:02d}" for slot in range(10))}
    scene = Scene({name: Mock() for name in names | {"robot"}})
    for name in names:
        scene[name].data.default_root_pose.torch = torch.tensor([0.0, 0, 0, 0, 0, 0, 1]).repeat(num_envs, 1)
    scene["robot"].data.default_joint_pos.torch = torch.zeros((num_envs, 9))
    board_pose = torch.tensor([0.452824, 0.19145, 0.0206, 0, 1, 0, 0])
    scene["nistboard"].data.default_root_pose.torch = board_pose.repeat(num_envs, 1)
    env = SimpleNamespace(scene=scene, num_envs=num_envs, device="cpu")
    torch.manual_seed(1701)
    demo["reset_parts"](env, slice(1, None, 2), assemblies)
    writes = [scene[f"held_{slot:02d}"].write_root_link_pose_to_sim_index.call_args.kwargs for slot in range(10)]
    assert all(torch.equal(write["env_ids"], env_ids) for write in writes)
    positions = torch.stack([write["root_pose"][:, :3] for write in writes], dim=1) - scene.env_origins[env_ids, None]
    assert (positions >= torch.tensor([0.15, -0.35, 0.12]) - 1e-5).all()
    assert (positions <= torch.tensor([0.85, 0.35, 0.14]) + 1e-5).all()
    assert (positions[..., 1].abs() < 0.2).any()
    distances = torch.cdist(positions[..., :2], positions[..., :2])
    distances[:, torch.arange(10), torch.arange(10)] = float("inf")
    assert distances.min() >= 0.1 - 1e-5


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
