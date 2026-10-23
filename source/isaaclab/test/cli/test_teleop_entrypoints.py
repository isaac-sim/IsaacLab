# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the unified teleoperation console entry point."""

import argparse
import re
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest
import tomllib

import isaaclab.cli as cli

pytestmark = pytest.mark.unit

# Each teleop workflow and the repository-relative script it forwards to.
TELEOP_WORKFLOWS = {
    "run": ("scripts", "environments", "teleoperation", "teleop_se3_agent.py"),
    "record": ("scripts", "tools", "record_demos.py"),
    "replay": ("scripts", "tools", "replay_demos.py"),
}


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_workflow_help_exposes_task_preset_selectors(source_checkout_root: Path, command, script_parts):
    """Every teleop workflow accepts the task preset selectors documented for teleoperation."""
    script = source_checkout_root.joinpath(*script_parts)

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=source_checkout_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "preset selection:" in result.stdout
    assert "physics=NAME" in result.stdout
    assert "renderer=NAME" in result.stdout
    assert "presets=NAME[,NAME,...]" in result.stdout
    if command == "run":
        assert re.search(r"^\s+--ovxr(?:\s|$)", result.stdout, re.MULTILINE)
        assert not re.search(r"^\s+--kitless_xr(?:[,\s]|$)", result.stdout, re.MULTILINE)
        assert "--ovxr_pip_camera" in result.stdout
        assert "--ovxr_desktop_mirror" in result.stdout
        assert "--kitless_xr_" not in result.stdout


def test_ovxr_omitted_launch_options_match_documented_defaults(source_checkout_root, monkeypatch):
    """The shortened user command retains the profile, launch policy and visualizers."""
    from isaaclab.app import validate_launch

    monkeypatch.delenv("LIVESTREAM", raising=False)
    script = source_checkout_root / "scripts/environments/teleoperation/teleop_se3_agent.py"

    def parse(arguments):
        monkeypatch.setattr(sys, "argv", [str(script), *arguments])
        return runpy.run_path(str(script))["args_cli"]

    implicit = parse(["--ovxr"])
    explicit = parse(["--ovxr", "--viz", "none", "--auto_launch_cloudxr", "--cloudxr_env", "cloudxrjs"])
    assert implicit.auto_launch_cloudxr is explicit.auto_launch_cloudxr is True
    assert implicit.cloudxr_env == explicit.cloudxr_env
    assert implicit.cloudxr_env.endswith(".env")
    left, right = vars(implicit).copy(), vars(explicit).copy()
    from isaaclab.sim import SimulationCfg

    left_cfg, right_cfg = SimulationCfg(), SimulationCfg()
    validate_launch(left_cfg, left)
    validate_launch(right_cfg, right)
    assert left_cfg.visualizer_cfgs == right_cfg.visualizer_cfgs == []
    assert parse(["--ovxr", "--no-auto_launch_cloudxr"]).auto_launch_cloudxr is False
    assert parse(["--ovxr", "--cloudxr_env", "none"]).cloudxr_env is None


def test_ovxr_output_options_reach_existing_runtime_fields(source_checkout_root, monkeypatch):
    """Camera options configure panels, the desktop mirror and XR startup."""
    options = [
        "--ovxr",
        "--ovxr_session_mode",
        "vr",
        "--ovxr_bootstrap_frames",
        "2",
        "--ovxr_worker_timeout",
        "60",
        "--ovxr_desktop_mirror",
        "--ovxr_desktop_mirror_fps",
        "20",
        "--ovxr_pip_camera",
        "robot_pov_cam",
        "--ovxr_pip_fps",
        "10",
        "--ovxr_pip_head_locked",
        "--ovxr_pip_view_position",
        "0",
        "0.2",
        "-0.5",
    ]
    script = source_checkout_root / "scripts/environments/teleoperation/teleop_se3_agent.py"
    monkeypatch.setattr(sys, "argv", [str(script), *options])
    parsed = runpy.run_path(str(script))["args_cli"]
    assert parsed.using_ovxr and parsed.ovxr_session_mode == "vr"
    assert (parsed.ovxr_bootstrap_frames, parsed.ovxr_worker_timeout) == (2, 60)
    assert parsed.ovxr_desktop_mirror
    assert parsed.ovxr_desktop_mirror_fps == 20
    assert parsed.ovxr_pip_camera == ["robot_pov_cam"] and parsed.ovxr_pip_fps == 10
    assert parsed.ovxr_pip_head_locked and parsed.ovxr_pip_view_position == [0, 0.2, -0.5]


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_dispatches_to_an_existing_script(source_checkout_root: Path, command, script_parts):
    """``isaaclab teleop`` forwards the remaining arguments to a script that exists."""
    selector = "--ovxr" if command == "run" else "--xr"
    args = [command, "--task", "IsaacContrib-PickPlace-Locomanipulation-G1-Abs", selector]

    with (
        mock.patch.object(sys, "argv", ["isaaclab", "teleop", *args]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
    ):
        cli.cli()

    script = source_checkout_root.joinpath(*script_parts)
    run_python.assert_called_once_with(script, args[1:], check=True)
    assert script.is_file()


@pytest.mark.parametrize("script_parts", TELEOP_WORKFLOWS.values(), ids=TELEOP_WORKFLOWS)
def test_teleop_callback_preserves_selector_aliases(source_checkout_root: Path, script_parts, monkeypatch):
    """Callback-owned flags are removed without losing task or viewer selections."""
    callback = ModuleType("_teleop_callback")

    def register():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--callback_option")
        args, remaining = parser.parse_known_args()
        assert args.callback_option == "42"
        return remaining

    callback.register = register
    monkeypatch.setitem(sys.modules, callback.__name__, callback)
    script = source_checkout_root.joinpath(*script_parts)
    monkeypatch.setattr(sys, "argv", [
        str(script), "--task", "Isaac-Cartpole-Direct", "--external_callback", "_teleop_callback.register",
        "--callback_option", "42", "--viz", "newton_gl", "--physics=newton_mjwarp", "--renderer", "newton_renderer",
        "env.sim.dt=0.01",
    ])  # fmt: skip
    namespace = runpy.run_path(str(script))
    assert namespace["args_cli"].task == "Isaac-Cartpole-Direct"
    assert set(sys.argv[1:]) == {
        "visualizer=newton_gl",
        "physics=newton_mjwarp",
        "renderer=newton_renderer",
        "env.sim.dt=0.01",
    }


def _requirement_names(requirements: list[str]) -> set[str]:
    """Return the importable module names for a list of requirement strings."""
    names = set()
    for requirement in requirements:
        name = re.split(r"[\s<>=!~\[;@]", requirement, maxsplit=1)[0]
        if name.startswith("isaaclab"):
            names.add(name.replace("-", "_"))
    return names


@pytest.mark.parametrize(("command", "script_parts"), TELEOP_WORKFLOWS.items())
def test_teleop_workflow_isaaclab_imports_are_covered_by_the_extras(source_checkout_root: Path, command, script_parts):
    """Every ``isaaclab_*`` package a teleop script imports must ship with the extras.

    ``record_demos.py`` imports ``isaaclab_mimic`` at module level, so an environment built
    from ``--extra teleop`` alone used to die with ``ModuleNotFoundError`` only after Isaac Sim
    had finished booting. This catches that class of gap without needing an install.
    """
    with (source_checkout_root / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    project = pyproject["project"]

    available = _requirement_names(project["dependencies"])
    available |= _requirement_names(project["optional-dependencies"]["teleop"])

    source = source_checkout_root.joinpath(*script_parts).read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:import|from)\s+(isaaclab_\w+)", source, re.MULTILINE))

    missing = imported - available
    assert not missing, f"'isaaclab teleop {command}' imports {sorted(missing)}, absent from the teleop extra"


def test_teleop_rejects_unknown_workflow():
    """An unknown workflow name fails at the sub-parser instead of launching a script."""
    with (
        mock.patch.object(sys, "argv", ["isaaclab", "teleop", "bogus"]),
        mock.patch("isaaclab.cli.run_python_command") as run_python,
        pytest.raises(SystemExit),
    ):
        cli.cli()

    run_python.assert_not_called()
