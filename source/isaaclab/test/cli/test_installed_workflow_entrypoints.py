# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for workflow commands exposed by an installed ``isaaclab`` package."""

from __future__ import annotations

import subprocess
import sys
from unittest import mock

import pytest

import isaaclab
import isaaclab.__main__ as package_main
import isaaclab._programs as programs
import isaaclab.cli as cli
import isaaclab.paths as paths

pytestmark = pytest.mark.unit


def test_cli_import_does_not_require_runtime_dependencies():
    """The installation CLI must load before core runtime dependencies are installed."""
    result = subprocess.run(
        [sys.executable, "-c", 'import sys; sys.modules["lazy_loader"] = None; import isaaclab.cli'],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_resolves_partial_source_checkout_root(tmp_path):
    """Source root resolution must not require resources copied by later Docker layers."""
    package_root = tmp_path / "source" / "isaaclab" / "isaaclab"
    package_root.mkdir(parents=True)

    with mock.patch.object(paths, "__file__", str(package_root / "paths.py")):
        assert paths._resolve_isaaclab_root() == tmp_path


def test_top_level_compatibility_api_is_preserved():
    """The flattened package must retain the aggregate wheel's public shims."""
    assert callable(isaaclab.bootstrap_kernel)
    with mock.patch.object(package_main, "main", return_value=0) as main, pytest.raises(SystemExit, match="0"):
        isaaclab.main()

    main.assert_called_once_with()


def test_editor_option_uses_cli_dispatcher():
    """The installed CLI must forward editor-specific arguments to the editor command."""
    with (
        mock.patch.object(sys, "argv", ["isaaclab", "--editor", "--isaac_path", "/sim", "--verbose"]),
        mock.patch.object(cli, "command_editor") as editor,
    ):
        cli.cli()

    editor.assert_called_once_with(["--isaac_path", "/sim", "--verbose"])


@pytest.mark.parametrize("option", ["--vscode", "--generate-vscode-settings"])
def test_removed_editor_options_are_rejected(option):
    """Removed editor setup options must not remain as hidden compatibility paths."""
    with mock.patch.object(sys, "argv", ["isaaclab", option]), pytest.raises(SystemExit, match="2"):
        cli.cli()


@pytest.mark.parametrize(
    ("command", "runner"),
    [
        (cli.train, "run_train_cli"),
        (cli.play, "run_play_cli"),
        (cli.train_multigpu, "run_train_multigpu_cli"),
        (cli.zero_agent, "run_zero_agent_cli"),
        (cli.random_agent, "run_random_agent_cli"),
    ],
)
def test_workflow_commands_dispatch_to_installed_entrypoints(command, runner):
    """Workflow commands must not depend on scripts from a source checkout."""
    args = ["--task", "Example"]
    with mock.patch(f"isaaclab_rl.entrypoints.{runner}", return_value=0) as run:
        command(args)

    run.assert_called_once_with(args)


def test_workflow_command_propagates_failure_status():
    """A nonzero in-process result must remain the console command's exit status."""
    with mock.patch("isaaclab_rl.entrypoints.run_train_cli", return_value=2), pytest.raises(SystemExit, match="2"):
        cli.train([])


def test_demo_catalog_lists_packaged_demos(capsys):
    """The demo command must expose stable names without importing simulator modules."""
    cli.demo(["list"])

    output = capsys.readouterr().out
    assert "zoo" in output
    assert "teapot-fill" in output
    assert "bin-packing" not in output


def test_example_catalog_lists_packaged_examples(capsys):
    """The example command must distinguish focused programs from showcases."""
    cli.example(["list"])

    output = capsys.readouterr().out
    assert "bin-packing" in output
    assert "mpm-two-way-coupling" in output
    assert "uvx --from 'isaaclab[isaacsim]' isaaclab example camera" in output
    assert "teapot-fill" not in output


@pytest.mark.parametrize("catalog", [programs.DEMOS, programs.EXAMPLES])
def test_program_catalog_resolves_modules(catalog):
    """Every core catalog entry must resolve to a packaged Python module."""
    core_programs = (program for program in catalog if program.module.startswith("isaaclab."))
    assert all(programs.find_spec(program.module) is not None for program in core_programs)


def test_owned_examples_resolve_from_their_packages():
    """Integration examples must stay with the package that owns their implementation."""
    modules_by_name = {program.name: program.module for program in programs.EXAMPLES}
    assert modules_by_name["arl-robot-1"] == "isaaclab_contrib.examples.arl_robot_1"
    assert modules_by_name["haply-teleoperation"] == "isaaclab_teleop.examples.haply_teleoperation"
    assert modules_by_name["ppisp-camera"] == "isaaclab_ppisp.examples.ppisp_camera"
    assert modules_by_name["tactile-sensor"] == "isaaclab_contrib.examples.tacsl_sensor"


def test_newton_raycast_scenes_share_one_example():
    """Newton ray-cast variants must stay behind one focused example entry."""
    modules_by_name = {program.name: program.module for program in programs.EXAMPLES}
    assert modules_by_name["newton-raycast"] == "isaaclab.examples.sensors.newton_raycast"
    assert "newton-raycast-heightfield" not in modules_by_name
    assert "newton-raycast-moving-geometry" not in modules_by_name


@pytest.mark.parametrize("catalog", [programs.DEMOS, programs.EXAMPLES])
def test_program_catalog_names_are_unique(catalog):
    """Each CLI program name must select exactly one module."""
    names = [program.name for program in catalog]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("command", "command_name", "program_name"),
    [(cli.demo, "demo", "zoo"), (cli.example, "example", "cables")],
)
def test_program_command_dispatches_to_packaged_module(command, command_name, program_name):
    """Program commands must forward all remaining arguments to the selected module."""
    with mock.patch.object(programs, "run_program") as run_program:
        command([program_name, "--physics", "newton_mjwarp"])

    catalog = programs.DEMOS if command_name == "demo" else programs.EXAMPLES
    selected = next(program for program in catalog if program.name == program_name)
    run_program.assert_called_once_with(command_name, selected, ["--physics", "newton_mjwarp"])


def test_program_command_forwards_help_to_selected_module():
    """Help after a program name belongs to that program, not the catalog parser."""
    with mock.patch.object(programs, "run_program") as run_program:
        cli.demo(["zoo", "--help"])

    run_program.assert_called_once_with("demo", programs.DEMOS[0], ["--help"])


def test_program_runner_restores_process_arguments():
    """Running a program must not leak its arguments to the caller."""
    original_argv = sys.argv
    program = programs.DEMOS[0]
    with mock.patch.object(programs.runpy, "run_module") as run_module:
        programs.run_program("demo", program, ["--physics", "newton_mjwarp"])

    run_module.assert_called_once_with(program.module, run_name="__main__")
    assert sys.argv is original_argv


@pytest.mark.parametrize("command", [cli.demo, cli.example])
def test_program_command_rejects_unknown_name(command):
    """Unknown program names must fail before attempting a module import."""
    with pytest.raises(SystemExit, match="2"):
        command(["does-not-exist"])


def test_program_command_reports_missing_optional_dependencies(capsys):
    """A program with missing extras must print its complete uvx installation command."""
    with mock.patch.object(programs, "find_spec", return_value=None), pytest.raises(SystemExit, match="2"):
        cli.example(["camera"])

    assert "uvx --from 'isaaclab[isaacsim]' isaaclab example camera" in capsys.readouterr().err


@pytest.mark.parametrize(("command_name", "args"), [("demo", ["zoo", "--headless"]), ("example", ["cables"])])
def test_cli_routes_program_without_loading_external_tasks(command_name, args):
    """Packaged programs do not need task plug-in discovery before dispatch."""
    with (
        mock.patch.object(cli, "_load_external_tasks") as load_external_tasks,
        mock.patch.object(cli, command_name) as command,
        mock.patch.object(sys, "argv", ["isaaclab", command_name, *args]),
    ):
        cli.cli()

    load_external_tasks.assert_not_called()
    command.assert_called_once_with(args)


def test_cli_loads_downstream_tasks_before_benchmark():
    """Benchmarking must discover tasks from installed projects."""
    task_entry_point = mock.Mock()
    with (
        mock.patch.object(cli.importlib.metadata, "entry_points", return_value=[task_entry_point]) as entry_points,
        mock.patch.object(cli, "benchmark") as benchmark,
        mock.patch.object(sys, "argv", ["isaaclab", "benchmark", "runtime", "--task", "Example"]),
    ):
        cli.cli()

    entry_points.assert_called_once_with(group="isaaclab.tasks")
    task_entry_point.load.assert_called_once_with()
    benchmark.assert_called_once_with(["runtime", "--task", "Example"])


def test_list_envs_cli_dispatches_without_preloading_tasks():
    """Environment listing owns task discovery so its script wrapper can share the same behavior."""
    with (
        mock.patch.object(cli, "_load_external_tasks") as load_external_tasks,
        mock.patch.object(cli, "list_envs") as list_environments,
        mock.patch.object(sys, "argv", ["isaaclab", "list_envs", "--show_presets"]),
    ):
        cli.cli()

    load_external_tasks.assert_not_called()
    list_environments.assert_called_once_with(["--show_presets"])
