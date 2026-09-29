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
import isaaclab.cli as cli
import isaaclab.paths as paths
import isaaclab.program_browser as browser
import isaaclab.programs as programs
from isaaclab.sim import SimulationContext

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


_RL_ENTRYPOINTS = "isaaclab_rl.entrypoints"
_WORKFLOW_ARGS = ["--task", "Example"]
_LEAPP_DEPLOY_ARGS = ["--task", "Isaac-Cartpole", "--pipeline", "exported/Isaac-Cartpole.yaml", "physics=newton_mjwarp"]


@pytest.mark.parametrize(
    ("argv", "module", "runner", "runner_args", "status"),
    [
        (["train", *_WORKFLOW_ARGS], _RL_ENTRYPOINTS, "run_train_cli", _WORKFLOW_ARGS, 0),
        (["train", *_WORKFLOW_ARGS], _RL_ENTRYPOINTS, "run_train_cli", _WORKFLOW_ARGS, 3),
        (["play", *_WORKFLOW_ARGS], _RL_ENTRYPOINTS, "run_play_cli", _WORKFLOW_ARGS, 3),
        (["train_multigpu", *_WORKFLOW_ARGS], _RL_ENTRYPOINTS, "run_train_multigpu_cli", _WORKFLOW_ARGS, 3),
        (["zero_agent", *_WORKFLOW_ARGS], _RL_ENTRYPOINTS, "run_zero_agent_cli", _WORKFLOW_ARGS, 3),
        (["random_agent", *_WORKFLOW_ARGS], _RL_ENTRYPOINTS, "run_random_agent_cli", _WORKFLOW_ARGS, 3),
        (["benchmark", "training", "--help"], "isaaclab.benchmark", "run_benchmark_cli", ["training", "--help"], 3),
        (
            ["microbenchmark", "--component", "articulation", "physics=physx"],
            "isaaclab.benchmark",
            "run_microbenchmark_cli",
            ["--component", "articulation", "physics=physx"],
            3,
        ),
        (
            ["leapp", "export", "--rl_library", "rsl_rl", *_WORKFLOW_ARGS],
            _RL_ENTRYPOINTS,
            "run_export_cli",
            ["--rl_library", "rsl_rl", *_WORKFLOW_ARGS],
            3,
        ),
        (
            ["leapp", "deploy", *_LEAPP_DEPLOY_ARGS],
            "isaaclab.cli.commands.deploy",
            "command_deploy_leapp",
            _LEAPP_DEPLOY_ARGS,
            3,
        ),
    ],
    ids=[
        "train-success",
        "train",
        "play",
        "train_multigpu",
        "zero_agent",
        "random_agent",
        "benchmark",
        "microbenchmark",
        "leapp-export",
        "leapp-deploy",
    ],
)
def test_cli_subcommand_dispatches_and_propagates_status(argv, module, runner, runner_args, status):
    """Subcommands forward their arguments in-process and turn a nonzero result into the exit status."""
    run = mock.Mock(return_value=status)
    with (
        mock.patch.dict(sys.modules, {module: mock.Mock(**{runner: run})}),
        mock.patch.object(cli, "_load_external_tasks"),
        mock.patch.object(sys, "argv", ["isaaclab", *argv]),
    ):
        if status:
            with pytest.raises(SystemExit) as exc_info:
                cli.cli()
            assert exc_info.value.code == status
        else:
            cli.cli()

    run.assert_called_once_with(runner_args)


def test_demo_catalog_lists_packaged_demos(capsys):
    """The demo command must expose stable names without importing simulator modules."""
    cli.demo(["list"])

    output = capsys.readouterr().out
    assert "zoo" in output
    assert "teapot-fill" in output
    assert "newton-dominoes" not in output
    assert "bin-packing" not in output


def test_example_catalog_lists_packaged_examples(capsys):
    """The example command must distinguish focused programs from showcases."""
    cli.example(["list"])

    output = capsys.readouterr().out
    assert "bin-packing" in output
    assert "newton-dominoes" in output
    assert "mpm-two-way-coupling" in output
    assert "uvx --from 'isaaclab[isaacsim]' isaaclab example camera" in output
    assert "teapot-fill" not in output


def _visualizer(visualizer_type: str) -> mock.Mock:
    """Return a visualizer stand-in of the given type."""
    visualizer = mock.Mock()
    visualizer.cfg.visualizer_type = visualizer_type
    return visualizer


def test_newton_gl_selector_omits_incompatible_programs():
    """The selector must feature GL-compatible showcases and omit Kit-only programs."""
    gl_visualizer = _visualizer("newton_gl")
    kit_visualizer = _visualizer("kit")
    sim = mock.Mock(visualizers=[gl_visualizer, kit_visualizer])
    imgui = mock.Mock()
    imgui.collapsing_header.return_value = True
    imgui.tree_node.return_value = True
    imgui.selectable.return_value = (False, False)
    imgui.is_item_hovered.return_value = False
    program_browser = browser.ProgramBrowser({"demo": programs.DEMOS, "example": programs.EXAMPLES})
    program_browser.attach(sim)
    program_browser.attach(sim)
    gl_visualizer.register_ui_callback.call_args.args[0](imgui)

    labels = {call.args[0] for call in imgui.selectable.call_args_list}
    gl_visualizer.register_ui_callback.assert_called_once()
    assert gl_visualizer.register_ui_callback.call_args.kwargs == {"position": "panel"}
    kit_visualizer.register_ui_callback.assert_not_called()
    assert "Zoo##demo:zoo" in labels
    assert "Cables##example:cables" in labels
    assert "Newton Dominoes##example:newton-dominoes" in labels
    assert "Newton Dominoes##demo:newton-dominoes" not in labels
    assert "H1 Locomotion##demo:h1-locomotion" in labels
    assert "Pick And Place##demo:pick-and-place" not in labels
    assert "Ppisp Camera##example:ppisp-camera" not in labels
    assert "Haply Teleoperation##example:haply-teleoperation" not in labels
    assert imgui.set_next_item_open.call_count == 2


def test_programs_run_without_warp_backward_kernels(monkeypatch):
    """Programs only run inference, so their kernels must skip backward code generation."""
    import warp as wp

    monkeypatch.setattr(wp.config, "enable_backward", True)
    observed = []
    monkeypatch.setattr(programs, "_run_script", lambda _path: observed.append(wp.config.enable_backward))

    cli.demo(["zoo"])

    assert observed == [False]


def test_newton_gl_selector_switches_programs_after_script_returns(monkeypatch):
    """A GL selection must restart with GL after the current program finishes."""
    visualizer = _visualizer("newton_gl")
    imgui = mock.Mock()
    imgui.collapsing_header.return_value = True
    imgui.tree_node.return_value = True
    imgui.selectable.side_effect = lambda label, _selected: (label == "Cables##example:cables", False)
    imgui.is_item_hovered.return_value = False

    def run_script(_path):
        # Stand in for the program's simulation reset, which fires the launcher's callback.
        for callback in tuple(SimulationContext._reset_callbacks.values()):
            callback(mock.Mock(visualizers=[visualizer]))
        visualizer.register_ui_callback.call_args.args[0](imgui)

    monkeypatch.setattr(programs, "_run_script", run_script)
    with mock.patch.object(programs.os, "execv") as execv:
        cli.demo(["zoo", "--viz", "newton_gl"])

    visualizer.request_close.assert_called_once_with()
    assert not SimulationContext._reset_callbacks
    execv.assert_called_once_with(
        sys.executable, [sys.executable, "-m", "isaaclab", "example", "cables", "--viz", "newton_gl"]
    )


@pytest.mark.parametrize(
    ("catalog", "directory"), [(programs.DEMOS, "examples/demos"), (programs.EXAMPLES, "examples")]
)
def test_program_catalog_resolves_paths(catalog, directory):
    """Both catalogs must resolve inside the single examples tree."""
    assert all(program.relative_path.startswith(f"{directory}/") for program in catalog)
    assert all(program.path.is_file() for program in catalog)


@pytest.mark.parametrize("catalog", [programs.DEMOS, programs.EXAMPLES])
def test_program_catalog_names_are_unique(catalog):
    """Each CLI program name must select exactly one module."""
    names = [program.name for program in catalog]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("command", "command_name", "program_name"),
    [(cli.demo, "demo", "zoo"), (cli.example, "example", "cables")],
)
def test_program_command_dispatches_to_packaged_script(command, command_name, program_name):
    """Program commands must forward all remaining arguments to the selected script."""
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


def test_program_runner_preserves_command_name_and_restores_process_arguments(tmp_path):
    """Running a program must not leak its arguments to the caller."""
    original_argv = sys.argv
    script = tmp_path / "demos" / "example.py"
    script.parent.mkdir()
    script.write_text("import sys\nassert sys.argv[0] == 'isaaclab demo temporary'\n", encoding="utf-8")
    program = programs.ProgramSpec("temporary", f"examples/demos/{script.name}", "Temporary test program.")
    with mock.patch.object(programs, "_program_root", return_value=tmp_path):
        programs.run_program("demo", program, ["--physics", "newton_mjwarp"])

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
