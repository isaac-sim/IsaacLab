# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for workflow commands exposed by an installed ``isaaclab`` package."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

import isaaclab
import isaaclab.__main__ as package_main
import isaaclab.cli as cli
import isaaclab.paths as paths

pytestmark = pytest.mark.unit


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


def test_legacy_vscode_option_uses_compatibility_dispatcher():
    """The installed entry point must continue to recognize the legacy VS Code option."""
    with (
        mock.patch.object(sys, "argv", ["isaaclab", "--generate-vscode-settings"]),
        mock.patch.object(package_main, "generate_vscode_settings") as generate,
    ):
        package_main.main()

    generate.assert_called_once_with()


def test_installed_vscode_generator_uses_pyright_config(tmp_path, monkeypatch):
    """The installed workflow must not emit the conflicting Pylance extraPaths setting."""
    monkeypatch.chdir(tmp_path)
    with (
        mock.patch.object(package_main, "resolve_isaacsim_dir", return_value=None),
        mock.patch.object(package_main, "build_extra_paths", return_value=["/sim/exts/example"]),
        mock.patch.object(package_main, "write_pyright_config") as write_pyright_config,
    ):
        package_main.generate_vscode_settings()

    settings = (tmp_path / ".vscode" / "settings.json").read_text()
    assert "python.analysis.extraPaths" not in settings
    write_pyright_config.assert_called_once_with(tmp_path, ["/sim/exts/example"])


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
