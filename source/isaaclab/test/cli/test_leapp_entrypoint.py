# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the installed LEAPP commands."""

import os
import subprocess
import sys
from unittest import mock

import pytest

import isaaclab.cli as cli

pytestmark = pytest.mark.unit


def _task_helper_stays_python(argv: list[str], runner: str) -> bool:
    """Import one task entry point through the CLI and report whether its helper stayed uncompiled."""
    import torch

    def helper(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    compiled: dict[str, object] = {}

    def load() -> None:
        compiled["fn"] = torch.jit.script(helper)

    entry_point = mock.Mock()
    entry_point.load.side_effect = load
    run = mock.Mock(return_value=0)
    enabled = bool(torch.jit._state._enabled)
    try:
        torch.jit._state.enable()  # known starting state, independent of earlier tests
        with (
            mock.patch.dict(sys.modules, {"isaaclab_rl.entrypoints": mock.Mock(**{runner: run})}),
            mock.patch.object(cli.importlib.metadata, "entry_points", return_value=[entry_point]),
            mock.patch.object(sys, "argv", ["isaaclab", *argv]),
        ):
            cli.cli()
    finally:
        if enabled:
            torch.jit._state.enable()
        else:
            torch.jit._state.disable()
    run.assert_called_once()
    return compiled["fn"] is helper


def test_leapp_export_does_not_compile_helpers_imported_by_external_tasks():
    """LEAPP export must disable TorchScript before downstream task packages import scripted helpers."""
    assert _task_helper_stays_python(
        ["leapp", "export", "--rl_library", "rsl_rl", "--task", "Example"],
        "run_export_cli",
    )


def test_train_still_compiles_helpers_imported_by_external_tasks():
    """Training must keep TorchScript enabled while external task packages import."""
    assert not _task_helper_stays_python(["train", "--task", "Example"], "run_train_cli")


def test_deploy_module_does_not_import_pxr_before_app_launch():
    """Importing deploy must not load pxr before SimulationApp starts."""
    env = os.environ.copy()
    env.update({"ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "Y"})
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import isaaclab.cli.commands.deploy; assert 'pxr' not in sys.modules"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
