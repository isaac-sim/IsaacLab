# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the multi-GPU training launcher's command building.

Regression tests for forwarding ``--kit_args`` to the training script. Kit
arguments always start with ``--``, so the space-separated form
``--kit_args "--foo=/bar"`` used to be forwarded to the child training script
as two tokens, which argparse rejects with "expected one argument" (exit code
2) on every rank. The launcher now fuses the pair into a single
``--kit_args=--foo=/bar`` token.

These tests exercise the pure command-building logic and run without a GPU or
Isaac Sim installation.
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
from pathlib import Path

import pytest

# The launcher script lives outside the installed packages; load it by path.
# This test lives at source/isaaclab/test/cli/test_train_multigpu_command_building.py.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TRAIN_MULTIGPU_PATH = _REPO_ROOT / "scripts" / "reinforcement_learning" / "train_multigpu.py"

_spec = importlib.util.spec_from_file_location("train_multigpu", _TRAIN_MULTIGPU_PATH)
train_multigpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_multigpu)


def _build_command(argv: list[str]) -> list[str]:
    """Build the distributed launcher command the same way ``main`` does."""
    args_cli, train_args = train_multigpu._parse_args(argv)
    return train_multigpu._build_distributed_command(args_cli, train_args)


def _forwarded_train_argv(command: list[str]) -> list[str]:
    """Return the argv forwarded to the child training script."""
    return command[command.index(str(train_multigpu.TRAIN_SCRIPT)) + 1 :]


def _parse_as_training_script(child_argv: list[str]) -> str:
    """Parse forwarded argv with the training script's ``--kit_args`` argparse spec.

    Replicates how ``AppLauncher.add_app_launcher_args`` registers ``--kit_args``
    (``type=str``) in the child training script and returns the parsed value.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit_args", type=str, default="")
    try:
        args, _unknown = parser.parse_known_args(child_argv)
    except SystemExit:
        pytest.fail(f"training script argparse rejected the forwarded arguments: {child_argv}")
    return args.kit_args


class TestKitArgsForwarding:
    """Tests for forwarding ``--kit_args`` through the multi-GPU launcher."""

    def test_space_separated_kit_args_fused_in_torchrun_command(self):
        command = _build_command(["--task", "Isaac-Cartpole-Direct", "--kit_args", "--foo=/bar"])
        assert "--kit_args=--foo=/bar" in command
        assert "--kit_args" not in command

    def test_space_separated_kit_args_fused_in_skrl_jax_command(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        command = _build_command(
            [
                "--rl_library",
                "skrl",
                "--num_gpus",
                "2",
                "--task",
                "Isaac-Cartpole-Direct",
                "--ml_framework",
                "jax",
                "--kit_args",
                "--foo=/bar",
            ]
        )
        assert "skrl.utils.distributed.jax" in command
        assert "--kit_args=--foo=/bar" in command
        assert "--kit_args" not in command

    def test_equals_attached_kit_args_forwarded_unchanged(self):
        command = _build_command(["--task", "Isaac-Cartpole-Direct", "--kit_args=--foo=/bar"])
        assert "--kit_args=--foo=/bar" in command
        assert "--kit_args" not in command

    def test_multi_token_kit_args_value_forwarded_as_single_token(self):
        command = _build_command(["--task", "Isaac-Cartpole-Direct", "--kit_args", "--foo=/bar --baz=1"])
        assert "--kit_args=--foo=/bar --baz=1" in command

    def test_forwarded_kit_args_accepted_by_training_script_argparse(self):
        command = _build_command(["--task", "Isaac-Cartpole-Direct", "--kit_args", "--foo=/bar"])
        kit_args = _parse_as_training_script(_forwarded_train_argv(command))
        assert kit_args == "--foo=/bar"

    def test_dry_run_prints_parsable_command_with_fused_kit_args(self, capsys):
        exit_code = train_multigpu.main(["--dry_run", "--task", "Isaac-Cartpole-Direct", "--kit_args", "--foo=/bar"])
        assert exit_code == 0
        printed = capsys.readouterr().out.strip()
        tokens = shlex.split(printed)
        assert "--kit_args=--foo=/bar" in tokens
