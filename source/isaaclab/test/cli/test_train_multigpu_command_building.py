# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for passing ``--kit_args`` with an option-like value.

Kit arguments always start with ``--``, and argparse rejects a value token that
itself looks like an option (starts with ``-`` and contains no space) with
"expected one argument" (exit code 2). This used to break the documented
space-separated form ``--kit_args "--foo=/bar"`` for a single Kit argument on
every entry point, including all ranks of the multi-GPU launcher.

:meth:`~isaaclab.app.AppLauncher.add_app_launcher_args` now fuses such pairs in
``sys.argv`` into single ``--kit_args=<value>`` tokens before any parsing, and
the unified RL dispatcher applies the same fusing to the explicit argv list it
forwards to the backend modules (which parse that list, not ``sys.argv``).
The multi-GPU launcher forwards the tokens verbatim; each child rank runs through
the dispatcher and normalizes them at startup.

These tests exercise the pure normalization and command-building logic and run
without a GPU or Isaac Sim installation.
"""

from __future__ import annotations

import argparse
import shlex
import sys

import pytest

from isaaclab.app.app_launcher import AppLauncher

from isaaclab_rl.entrypoints import multigpu as train_multigpu


def _build_command(argv: list[str]) -> list[str]:
    """Build the distributed launcher command the same way ``main`` does."""
    args_cli, train_args = train_multigpu._parse_args(argv)
    return train_multigpu._build_distributed_command(args_cli, train_args)


def _forwarded_train_argv(command: list[str]) -> list[str]:
    """Return the argv forwarded to the child training script."""
    return command[command.index(train_multigpu.WORKER_SCRIPT) + 1 :]


def _parse_as_training_script(child_argv: list[str], monkeypatch: pytest.MonkeyPatch) -> str:
    """Parse forwarded argv the way the child training script does.

    Replicates the training-script startup: ``sys.argv`` holds the forwarded
    tokens, the parser is extended via :meth:`AppLauncher.add_app_launcher_args`
    (which installs the ``--kit_args`` normalization), and parsing must succeed.
    """
    monkeypatch.setattr(sys, "argv", ["train.py", *child_argv])
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    try:
        args, _unknown = parser.parse_known_args()
    except SystemExit:
        pytest.fail(f"training script argparse rejected the forwarded arguments: {child_argv}")
    return args.kit_args


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--kit_args", "--foo=/bar"], ["--kit_args=--foo=/bar"]),
        (["--kit_args=--foo=/bar"], None),
        # a token containing a space cannot be mistaken for an option by argparse
        (["--kit_args", "--foo=/bar --baz=1"], None),
        (["--kit_args", "foo.txt"], None),
        # argparse should still report the missing value normally
        (["--task", "Isaac-Cartpole-Direct", "--kit_args"], None),
        (
            ["--kit_args", "--foo=/a", "--task", "X", "--kit_args", "--bar=/b", "--num_envs", "16"],
            ["--kit_args=--foo=/a", "--task", "X", "--kit_args=--bar=/b", "--num_envs", "16"],
        ),
    ],
    ids=["fused", "equals-form", "value-with-space", "plain-value", "trailing", "multiple"],
)
def test_fuse_kit_args(argv, expected):
    """Only a space-separated option-like ``--kit_args`` value is fused; everything else passes through."""
    assert AppLauncher._fuse_kit_args(argv) == (argv if expected is None else expected)


class TestAddAppLauncherArgsNormalization:
    """Integration tests for the ``sys.argv`` normalization in ``add_app_launcher_args``."""

    def test_space_separated_single_kit_arg_parses(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--task", "X", "--kit_args", "--foo=/bar"])
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", type=str, default=None)
        AppLauncher.add_app_launcher_args(parser)
        args, unknown = parser.parse_known_args()
        assert args.kit_args == "--foo=/bar"
        assert unknown == []

    def test_unknown_leftovers_are_preserved(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--task", "X", "--kit_args", "--foo=/bar", "env.param=1"])
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", type=str, default=None)
        AppLauncher.add_app_launcher_args(parser)
        args, unknown = parser.parse_known_args()
        assert args.kit_args == "--foo=/bar"
        assert unknown == ["env.param=1"]


class TestKitArgsForwarding:
    """Tests for forwarding ``--kit_args`` through the multi-GPU launcher."""

    def test_space_separated_kit_args_forwarded_verbatim(self):
        command = _build_command(["--task", "Isaac-Cartpole-Direct", "--kit_args", "--foo=/bar"])
        index = command.index("--kit_args")
        assert command[index + 1] == "--foo=/bar"

    def test_equals_attached_kit_args_forwarded_unchanged(self):
        command = _build_command(["--task", "Isaac-Cartpole-Direct", "--kit_args=--foo=/bar"])
        assert "--kit_args=--foo=/bar" in command

    def test_multi_token_kit_args_value_forwarded_as_single_token(self):
        command = _build_command(["--task", "Isaac-Cartpole-Direct", "--kit_args", "--foo=/bar --baz=1"])
        index = command.index("--kit_args")
        assert command[index + 1] == "--foo=/bar --baz=1"

    def test_forwarded_space_separated_kit_args_accepted_by_training_script(self, monkeypatch):
        command = _build_command(["--task", "Isaac-Cartpole-Direct", "--kit_args", "--foo=/bar"])
        kit_args = _parse_as_training_script(_forwarded_train_argv(command), monkeypatch)
        assert kit_args == "--foo=/bar"

    def test_forwarded_skrl_jax_kit_args_accepted_by_training_script(self, monkeypatch):
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
        kit_args = _parse_as_training_script(_forwarded_train_argv(command), monkeypatch)
        assert kit_args == "--foo=/bar"

    def test_dry_run_prints_shell_parsable_command(self, capsys):
        exit_code = train_multigpu.run_train_multigpu_cli(
            ["--dry_run", "--task", "Isaac-Cartpole-Direct", "--kit_args", "--foo=/bar"]
        )
        assert exit_code == 0
        printed = capsys.readouterr().out.strip()
        tokens = shlex.split(printed)
        index = tokens.index("--kit_args")
        assert tokens[index + 1] == "--foo=/bar"
