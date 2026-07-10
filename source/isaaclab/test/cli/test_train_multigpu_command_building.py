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
``dispatch_library_entrypoint`` applies the same fusing to the explicit argv list
it forwards to the per-library scripts (which parse that list, not ``sys.argv``).
The multi-GPU launcher forwards the tokens verbatim; each child rank runs through
the dispatcher and normalizes them at startup.

These tests exercise the pure normalization and command-building logic and run
without a GPU or Isaac Sim installation.
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import sys
from pathlib import Path

import pytest

from isaaclab.app.app_launcher import AppLauncher

# The launcher script lives outside the installed packages; load it by path.
# This test lives at source/isaaclab/test/cli/test_train_multigpu_command_building.py.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TRAIN_MULTIGPU_PATH = _REPO_ROOT / "scripts" / "reinforcement_learning" / "train_multigpu.py"

_spec = importlib.util.spec_from_file_location("train_multigpu", _TRAIN_MULTIGPU_PATH)
train_multigpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_multigpu)

_COMMON_PATH = _REPO_ROOT / "scripts" / "reinforcement_learning" / "common.py"
_common_spec = importlib.util.spec_from_file_location("rl_common", _COMMON_PATH)
rl_common = importlib.util.module_from_spec(_common_spec)
_common_spec.loader.exec_module(rl_common)


def _build_command(argv: list[str]) -> list[str]:
    """Build the distributed launcher command the same way ``main`` does."""
    args_cli, train_args = train_multigpu._parse_args(argv)
    return train_multigpu._build_distributed_command(args_cli, train_args)


def _forwarded_train_argv(command: list[str]) -> list[str]:
    """Return the argv forwarded to the child training script."""
    return command[command.index(str(train_multigpu.TRAIN_SCRIPT)) + 1 :]


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


class TestFuseKitArgs:
    """Unit tests for :meth:`AppLauncher._fuse_kit_args`."""

    def test_space_separated_option_like_value_is_fused(self):
        assert AppLauncher._fuse_kit_args(["--kit_args", "--foo=/bar"]) == ["--kit_args=--foo=/bar"]

    def test_equals_attached_value_is_unchanged(self):
        argv = ["--kit_args=--foo=/bar"]
        assert AppLauncher._fuse_kit_args(argv) == argv

    def test_value_with_space_is_unchanged(self):
        # a token containing a space cannot be mistaken for an option by argparse
        argv = ["--kit_args", "--foo=/bar --baz=1"]
        assert AppLauncher._fuse_kit_args(argv) == argv

    def test_non_option_value_is_unchanged(self):
        argv = ["--kit_args", "foo.txt"]
        assert AppLauncher._fuse_kit_args(argv) == argv

    def test_trailing_kit_args_is_unchanged(self):
        # argparse should still report the missing value normally
        argv = ["--task", "Isaac-Cartpole-Direct", "--kit_args"]
        assert AppLauncher._fuse_kit_args(argv) == argv

    def test_multiple_occurrences_are_all_fused(self):
        argv = ["--kit_args", "--foo=/a", "--task", "X", "--kit_args", "--bar=/b"]
        assert AppLauncher._fuse_kit_args(argv) == ["--kit_args=--foo=/a", "--task", "X", "--kit_args=--bar=/b"]

    def test_surrounding_tokens_are_preserved(self):
        argv = ["--task", "X", "--kit_args", "--foo=/bar", "--headless"]
        assert AppLauncher._fuse_kit_args(argv) == ["--task", "X", "--kit_args=--foo=/bar", "--headless"]


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


class TestDispatchLibraryEntrypoint:
    """Tests for the ``--kit_args`` normalization in ``dispatch_library_entrypoint``.

    The unified ``train.py``/``play.py`` dispatchers hand the per-library scripts an
    explicit argv list (not ``sys.argv``), so the normalization must also run on that
    list before it is forwarded.
    """

    @staticmethod
    def _dispatch(tmp_path: Path, argv: list[str]) -> list[str]:
        """Dispatch to a stub library entry point and return the argv it received."""
        stub = tmp_path / "stub_train.py"
        stub.write_text(
            "received = None\n\n\ndef run(argv):\n    global received\n    globals()['received'] = list(argv)\n"
        )
        exit_code = rl_common.dispatch_library_entrypoint(
            argv, {"stub": stub}, action="train", description="test", library_help="test"
        )
        assert exit_code == 0
        return sys.modules["isaaclab_rl_train_stub"].received

    def test_space_separated_kit_args_fused_before_library_script(self, tmp_path):
        received = self._dispatch(tmp_path, ["--rl_library", "stub", "--task", "X", "--kit_args", "--foo=/bar"])
        assert received == ["--task", "X", "--kit_args=--foo=/bar"]

    def test_already_working_argv_forms_forwarded_unchanged(self, tmp_path):
        argv = ["--task", "X", "--kit_args=--foo=/bar", "--kit_args", "--a=/x --b=/y", "--num_envs", "16", "--headless"]
        received = self._dispatch(tmp_path, ["--rl_library", "stub", *argv])
        assert received == argv


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
        exit_code = train_multigpu.main(["--dry_run", "--task", "Isaac-Cartpole-Direct", "--kit_args", "--foo=/bar"])
        assert exit_code == 0
        printed = capsys.readouterr().out.strip()
        tokens = shlex.split(printed)
        index = tokens.index("--kit_args")
        assert tokens[index + 1] == "--foo=/bar"
