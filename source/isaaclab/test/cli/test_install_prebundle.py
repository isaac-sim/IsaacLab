# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for prebundle probe and _split_install_items.

Supplements test_install_commands.py with tests that verify the probe
script text and the comma-separated install item parser.
"""

from unittest import mock

from isaaclab.cli.commands.install import (
    _split_install_items,
    _torch_first_on_sys_path_is_prebundle,
)

# ---------------------------------------------------------------------------
# _split_install_items
# ---------------------------------------------------------------------------


class TestSplitInstallItems:
    """Tests for :func:`_split_install_items`."""

    def test_single_item(self):
        assert _split_install_items("assets") == ["assets"]

    def test_comma_separated(self):
        assert _split_install_items("assets,tasks,rl") == ["assets", "tasks", "rl"]

    def test_with_spaces(self):
        assert _split_install_items(" assets , tasks , rl ") == ["assets", "tasks", "rl"]

    def test_brackets_preserved(self):
        """Commas inside brackets should not split."""
        assert _split_install_items("visualizers[rerun,newton],tasks") == [
            "visualizers[rerun,newton]",
            "tasks",
        ]

    def test_nested_brackets(self):
        assert _split_install_items("a[b[c,d],e],f") == ["a[b[c,d],e]", "f"]

    def test_empty_string(self):
        assert _split_install_items("") == []

    def test_trailing_comma(self):
        assert _split_install_items("assets,tasks,") == ["assets", "tasks"]

    def test_single_with_extra(self):
        assert _split_install_items("visualizers[all]") == ["visualizers[all]"]


# ---------------------------------------------------------------------------
# _torch_first_on_sys_path_is_prebundle — probe script verification
# ---------------------------------------------------------------------------


class TestTorchProbeScriptContent:
    """Verify that the probe script checks for 'pip_prebundle' not 'extsDeprecated'."""

    def test_probe_script_checks_pip_prebundle(self):
        """The inline Python probe must use 'pip_prebundle' as its path indicator."""
        import subprocess

        captured_cmd = None

        def fake_run(cmd, *, env=None, check=False, capture_output=False, text=False):
            nonlocal captured_cmd
            captured_cmd = cmd
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        with mock.patch("isaaclab.cli.commands.install.run_command", side_effect=fake_run):
            _torch_first_on_sys_path_is_prebundle("/fake/python", env={})

        assert captured_cmd is not None
        probe_script = captured_cmd[2]  # [python_exe, "-c", probe]
        assert "pip_prebundle" in probe_script, "Probe must check for 'pip_prebundle'"
        assert "extsDeprecated" not in probe_script, "Probe must NOT check only for 'extsDeprecated'"
