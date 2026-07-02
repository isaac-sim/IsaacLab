# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for prebundle probe, _split_install_items, and prebundle dist integrity.

Supplements test_install_commands.py with tests that verify the probe
script text, the comma-separated install item parser, and the
snapshot/assert pair guarding Isaac Sim prebundles against pip removals.
"""

import shutil
from unittest import mock

import pytest

from isaaclab.cli.commands.install import (
    _assert_prebundle_dists_intact,
    _snapshot_prebundle_dists,
    _torch_first_on_sys_path_is_prebundle,
    split_install_items,
)

# ---------------------------------------------------------------------------
# split_install_items
# ---------------------------------------------------------------------------


class TestSplitInstallItems:
    """Tests for :func:`split_install_items`."""

    def test_single_item(self):
        assert split_install_items("assets") == ["assets"]

    def test_comma_separated(self):
        assert split_install_items("assets,tasks,rl") == ["assets", "tasks", "rl"]

    def test_with_spaces(self):
        assert split_install_items(" assets , tasks , rl ") == ["assets", "tasks", "rl"]

    def test_brackets_preserved(self):
        """Commas inside brackets should not split."""
        assert split_install_items("visualizers[rerun,newton],tasks") == [
            "visualizers[rerun,newton]",
            "tasks",
        ]

    def test_nested_brackets(self):
        assert split_install_items("a[b[c,d],e],f") == ["a[b[c,d],e]", "f"]

    def test_empty_string(self):
        assert split_install_items("") == []

    def test_trailing_comma(self):
        assert split_install_items("assets,tasks,") == ["assets", "tasks"]

    def test_single_with_extra(self):
        assert split_install_items("visualizers[all]") == ["visualizers[all]"]


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


# ---------------------------------------------------------------------------
# prebundle dist integrity snapshot/assert
# ---------------------------------------------------------------------------


class TestPrebundleDistsIntegrity:
    """Tests for :func:`_snapshot_prebundle_dists` and :func:`_assert_prebundle_dists_intact`.

    Regression guard for nvbugs 6343978: a pip operation running with
    pip_prebundle paths visible uninstalled ``packaging`` from Isaac Sim's
    ``omni.isaac.core_archive`` prebundle, dangling the symlink farms other
    extensions share and cascading into 14 extension startup failures.
    """

    def _make_prebundle(self, tmp_path):
        prebundle = tmp_path / "exts" / "omni.isaac.core_archive" / "pip_prebundle"
        (prebundle / "packaging-26.0.dist-info").mkdir(parents=True)
        (prebundle / "packaging").mkdir()
        (prebundle / "six-1.16.0.dist-info").mkdir()
        return prebundle

    def test_snapshot_records_dist_info_names(self, tmp_path):
        prebundle = self._make_prebundle(tmp_path)
        with mock.patch("isaaclab.cli.commands.install._discover_prebundle_dirs", return_value={prebundle}):
            snapshot = _snapshot_prebundle_dists()
        assert snapshot == {prebundle: {"packaging-26.0.dist-info", "six-1.16.0.dist-info"}}

    def test_intact_when_unchanged(self, tmp_path):
        prebundle = self._make_prebundle(tmp_path)
        with mock.patch("isaaclab.cli.commands.install._discover_prebundle_dirs", return_value={prebundle}):
            snapshot = _snapshot_prebundle_dists()
        _assert_prebundle_dists_intact(snapshot)

    def test_intact_when_dist_added(self, tmp_path):
        """New distributions appearing in a prebundle are not a violation."""
        prebundle = self._make_prebundle(tmp_path)
        with mock.patch("isaaclab.cli.commands.install._discover_prebundle_dirs", return_value={prebundle}):
            snapshot = _snapshot_prebundle_dists()
        (prebundle / "tomli-2.0.1.dist-info").mkdir()
        _assert_prebundle_dists_intact(snapshot)

    def test_raises_when_dist_removed(self, tmp_path):
        """Removing a prebundled distribution must fail the install loudly."""
        prebundle = self._make_prebundle(tmp_path)
        with mock.patch("isaaclab.cli.commands.install._discover_prebundle_dirs", return_value={prebundle}):
            snapshot = _snapshot_prebundle_dists()
        shutil.rmtree(prebundle / "packaging-26.0.dist-info")
        shutil.rmtree(prebundle / "packaging")
        with pytest.raises(RuntimeError, match="packaging-26.0.dist-info") as excinfo:
            _assert_prebundle_dists_intact(snapshot)
        assert str(prebundle) in str(excinfo.value)

    def test_raises_when_prebundle_dir_removed(self, tmp_path):
        """A wholesale-deleted prebundle reports every distribution it held."""
        prebundle = self._make_prebundle(tmp_path)
        with mock.patch("isaaclab.cli.commands.install._discover_prebundle_dirs", return_value={prebundle}):
            snapshot = _snapshot_prebundle_dists()
        shutil.rmtree(prebundle)
        with pytest.raises(RuntimeError, match="six-1.16.0.dist-info"):
            _assert_prebundle_dists_intact(snapshot)
