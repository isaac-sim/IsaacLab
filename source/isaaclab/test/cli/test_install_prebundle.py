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
    _assert_no_new_dangling_prebundle_symlinks,
    _find_dangling_prebundle_symlinks,
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
# prebundle dangling-symlink integrity
# ---------------------------------------------------------------------------


class TestPrebundleSymlinkIntegrity:
    """Tests for :func:`_find_dangling_prebundle_symlinks` and
    :func:`_assert_no_new_dangling_prebundle_symlinks`.

    Regression guard for nvbugs 6343978: a pip downgrade deleted ``packaging``
    from the ``omni.isaac.core_archive`` prebundle, dangling the per-file
    symlink farm ``omni.services.pip_archive`` shares with it and cascading
    into 14 extension startup failures. Prebundle deletions by themselves are
    routine (site-packages shadows them); only new dangling symlinks fail.
    """

    def _make_prebundles(self, tmp_path):
        core = tmp_path / "exts" / "omni.isaac.core_archive" / "pip_prebundle"
        (core / "packaging").mkdir(parents=True)
        (core / "packaging" / "__init__.py").write_text("")
        services = tmp_path / "extscache" / "omni.services.pip_archive" / "pip_prebundle"
        (services / "packaging").mkdir(parents=True)
        (services / "packaging" / "__init__.py").symlink_to(core / "packaging" / "__init__.py")
        return core, services

    def test_intact_farm_has_no_dangling_links(self, tmp_path):
        core, services = self._make_prebundles(tmp_path)
        with mock.patch("isaaclab.cli.commands.install._discover_prebundle_dirs", return_value={core, services}):
            assert _find_dangling_prebundle_symlinks() == set()
            _assert_no_new_dangling_prebundle_symlinks(set())

    def test_raises_when_symlink_target_deleted(self, tmp_path):
        """Deleting the shared copy must fail the install, naming the broken link."""
        core, services = self._make_prebundles(tmp_path)
        with mock.patch("isaaclab.cli.commands.install._discover_prebundle_dirs", return_value={core, services}):
            before = _find_dangling_prebundle_symlinks()
            shutil.rmtree(core / "packaging")
            with pytest.raises(RuntimeError, match="dangling symlink") as excinfo:
                _assert_no_new_dangling_prebundle_symlinks(before)
        assert str(services / "packaging" / "__init__.py") in str(excinfo.value)

    def test_preexisting_dangling_links_are_tolerated(self, tmp_path):
        """Links already broken before the install do not fail it."""
        core, services = self._make_prebundles(tmp_path)
        (services / "stale.py").symlink_to(core / "does-not-exist.py")
        with mock.patch("isaaclab.cli.commands.install._discover_prebundle_dirs", return_value={core, services}):
            before = _find_dangling_prebundle_symlinks()
            assert before == {services / "stale.py"}
            _assert_no_new_dangling_prebundle_symlinks(before)

    def test_routine_dist_replacement_passes(self, tmp_path):
        """Deleting a prebundled package nothing links into is not a violation."""
        core, services = self._make_prebundles(tmp_path)
        (core / "six.py").write_text("")
        with mock.patch("isaaclab.cli.commands.install._discover_prebundle_dirs", return_value={core, services}):
            before = _find_dangling_prebundle_symlinks()
            (core / "six.py").unlink()
            _assert_no_new_dangling_prebundle_symlinks(before)

    def test_non_package_dangles_warn_but_pass(self, tmp_path):
        """Routine residue (dangling non-__init__ files) warns without failing.

        Every docker build leaves a few dozen dangling links to files Python
        never imports at startup (test modules, WHEEL/license files, cmake
        hooks); those must not abort the install.
        """
        core, services = self._make_prebundles(tmp_path)
        (services / "WHEEL").symlink_to(core / "gone-WHEEL")
        (services / "test_module.py").symlink_to(core / "gone-test.py")
        with mock.patch("isaaclab.cli.commands.install._discover_prebundle_dirs", return_value={core, services}):
            _assert_no_new_dangling_prebundle_symlinks(set())
