# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for install.py token parsing and command_install dispatch logic.

These tests exercise the pure parsing logic and the install dispatch logic by
mocking all external I/O (pip, subprocess, filesystem), so they can run
without a GPU or Isaac Sim installation.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from isaaclab.cli.commands.install import (
    CORE_ISAACLAB_SUBMODULES,
    OPTIONAL_ISAACLAB_SUBMODULES,
    _install_extra_feature,
    _install_ov_extra_dependencies,
    command_install,
    split_install_items,
)

pytestmark = pytest.mark.unit


def _optional_submodule_packages() -> list[str]:
    """Return flattened optional submodule source package names."""
    return [pkg for packages in OPTIONAL_ISAACLAB_SUBMODULES.values() for pkg in packages]


# ---------------------------------------------------------------------------
# split_install_items
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("install_type", "expected"),
    [
        ("newton", ["newton"]),
        (" assets , tasks , rl ", ["assets", "tasks", "rl"]),
        # Commas inside brackets do not split; the outer comma does.
        ("visualizers[rerun,newton],tasks", ["visualizers[rerun,newton]", "tasks"]),
        # Depth > 1 does not split, and splitting resumes after the nested bracket closes.
        ("a[b[c,d],e],f", ["a[b[c,d],e]", "f"]),
        ("", []),
        ("assets,tasks,", ["assets", "tasks"]),
        # A malformed token with no closing ']' comes through as one item; the
        # install dispatcher is responsible for emitting the warning.
        ("rl[rsl-rl", ["rl[rsl-rl"]),
    ],
)
def test_split_install_items(install_type, expected):
    assert split_install_items(install_type) == expected


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


class TestInstallConstants:
    """Sanity checks for module-level install constants."""

    def test_submodule_tables_cover_every_source_package(self, source_checkout_root):
        source_packages = {
            path.parent.name for path in (source_checkout_root / "source").glob("isaaclab*/pyproject.toml")
        }
        optional_packages = set(_optional_submodule_packages())
        assert CORE_ISAACLAB_SUBMODULES[0] == "isaaclab", (
            "isaaclab must be first so dependents resolve against the local copy"
        )
        assert set(CORE_ISAACLAB_SUBMODULES) | optional_packages == source_packages
        assert set(OPTIONAL_ISAACLAB_SUBMODULES) == {"mimic", "teleop"}
        assert "isaaclab_teleop" in OPTIONAL_ISAACLAB_SUBMODULES["mimic"]

    def test_optional_submodules_not_in_core(self):
        core_names = set(CORE_ISAACLAB_SUBMODULES)
        for pkg in _optional_submodule_packages():
            assert pkg not in core_names


@pytest.mark.parametrize(
    ("selector", "expected_extra"),
    [
        ("ovphysx", "ovphysx"),
        ("ovrtx", "ovrtx"),
    ],
)
def test_ov_selector_installs_matching_root_extra(selector, expected_extra):
    """OV selectors dispatch to root extras with the same discoverable name."""
    with patch("isaaclab.cli.commands.install._install_root_extra") as install_root_extra:
        _install_ov_extra_dependencies(selector)

    install_root_extra.assert_called_once_with(expected_extra)


# ---------------------------------------------------------------------------
# command_install dispatch tests (all external I/O mocked)
# ---------------------------------------------------------------------------

_INSTALL_MODULE = "isaaclab.cli.commands.install"

# Functions that must be mocked to prevent actual system calls.
_PATCHES = [
    f"{_INSTALL_MODULE}._install_system_deps",
    f"{_INSTALL_MODULE}._arm_cmake_policy_compatibility",
    f"{_INSTALL_MODULE}._install_isaaclab_submodules",
    f"{_INSTALL_MODULE}._install_extra_feature",
    f"{_INSTALL_MODULE}._install_optional_submodule_extra_dependencies",
    # Centralized dependency installs read the root pyproject and shell out to pip.
    f"{_INSTALL_MODULE}._root_core_dependencies",
    f"{_INSTALL_MODULE}._install_root_extra",
    f"{_INSTALL_MODULE}._install_isaacsim",
    f"{_INSTALL_MODULE}._ensure_cuda_torch",
    f"{_INSTALL_MODULE}._maybe_uninstall_prebundled_torch",
    f"{_INSTALL_MODULE}._ensure_pink_ik_dependencies_installed",
    f"{_INSTALL_MODULE}._repoint_prebundle_packages",
    # Prebundle integrity checks probe for Isaac Sim (subprocesses) and scan the host's real OV cache.
    f"{_INSTALL_MODULE}._find_dangling_prebundle_symlinks",
    f"{_INSTALL_MODULE}._assert_no_new_dangling_prebundle_symlinks",
    f"{_INSTALL_MODULE}._ensure_newton",
    f"{_INSTALL_MODULE}._install_centralized_dependencies",
    f"{_INSTALL_MODULE}.command_editor",
    f"{_INSTALL_MODULE}.get_pip_command",
    f"{_INSTALL_MODULE}.extract_python_exe",
    # run_command is called directly inside command_install for pip/setuptools upgrades.
    f"{_INSTALL_MODULE}.run_command",
]


def _make_mock_env(**extra_env):
    """Return an os.environ copy suitable for mocking docker-detection."""
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
    env.update(extra_env)
    return env


@pytest.mark.usefixtures("source_checkout_root")
class TestCommandInstallDispatch:
    """Test that command_install() calls the right functions with the right args."""

    def _run(self, install_type: str):
        """Invoke command_install() with all I/O mocked; return captured mock calls."""
        mocks = {}
        patchers = []
        for target in _PATCHES:
            p = patch(target)
            m = p.start()
            mocks[target.split(".")[-1]] = m
            patchers.append(p)

        # Prevent docker-detection from reading /proc or .dockerenv.
        env_patcher = patch.dict(os.environ, {}, clear=False)
        exists_patcher = patch("os.path.exists", return_value=False)
        env_patcher.start()
        exists_patcher.start()
        patchers.extend([env_patcher, exists_patcher])

        try:
            command_install(install_type)
        finally:
            for p in patchers:
                p.stop()

        return mocks

    # --- "all" ---

    def test_all_installs_core_plus_optional_submodules(self):
        mocks = self._run("all")
        mocks["_arm_cmake_policy_compatibility"].assert_called_once_with()
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert installed[0] == "isaaclab"
        # Core set must be present.
        for pkg in CORE_ISAACLAB_SUBMODULES:
            assert pkg in installed, f"Expected {pkg} in submodules for 'all'"
        # Optional submodules must be present.
        for pkg in _optional_submodule_packages():
            assert pkg in installed, f"Expected {pkg} (optional) in submodules for 'all'"
        # Only the automatic extra features; manual ones (contrib, ov, tetrahedralization) are opt-in.
        called_features = {c.args[0] for c in mocks["_install_extra_feature"].call_args_list}
        assert called_features == {"newton", "rl", "visualizer"}
        mocks["_install_isaacsim"].assert_not_called()

    # --- "core" ---

    def test_core_installs_only_core_submodules(self):
        mocks = self._run("core")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert set(installed) == set(CORE_ISAACLAB_SUBMODULES)
        mocks["_install_extra_feature"].assert_not_called()

    def test_none_is_alias_for_core(self):
        # back-compat: "none" is the old name for "core"
        mocks_none = self._run("none")
        mocks_core = self._run("core")
        assert set(mocks_none["_install_isaaclab_submodules"].call_args[0][0]) == set(
            mocks_core["_install_isaaclab_submodules"].call_args[0][0]
        )
        mocks_none["_install_extra_feature"].assert_not_called()

    # --- extra features ---

    @pytest.mark.parametrize(
        ("install_type", "feature", "selector"),
        [
            ("newton", "newton", ""),
            ("rl[rsl-rl]", "rl", "rsl-rl"),
            ("visualizer[rerun]", "visualizer", "rerun"),
            ("tetrahedralization", "tetrahedralization", ""),
            ("contrib", "contrib", ""),
            ("ov", "ov", ""),
        ],
    )
    def test_extra_feature_installs_core_plus_feature(self, install_type, feature, selector):
        mocks = self._run(install_type)
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert set(installed) == set(CORE_ISAACLAB_SUBMODULES)
        mocks["_install_extra_feature"].assert_called_once_with(feature, selector)

    def test_tetrahedralization_installs_root_extra(self):
        with patch(f"{_INSTALL_MODULE}._install_root_extra") as install_root_extra:
            _install_extra_feature("tetrahedralization")

        install_root_extra.assert_called_once_with("tetrahedralization")

    # --- optional submodules ---

    def test_mimic_adds_mimic_to_submodules(self):
        mocks = self._run("mimic")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert "isaaclab_mimic" in installed
        mocks["_install_extra_feature"].assert_not_called()
        mocks["_install_optional_submodule_extra_dependencies"].assert_not_called()

    # --- combined tokens ---

    def test_newton_and_rl_rsl_rl(self):
        mocks = self._run("newton,rl[rsl-rl]")
        calls = mocks["_install_extra_feature"].call_args_list
        features = {(c.args[0], c.args[1]) for c in calls}
        assert ("newton", "") in features
        assert ("rl", "rsl-rl") in features

    def test_mimic_and_teleop(self):
        mocks = self._run("mimic,teleop")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert "isaaclab_mimic" in installed
        # Both submodules pull isaaclab_teleop; it must be installed once.
        assert installed.count("isaaclab_teleop") == 1
        mocks["_install_extra_feature"].assert_not_called()

    # --- isaacsim token ---

    def test_isaacsim_token_triggers_isaacsim_install(self):
        mocks = self._run("isaacsim")
        mocks["_install_isaacsim"].assert_called_once()
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert set(installed) == set(CORE_ISAACLAB_SUBMODULES)

    # --- malformed tokens ---

    def test_malformed_bracket_token_emits_warning_and_installs_core(self):
        with patch(f"{_INSTALL_MODULE}.print_warning") as mock_warn:
            mocks = self._run("rl[rsl-rl")  # missing closing bracket
        mock_warn.assert_called_once()
        warn_msg = mock_warn.call_args[0][0]
        assert "rl[rsl-rl" in warn_msg
        # Core submodules still installed.
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert set(installed) == set(CORE_ISAACLAB_SUBMODULES)
        # No extra feature should be installed.
        mocks["_install_extra_feature"].assert_not_called()

    # --- unknown token ---

    def test_unknown_token_emits_warning_and_installs_core(self):
        with patch(f"{_INSTALL_MODULE}.print_warning") as mock_warn:
            mocks = self._run("totally_unknown_package")
        mock_warn.assert_called_once()
        warn_msg = mock_warn.call_args[0][0]
        assert "totally_unknown_package" in warn_msg
        # Core submodules still installed.
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert set(installed) == set(CORE_ISAACLAB_SUBMODULES)
