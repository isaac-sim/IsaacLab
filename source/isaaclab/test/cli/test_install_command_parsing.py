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

from isaaclab.cli.commands.install import (
    CORE_ISAACLAB_SUBMODULES,
    MANUAL_EXTRA_FEATURES,
    OPTIONAL_ISAACLAB_SUBMODULES,
    VALID_EXTRA_FEATURES,
    command_install,
    split_install_items,
)


def _optional_submodule_packages() -> list[str]:
    """Return flattened optional submodule source package names."""
    return [pkg for packages in OPTIONAL_ISAACLAB_SUBMODULES.values() for pkg in packages]


# ---------------------------------------------------------------------------
# split_install_items
# ---------------------------------------------------------------------------


class TestSplitInstallItems:
    """Tests for split_install_items()."""

    def test_single_token(self):
        assert split_install_items("newton") == ["newton"]

    def test_two_plain_tokens(self):
        assert split_install_items("newton,mimic") == ["newton", "mimic"]

    def test_token_with_selector(self):
        assert split_install_items("rl[rsl-rl]") == ["rl[rsl-rl]"]

    def test_comma_inside_brackets_not_split(self):
        assert split_install_items("rl[rsl-rl,skrl]") == ["rl[rsl-rl,skrl]"]

    def test_mixed_tokens(self):
        result = split_install_items("newton,rl[rsl-rl],mimic")
        assert result == ["newton", "rl[rsl-rl]", "mimic"]

    def test_whitespace_stripped(self):
        assert split_install_items("newton , mimic") == ["newton", "mimic"]

    def test_empty_string(self):
        assert split_install_items("") == []

    def test_all_special_value(self):
        assert split_install_items("all") == ["all"]

    def test_core_special_value(self):
        assert split_install_items("core") == ["core"]

    def test_none_special_value_alias(self):
        # back-compat: "none" is the old name for "core"
        assert split_install_items("none") == ["none"]

    def test_visualizer_with_selector(self):
        assert split_install_items("visualizer[rerun]") == ["visualizer[rerun]"]

    def test_multiple_selectors_mixed(self):
        result = split_install_items("mimic,visualizer[rerun],rl[rsl-rl]")
        assert result == ["mimic", "visualizer[rerun]", "rl[rsl-rl]"]

    def test_nested_brackets_depth(self):
        # Depth > 1 should not split on commas.
        result = split_install_items("contrib[a[b,c]]")
        assert result == ["contrib[a[b,c]]"]

    def test_missing_closing_bracket_not_split(self):
        # A malformed token with no closing ']' should come through as one item;
        # the install dispatcher is responsible for emitting the warning.
        result = split_install_items("rl[rsl-rl")
        assert result == ["rl[rsl-rl"]


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


class TestInstallConstants:
    """Sanity checks for module-level install constants."""

    def test_core_submodules_starts_with_isaaclab(self):
        assert CORE_ISAACLAB_SUBMODULES[0] == "isaaclab", (
            "isaaclab must be first so dependents resolve against the local copy"
        )

    def test_core_submodules_contains_expected_packages(self):
        expected = {
            "isaaclab",
            "isaaclab_ppisp",
            "isaaclab_assets",
            "isaaclab_contrib",
            "isaaclab_experimental",
            "isaaclab_newton",
            "isaaclab_ov",
            "isaaclab_ovphysx",
            "isaaclab_physx",
            "isaaclab_rl",
            "isaaclab_tasks",
            "isaaclab_tasks_experimental",
            "isaaclab_visualizers",
        }
        assert set(CORE_ISAACLAB_SUBMODULES) == expected

    def test_optional_submodules_contains_expected_packages(self):
        assert set(OPTIONAL_ISAACLAB_SUBMODULES.keys()) == {"mimic", "teleop"}
        assert OPTIONAL_ISAACLAB_SUBMODULES["mimic"] == ("isaaclab_teleop", "isaaclab_mimic")
        assert OPTIONAL_ISAACLAB_SUBMODULES["teleop"] == ("isaaclab_teleop",)

    def test_valid_extra_features(self):
        expected = {"contrib", "newton", "ov", "rl", "visualizer"}
        assert expected == VALID_EXTRA_FEATURES

    def test_manual_extra_features_subset_of_valid(self):
        assert MANUAL_EXTRA_FEATURES <= VALID_EXTRA_FEATURES

    def test_manual_extra_features(self):
        assert {"contrib", "ov"} == MANUAL_EXTRA_FEATURES

    def test_no_overlap_between_optional_submodules_and_extra_features(self):
        assert not (set(OPTIONAL_ISAACLAB_SUBMODULES.keys()) & VALID_EXTRA_FEATURES)

    def test_optional_submodules_not_in_core(self):
        core_names = set(CORE_ISAACLAB_SUBMODULES)
        for pkg in _optional_submodule_packages():
            assert pkg not in core_names


# ---------------------------------------------------------------------------
# command_install dispatch tests (all external I/O mocked)
# ---------------------------------------------------------------------------

_INSTALL_MODULE = "isaaclab.cli.commands.install"

# Functions that must be mocked to prevent actual system calls.
_PATCHES = [
    f"{_INSTALL_MODULE}._install_system_deps",
    f"{_INSTALL_MODULE}._install_isaaclab_submodules",
    f"{_INSTALL_MODULE}._install_extra_feature",
    f"{_INSTALL_MODULE}._install_optional_submodule_extra_dependencies",
    f"{_INSTALL_MODULE}._install_isaacsim",
    f"{_INSTALL_MODULE}._ensure_cuda_torch",
    f"{_INSTALL_MODULE}._maybe_preinstall_arm_nlopt",
    f"{_INSTALL_MODULE}._maybe_uninstall_prebundled_torch",
    f"{_INSTALL_MODULE}._ensure_pink_ik_dependencies_installed",
    f"{_INSTALL_MODULE}._repoint_prebundle_packages",
    f"{_INSTALL_MODULE}.command_vscode_settings",
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
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        # Core set must be present.
        for pkg in CORE_ISAACLAB_SUBMODULES:
            assert pkg in installed, f"Expected {pkg} in submodules for 'all'"
        # Optional submodules must be present.
        for pkg in _optional_submodule_packages():
            assert pkg in installed, f"Expected {pkg} (optional) in submodules for 'all'"

    def test_all_installs_auto_extra_features_not_manual(self):
        mocks = self._run("all")
        called_features = {c.args[0] for c in mocks["_install_extra_feature"].call_args_list}
        expected = VALID_EXTRA_FEATURES - MANUAL_EXTRA_FEATURES
        assert called_features == expected, f"'all' should install {expected}, got {called_features}"

    def test_all_does_not_install_optional_submodule_extras(self):
        mocks = self._run("all")
        mocks["_install_optional_submodule_extra_dependencies"].assert_not_called()

    def test_all_does_not_install_manual_extra_dependencies(self):
        mocks = self._run("all")
        called_features = {c.args[0] for c in mocks["_install_extra_feature"].call_args_list}
        assert "contrib" not in called_features
        assert "ov" not in called_features

    def test_all_does_not_call_install_isaacsim(self):
        mocks = self._run("all")
        mocks["_install_isaacsim"].assert_not_called()

    # --- "core" ---

    def test_core_installs_only_core_submodules(self):
        mocks = self._run("core")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert set(installed) == set(CORE_ISAACLAB_SUBMODULES)

    def test_core_installs_no_extra_features(self):
        mocks = self._run("core")
        mocks["_install_extra_feature"].assert_not_called()

    def test_core_does_not_install_optional_submodules(self):
        mocks = self._run("core")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        for pkg in _optional_submodule_packages():
            assert pkg not in installed

    def test_none_is_alias_for_core(self):
        # back-compat: "none" is the old name for "core"
        mocks_none = self._run("none")
        mocks_core = self._run("core")
        assert set(mocks_none["_install_isaaclab_submodules"].call_args[0][0]) == set(
            mocks_core["_install_isaaclab_submodules"].call_args[0][0]
        )
        mocks_none["_install_extra_feature"].assert_not_called()

    # --- extra features ---

    def test_newton_installs_core_plus_newton_extra(self):
        mocks = self._run("newton")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert set(installed) == set(CORE_ISAACLAB_SUBMODULES)
        mocks["_install_extra_feature"].assert_called_once_with("newton", "")

    def test_rl_with_selector(self):
        mocks = self._run("rl[rsl-rl]")
        mocks["_install_extra_feature"].assert_called_once_with("rl", "rsl-rl")

    def test_rl_without_selector(self):
        mocks = self._run("rl")
        mocks["_install_extra_feature"].assert_called_once_with("rl", "")

    def test_visualizer_with_selector(self):
        mocks = self._run("visualizer[rerun]")
        mocks["_install_extra_feature"].assert_called_once_with("visualizer", "rerun")

    # --- manual extra features and optional submodules ---

    def test_contrib_without_selector_dispatches_manual_extra_feature(self):
        mocks = self._run("contrib")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert set(installed) == set(CORE_ISAACLAB_SUBMODULES)
        mocks["_install_extra_feature"].assert_called_once_with("contrib", "")
        mocks["_install_optional_submodule_extra_dependencies"].assert_not_called()

    def test_contrib_with_selector_dispatches_manual_extra_feature(self):
        mocks = self._run("contrib[rlinf]")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert set(installed) == set(CORE_ISAACLAB_SUBMODULES)
        mocks["_install_extra_feature"].assert_called_once_with("contrib", "rlinf")
        mocks["_install_optional_submodule_extra_dependencies"].assert_not_called()

    def test_mimic_adds_mimic_to_submodules(self):
        mocks = self._run("mimic")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert "isaaclab_mimic" in installed
        mocks["_install_extra_feature"].assert_not_called()
        mocks["_install_optional_submodule_extra_dependencies"].assert_not_called()

    def test_ov_without_selector_dispatches_manual_extra_feature(self):
        mocks = self._run("ov")
        mocks["_install_extra_feature"].assert_called_once_with("ov", "")
        mocks["_install_optional_submodule_extra_dependencies"].assert_not_called()

    def test_ov_with_selector_dispatches_manual_extra_feature(self):
        mocks = self._run("ov[ovrtx]")
        mocks["_install_extra_feature"].assert_called_once_with("ov", "ovrtx")
        mocks["_install_optional_submodule_extra_dependencies"].assert_not_called()

    def test_teleop_adds_teleop_to_submodules(self):
        mocks = self._run("teleop")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert "isaaclab_teleop" in installed
        mocks["_install_extra_feature"].assert_not_called()
        mocks["_install_optional_submodule_extra_dependencies"].assert_not_called()

    # --- combined tokens ---

    def test_newton_and_rl_rsl_rl(self):
        mocks = self._run("newton,rl[rsl-rl]")
        calls = mocks["_install_extra_feature"].call_args_list
        features = {(c.args[0], c.args[1]) for c in calls}
        assert ("newton", "") in features
        assert ("rl", "rsl-rl") in features

    def test_mimic_and_newton(self):
        mocks = self._run("mimic,newton")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert "isaaclab_mimic" in installed
        mocks["_install_extra_feature"].assert_called_once_with("newton", "")

    def test_mimic_and_teleop(self):
        mocks = self._run("mimic,teleop")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert "isaaclab_mimic" in installed
        assert "isaaclab_teleop" in installed
        mocks["_install_extra_feature"].assert_not_called()

    # --- isaacsim token ---

    def test_isaacsim_token_triggers_isaacsim_install(self):
        mocks = self._run("isaacsim")
        mocks["_install_isaacsim"].assert_called_once()

    def test_isaacsim_still_installs_core_submodules(self):
        mocks = self._run("isaacsim")
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

    def test_newton_with_selector_still_dispatches(self):
        # The selector is forwarded to _install_extra_feature which emits the warning
        # internally (that function is mocked here; the warning itself is tested separately).
        mocks = self._run("newton[sim]")
        mocks["_install_extra_feature"].assert_called_once_with("newton", "sim")

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

    # --- isaaclab is always first ---

    def test_isaaclab_is_first_in_submodules_for_all(self):
        mocks = self._run("all")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert installed[0] == "isaaclab"

    def test_isaaclab_is_first_in_submodules_for_core(self):
        mocks = self._run("core")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert installed[0] == "isaaclab"

    def test_isaaclab_is_first_when_mimic_added(self):
        mocks = self._run("mimic")
        installed = mocks["_install_isaaclab_submodules"].call_args[0][0]
        assert installed[0] == "isaaclab"
