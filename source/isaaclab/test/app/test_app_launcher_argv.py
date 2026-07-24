# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for filtering command-line arguments before Kit startup."""

import sys

import pytest

from isaaclab.app.app_launcher import _sanitize_sys_argv_for_kit


def test_sanitize_sys_argv_removes_trailing_pytest_verbosity(monkeypatch):
    """Remove a pytest verbosity flag even when it is the final argument."""
    monkeypatch.setitem(sys.modules, "pytest", object())

    result = _sanitize_sys_argv_for_kit(["test_script.py", "--capture=no", "-vv"])

    assert result == ["test_script.py"]


def test_sanitize_sys_argv_removes_ci_profile_option(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytest", object())

    result = _sanitize_sys_argv_for_kit(["test_script.py", "--run-ci-tests", "--keep"])

    assert result == ["test_script.py", "--keep"]


def test_sanitize_sys_argv_preserves_user_verbosity_outside_pytest(monkeypatch):
    """Preserve application verbosity flags when pytest is not running."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    argv = ["script.py", "-v"]

    result = _sanitize_sys_argv_for_kit(argv)

    assert result is argv


@pytest.mark.parametrize("marker_expression", ["not isaacsim_ci", "requires_kit"])
def test_sanitize_sys_argv_removes_pytest_marker_pair(monkeypatch, marker_expression):
    """Remove a pytest marker option together with its expression."""
    monkeypatch.setitem(sys.modules, "pytest", object())

    result = _sanitize_sys_argv_for_kit(["test_script.py", "-m", marker_expression, "--keep"])

    assert result == ["test_script.py", "--keep"]
