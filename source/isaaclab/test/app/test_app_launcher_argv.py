# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the command-line arguments passed to Kit at startup."""

import sys

from isaaclab.app.app_launcher import AppLauncher, _sanitize_sys_argv_for_kit


def test_sanitize_sys_argv_removes_trailing_pytest_verbosity(monkeypatch):
    """Remove a pytest verbosity flag even when it is the final argument."""
    monkeypatch.setitem(sys.modules, "pytest", object())

    result = _sanitize_sys_argv_for_kit(["test_script.py", "--capture=no", "-vv"])

    assert result == ["test_script.py"]


def test_sanitize_sys_argv_preserves_user_verbosity_outside_pytest(monkeypatch):
    """Preserve application verbosity flags when pytest is not running."""
    monkeypatch.delitem(sys.modules, "pytest", raising=False)
    argv = ["script.py", "-v"]

    result = _sanitize_sys_argv_for_kit(argv)

    assert result is argv


def test_sanitize_sys_argv_removes_pytest_marker_pair(monkeypatch):
    """Remove a pytest marker option together with its expression."""
    monkeypatch.setitem(sys.modules, "pytest", object())

    result = _sanitize_sys_argv_for_kit(["test_script.py", "-m", "not isaacsim_ci", "--keep"])

    assert result == ["test_script.py", "--keep"]


def _resolve_device(launcher_args: dict) -> dict:
    """Run device resolution without constructing an ``AppLauncher``."""
    launcher = AppLauncher.__new__(AppLauncher)
    launcher.device_id = 0
    launcher._deferred_cuda_device_id = None
    launcher._xr = False
    AppLauncher._resolve_device_settings(launcher, launcher_args)
    return launcher_args


def test_both_devices_selected_by_cuda_index():
    """Select both devices by CUDA index, the renderer through the setting that translates it.

    The trailing comma is part of the contract: the setting is parsed as a comma-separated
    string and a bare integer is silently ignored.
    """
    args = _resolve_device({"device": "cuda:1"})

    assert "--/renderer/multiGpu/activeCudaGpus=1," in args["extra_args"]
    assert args["physics_gpu"] == 1


def test_active_gpu_is_left_unset():
    """Leave ``activeGpu`` unset: the renderer only applies the CUDA translation without it."""
    args = _resolve_device({"device": "cuda:1"})

    assert args.get("active_gpu") is None


def test_user_extra_args_are_preserved():
    """Append to caller-provided ``extra_args`` rather than replacing them."""
    args = _resolve_device({"device": "cuda:0", "extra_args": ["--/app/fastShutdown=False"]})

    assert "--/app/fastShutdown=False" in args["extra_args"]
    assert any("activeCudaGpus" in arg for arg in args["extra_args"])
