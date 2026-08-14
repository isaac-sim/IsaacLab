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


def _resolve_devices_and_kit_args(launcher_args: dict, monkeypatch) -> tuple[dict, list[str]]:
    """Resolve device settings and Kit arguments without constructing an ``AppLauncher``.

    ``_resolve_kit_args`` extends ``sys.argv``, so the caller's argv is isolated.
    """
    monkeypatch.setattr(sys, "argv", ["script.py"])
    launcher = AppLauncher.__new__(AppLauncher)
    launcher.device_id = 0
    launcher._deferred_cuda_device_id = None
    launcher._xr = False
    AppLauncher._resolve_device_settings(launcher, launcher_args)
    AppLauncher._resolve_kit_args(launcher, launcher_args)
    return launcher_args, launcher._kit_args


def test_both_devices_selected_by_cuda_index(monkeypatch):
    """Select both devices by CUDA index, the renderer through the setting that translates it.

    The trailing comma is part of the contract: without it the value is stored as an int and the
    renderer, which reads the setting as a string, sees nothing.
    """
    args, kit_args = _resolve_devices_and_kit_args({"device": "cuda:1", "multi_gpu": False}, monkeypatch)

    assert "--/renderer/multiGpu/activeCudaGpus=1," in kit_args
    assert args["physics_gpu"] == 1


def test_active_gpu_is_left_unset(monkeypatch):
    """Leave ``activeGpu`` unset: the renderer only applies the CUDA translation without it."""
    args, _ = _resolve_devices_and_kit_args({"device": "cuda:1", "multi_gpu": False}, monkeypatch)

    assert args.get("active_gpu") is None


def test_user_supplied_device_setting_is_not_overridden(monkeypatch):
    """Leave a caller-specified renderer device alone rather than adding a second setting."""
    args, kit_args = _resolve_devices_and_kit_args(
        {"device": "cuda:1", "multi_gpu": False, "kit_args": "--/renderer/multiGpu/activeCudaGpus=3,"}, monkeypatch
    )

    assert [arg for arg in kit_args if "activeCudaGpus" in arg] == ["--/renderer/multiGpu/activeCudaGpus=3,"]


def test_renderer_device_is_not_pinned_for_multi_gpu_rendering(monkeypatch):
    """Leave the device unset when Kit renders across several GPUs in one process.

    The setting fills the renderer's active-device list, and a one-element list would cap the
    device count at one.
    """
    _, kit_args = _resolve_devices_and_kit_args({"device": "cuda:1"}, monkeypatch)

    assert not any("activeCudaGpus" in arg for arg in kit_args)
