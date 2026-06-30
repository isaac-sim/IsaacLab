# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import pytest

import isaaclab.app.app_launcher as app_launcher_module
from isaaclab.app import AppLauncher


@pytest.mark.usefixtures("mocker")
def test_livestream_launch_with_env_vars(mocker):
    """Test launching with environment variables."""
    # Mock the environment variables
    mocker.patch.dict(os.environ, {"LIVESTREAM": "1", "HEADLESS": "1"})
    # everything defaults to None
    app_launcher = AppLauncher()
    app = app_launcher.app

    from isaaclab.app.settings_manager import get_settings_manager

    settings = get_settings_manager()
    assert settings.get("/app/window/enabled") is False
    # Do not assert /app/livestream/enabled here:
    # this key is owned by Kit extensions and is not guaranteed to be populated
    # in all app startup paths/environments. AppLauncher behavior is driven by
    # its resolved launch state and livestream args.
    assert app_launcher._livestream == 1
    assert app_launcher._headless is True
    assert "omni.kit.livestream.app" in app_launcher._livestream_args

    # close the app on exit
    app.close()


def _resolve_kit_args(
    monkeypatch: pytest.MonkeyPatch, launcher_args: dict, argv: list[str] | None = None
) -> tuple[list[str], list[str]]:
    monkeypatch.setattr(sys, "argv", ["pytest", *(argv or [])])
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._resolve_kit_args({"multi_gpu": False, **launcher_args})
    return sys.argv[1:], launcher._kit_args


def test_renderer_multi_gpu_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, raising=False)
    kit_args, resolved_args = _resolve_kit_args(monkeypatch, {})

    assert kit_args == [
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/maxGpuCount=1",
    ]
    assert resolved_args == kit_args


def test_renderer_multi_gpu_can_be_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, raising=False)
    kit_args, resolved_args = _resolve_kit_args(monkeypatch, {"multi_gpu": True})

    assert kit_args == []
    assert resolved_args == []


@pytest.mark.parametrize("env_value, expected", [("0", "false"), ("1", "true")])
def test_fabric_gpu_interop_env_adds_independent_override(
    monkeypatch: pytest.MonkeyPatch, env_value: str, expected: str
):
    monkeypatch.setenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, env_value)

    kit_args, _ = _resolve_kit_args(monkeypatch, {"multi_gpu": True})

    assert kit_args == [f"--/physics/fabricUseGPUInterop={expected}"]


def test_fabric_gpu_interop_env_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, "false")

    with pytest.raises(ValueError, match="Expected: 0 or 1"):
        _resolve_kit_args(monkeypatch, {"multi_gpu": True})


def test_explicit_kit_setting_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, "0")
    explicit_arg = "--/physics/fabricUseGPUInterop=true"

    kit_args, resolved_args = _resolve_kit_args(monkeypatch, {"multi_gpu": True}, [explicit_arg])

    assert kit_args == [explicit_arg]
    assert resolved_args == []


def test_explicit_renderer_kit_arg_is_not_duplicated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(app_launcher_module._FABRIC_GPU_INTEROP_ENV, raising=False)
    explicit_arg = "--/renderer/multiGpu/enabled=true"

    kit_args, resolved_args = _resolve_kit_args(monkeypatch, {"kit_args": explicit_arg})

    assert kit_args.count(explicit_arg) == 1
    assert resolved_args[0] == explicit_arg
    assert "--/renderer/multiGpu/enabled=false" not in resolved_args
