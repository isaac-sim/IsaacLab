# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

import pytest

import isaaclab.app.app_launcher as app_launcher_module
from isaaclab.app import AppLauncher


@pytest.mark.usefixtures("mocker")
def test_livestream_launch_with_kwargs(mocker):
    """Test launching with keyword arguments."""
    # everything defaults to None
    app = AppLauncher(headless=True, livestream=1).app

    from isaaclab.app.settings_manager import get_settings_manager

    settings = get_settings_manager()
    assert settings.get("/app/window/enabled") is False
    assert settings.get("/app/livestream/enabled") is True

    # close the app on exit
    app.close()


class _DummySettings:
    def __init__(self):
        self.values = {}

    def set_string(self, path: str, value: str) -> None:
        self.values[path] = value

    def set_int(self, path: str, value: int) -> None:
        self.values[path] = value


def test_set_visualizer_settings_stores_values(monkeypatch: pytest.MonkeyPatch):
    settings = _DummySettings()
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", lambda: settings)

    launcher = AppLauncher.__new__(AppLauncher)
    launcher._set_visualizer_settings({"visualizer": ["viser", "rerun"], "visualizer_max_worlds": 0})

    assert settings.values == {
        "/isaaclab/visualizer/types": "viser rerun",
        "/isaaclab/visualizer/max_worlds": 0,
    }


def test_set_visualizer_settings_rejects_negative_max_worlds(monkeypatch: pytest.MonkeyPatch):
    def _unexpected_settings_manager():
        raise AssertionError("settings manager should not be queried for invalid values")

    monkeypatch.setattr(app_launcher_module, "get_settings_manager", _unexpected_settings_manager)

    launcher = AppLauncher.__new__(AppLauncher)
    with pytest.raises(ValueError, match="Invalid value for --visualizer_max_worlds: -5"):
        launcher._set_visualizer_settings({"visualizer": ["viser"], "visualizer_max_worlds": -5})


def test_set_visualizer_settings_suppresses_settings_manager_errors(monkeypatch: pytest.MonkeyPatch):
    def _raise_settings_error():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(app_launcher_module, "get_settings_manager", _raise_settings_error)

    launcher = AppLauncher.__new__(AppLauncher)
    launcher._set_visualizer_settings({"visualizer": ["viser"], "visualizer_max_worlds": 3})


def test_parse_visualizer_csv_accepts_comma_delimited_values():
    parsed = app_launcher_module._parse_visualizer_csv("kit,newton,rerun,viser")
    assert parsed == ["kit", "newton", "rerun", "viser"]


def test_parse_visualizer_csv_rejects_spaces_between_entries():
    with pytest.raises(argparse.ArgumentTypeError, match="spaces are not allowed"):
        app_launcher_module._parse_visualizer_csv("kit, newton")


def test_visualizer_csv_does_not_swallow_hydra_overrides():
    parser = argparse.ArgumentParser(add_help=False)
    app_launcher_module.AppLauncher.add_app_launcher_args(parser)

    args, hydra_args = parser.parse_known_args(
        ["--visualizer", "kit,newton,rerun", "presets=newton", "env.episode_length=10"]
    )

    assert args.visualizer == ["kit", "newton", "rerun"]
    assert hydra_args == ["presets=newton", "env.episode_length=10"]
