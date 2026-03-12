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
    app_launcher = AppLauncher(headless=True, livestream=1)
    app = app_launcher.app
    assert app_launcher._livestream == 1
    assert app_launcher._headless is True

    # close the app on exit
    app.close()


class _DummySettings:
    def __init__(self):
        self.values = {}

    def set_string(self, path: str, value: str) -> None:
        self.values[path] = value

    def set_int(self, path: str, value: int) -> None:
        self.values[path] = value

    def set_bool(self, path: str, value: bool) -> None:
        self.values[path] = value


def test_set_visualizer_settings_stores_values(monkeypatch: pytest.MonkeyPatch):
    settings = _DummySettings()
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", lambda: settings)

    launcher = AppLauncher.__new__(AppLauncher)
    launcher._set_visualizer_settings({"visualizer": ["viser", "rerun"], "visualizer_max_worlds": 0})

    assert settings.values == {
        "/isaaclab/visualizer/types": "viser rerun",
        "/isaaclab/visualizer/explicit": False,
        "/isaaclab/visualizer/disable_all": False,
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


def test_resolve_visualizer_settings_rejects_none_with_others():
    launcher = AppLauncher.__new__(AppLauncher)
    with pytest.raises(ValueError, match="'none' cannot be combined"):
        launcher._resolve_visualizer_settings(
            {"visualizer": ["none", "kit"], "visualizer_explicit": True},
        )


def test_visualizer_csv_does_not_swallow_hydra_overrides():
    parser = argparse.ArgumentParser(add_help=False)
    app_launcher_module.AppLauncher.add_app_launcher_args(parser)

    args, hydra_args = parser.parse_known_args(
        ["--visualizer", "kit,newton,rerun", "presets=newton", "env.episode_length=10"]
    )

    assert args.visualizer == ["kit", "newton", "rerun"]
    assert hydra_args == ["presets=newton", "env.episode_length=10"]


def _resolve_headless_for_case(monkeypatch: pytest.MonkeyPatch, launcher_args: dict) -> tuple[bool, AppLauncher]:
    monkeypatch.setenv("HEADLESS", "0")
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._livestream = 0
    launcher._resolve_visualizer_settings(launcher_args)
    launcher._resolve_headless_settings(launcher_args, livestream_arg=-1, livestream_env=0)
    return launcher._headless, launcher


def test_matrix_cli_kit_newton_with_custom_kit_cfg_intent_non_headless(monkeypatch: pytest.MonkeyPatch):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "visualizer": ["kit", "newton"],
            "visualizer_explicit": True,
            "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        },
    )
    assert headless is False
    assert launcher._cli_visualizer_types == ["kit", "newton"]


def test_matrix_cli_rerun_with_custom_kit_cfg_intent_headless(monkeypatch: pytest.MonkeyPatch):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "visualizer": ["rerun"],
            "visualizer_explicit": True,
            "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        },
    )
    assert headless is True
    assert launcher._cli_visualizer_types == ["rerun"]


def test_matrix_no_cli_with_cfg_kit_newton_non_headless(monkeypatch: pytest.MonkeyPatch):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        },
    )
    assert headless is False
    assert launcher._cli_visualizer_explicit is False


def test_matrix_viz_none_disables_all_and_headless(monkeypatch: pytest.MonkeyPatch):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "visualizer": ["none"],
            "visualizer_explicit": True,
            "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        },
    )
    assert headless is True
    assert launcher._cli_visualizer_disable_all is True
    assert launcher._cli_visualizer_types == []


def test_matrix_headless_flag_deprecated_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "headless": True,
            "headless_explicit": True,
            "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        },
    )
    assert headless is True
    assert launcher._cli_visualizer_types == []


def test_matrix_headless_with_viz_names_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "headless": True,
            "headless_explicit": True,
            "visualizer": ["kit", "newton"],
            "visualizer_explicit": True,
            "visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": True},
        },
    )
    assert headless is True
    assert launcher._cli_visualizer_disable_all is True
    assert launcher._cli_visualizer_types == []


def test_no_cli_and_no_cfg_visualizers_defaults_headless(monkeypatch: pytest.MonkeyPatch):
    headless, _ = _resolve_headless_for_case(monkeypatch, {})
    assert headless is True


def test_no_cli_and_non_kit_cfg_visualizers_defaults_headless(monkeypatch: pytest.MonkeyPatch):
    headless, _ = _resolve_headless_for_case(
        monkeypatch,
        {"visualizer_intent": {"has_any_visualizers": True, "has_kit_visualizer": False}},
    )
    assert headless is True


def test_invalid_visualizer_intent_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEADLESS", "0")
    launcher = AppLauncher.__new__(AppLauncher)
    with pytest.raises(ValueError, match="visualizer_intent"):
        launcher._resolve_visualizer_settings({"visualizer_intent": {"has_any_visualizers": "yes"}})
