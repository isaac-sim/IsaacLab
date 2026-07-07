# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import logging

import pytest

import isaaclab.app as app_module
import isaaclab.app.app_launcher as app_launcher_module
import isaaclab.app.sim_launcher as sim_launcher
import isaaclab.utils as utils_module
from isaaclab.app import AppLauncher
from isaaclab.app.sim_launcher import _ensure_livestream_kit_visualizer

pytestmark = pytest.mark.integration


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


def test_livestream_injects_kit_visualizer_when_missing():
    args = argparse.Namespace(livestream=2, visualizer=None, visualizer_explicit=False)

    _ensure_livestream_kit_visualizer(args)

    assert args.visualizer == ["kit"]


def test_livestream_rejects_disabled_visualizers():
    args = argparse.Namespace(livestream=2, visualizer=None, visualizer_explicit=True)

    with pytest.raises(ValueError, match="Livestreaming requires the Kit visualizer"):
        _ensure_livestream_kit_visualizer(args)


def test_launch_simulation_preserves_failure_exit_code(monkeypatch: pytest.MonkeyPatch):
    close_args = {}

    class _FakeApp:
        def close(self, *, exit_code: int = 0) -> None:
            close_args["exit_code"] = exit_code

    class _FakeAppLauncher:
        def __init__(self, _launcher_args):
            self.app = _FakeApp()

    scan = sim_launcher.Scan(
        resolved_physics_cfg=None,
        effective_cfg=object(),
        visualizer_intent={"has_any_visualizers": False, "has_kit_visualizer": False},
        has_ovrtx=False,
        has_kit_camera=False,
        has_kit_physics=True,
        has_kitless_physics=False,
        has_ovphysx_physics=False,
        needs_kit=True,
    )
    monkeypatch.setattr(sim_launcher, "scan", lambda cfg, physics: scan)
    monkeypatch.setattr(sim_launcher, "_ensure_isaac_sim_available", lambda: None)
    monkeypatch.setattr(app_module, "AppLauncher", _FakeAppLauncher)
    monkeypatch.setattr(utils_module, "has_kit", lambda: False)

    with pytest.raises(RuntimeError, match="sentinel"):
        with sim_launcher.launch_simulation(object(), argparse.Namespace()):
            raise RuntimeError("sentinel")

    assert close_args == {"exit_code": 1}


def test_deferred_cuda_device_synchronizes_torch_and_warp(monkeypatch: pytest.MonkeyPatch):
    """The post-Kit device hook must synchronize both CUDA runtimes."""
    devices = []
    monkeypatch.setattr(app_launcher_module, "set_cuda_device", devices.append)
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._deferred_cuda_device_id = 2

    launcher._set_deferred_cuda_device()

    assert devices == [2]


class _DummySettings:
    def __init__(self):
        self.values = {}

    def set_string(self, path: str, value: str) -> None:
        self.values[path] = value

    def set_int(self, path: str, value: int) -> None:
        self.values[path] = value

    def set_bool(self, path: str, value: bool) -> None:
        self.values[path] = value


@pytest.mark.parametrize(
    ("headless", "livestream", "xr", "expected_has_gui"),
    [
        pytest.param(False, 0, False, True, id="local-window"),
        pytest.param(True, 0, False, False, id="headless"),
        pytest.param(True, 1, False, True, id="livestream"),
        pytest.param(True, 0, True, True, id="xr"),
    ],
)
def test_load_extensions_publishes_has_gui_setting(
    monkeypatch: pytest.MonkeyPatch, headless: bool, livestream: int, xr: bool, expected_has_gui: bool
):
    """Publish the GUI state consumed by SimulationContext and RTX rendering."""
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._apply_rtx_determinism = False
    launcher._python_logging_level = logging.ERROR
    launcher._headless = headless
    launcher._livestream = livestream
    launcher._enable_cameras = False
    launcher._offscreen_render = False
    launcher._render_viewport = False
    launcher._xr = xr
    launcher._video_enabled = False

    settings = _DummySettings()
    monkeypatch.setattr(app_launcher_module, "initialize_carb_settings", lambda: None)
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", lambda: settings)
    monkeypatch.setattr(AppLauncher, "_apply_python_logging_level", lambda _level: None)

    launcher._load_extensions()

    assert settings.values["/isaaclab/has_gui"] is expected_has_gui


def test_set_visualizer_settings_stores_values(monkeypatch: pytest.MonkeyPatch):
    settings = _DummySettings()
    monkeypatch.setattr(app_launcher_module, "get_settings_manager", lambda: settings)

    launcher = AppLauncher.__new__(AppLauncher)
    launcher._set_visualizer_settings({"visualizer": ["viser", "rerun"], "max_visible_envs": 0})

    assert settings.values == {
        "/isaaclab/visualizer/types": "viser rerun",
        "/isaaclab/visualizer/explicit": False,
        "/isaaclab/visualizer/disable_all": False,
        "/isaaclab/visualizer/max_visible_envs": 0,
    }


def test_set_visualizer_settings_rejects_negative_max_visible_envs(
    monkeypatch: pytest.MonkeyPatch,
):
    def _unexpected_settings_manager():
        raise AssertionError("settings manager should not be queried for invalid values")

    monkeypatch.setattr(app_launcher_module, "get_settings_manager", _unexpected_settings_manager)

    launcher = AppLauncher.__new__(AppLauncher)
    with pytest.raises(ValueError, match="Invalid value for --max_visible_envs: -5"):
        launcher._set_visualizer_settings({"visualizer": ["viser"], "max_visible_envs": -5})


def test_set_visualizer_settings_suppresses_settings_manager_errors(monkeypatch: pytest.MonkeyPatch):
    def _raise_settings_error():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(app_launcher_module, "get_settings_manager", _raise_settings_error)

    launcher = AppLauncher.__new__(AppLauncher)
    launcher._set_visualizer_settings({"visualizer": ["viser"], "max_visible_envs": 3})


def test_parse_visualizer_csv_accepts_comma_delimited_values():
    parsed = app_launcher_module.AppLauncher._parse_visualizer_csv("kit,newton,rerun,viser")
    assert parsed == ["kit", "newton", "rerun", "viser"]


def test_parse_visualizer_csv_rejects_spaces_between_entries():
    with pytest.raises(argparse.ArgumentTypeError, match="spaces are not allowed"):
        app_launcher_module.AppLauncher._parse_visualizer_csv("kit, newton")


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
        ["--visualizer", "kit,newton,rerun", "presets=newton_mjwarp", "env.episode_length=10"]
    )

    assert args.visualizer == ["kit", "newton", "rerun"]
    assert hydra_args == ["presets=newton_mjwarp", "env.episode_length=10"]


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


@pytest.mark.parametrize("visualizer", [None, ["none"]])
def test_matrix_viz_none_disables_all_and_headless(monkeypatch: pytest.MonkeyPatch, visualizer):
    headless, launcher = _resolve_headless_for_case(
        monkeypatch,
        {
            "visualizer": visualizer,
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


def _new_launcher_for_experience_check():
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._enable_cameras = False
    launcher._headless = False
    launcher._xr = False
    launcher._apply_rtx_determinism = False
    launcher.is_isaac_sim_version_5 = lambda: False
    return launcher


def test_rejects_isaacsim_full_streaming_experience_with_livestream(tmp_path, monkeypatch: pytest.MonkeyPatch):
    experience = tmp_path / "isaacsim.exp.full.streaming.kit"
    experience.write_text('[dependencies]\n"isaacsim.exp.full" = {}\n', encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    launcher = _new_launcher_for_experience_check()
    launcher._livestream = 2

    with pytest.raises(ValueError, match="depends on 'isaacsim.exp.full'"):
        launcher._resolve_experience_file({"experience": str(experience)})


def test_rejects_custom_experience_with_isaacsim_full_dependency_and_livestream(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    experience = tmp_path / "merged.kit"
    experience.write_text('[dependencies]\n"isaaclab.python" = {}\n"isaacsim.exp.full" = {}\n', encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    launcher = _new_launcher_for_experience_check()
    launcher._livestream = 2

    with pytest.raises(ValueError, match="depends on 'isaacsim.exp.full'"):
        launcher._resolve_experience_file({"experience": str(experience)})


def test_allows_isaacsim_full_streaming_experience_when_livestream_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch):
    experience = tmp_path / "isaacsim.exp.full.streaming.kit"
    experience.write_text('[dependencies]\n"isaacsim.exp.full" = {}\n', encoding="utf-8")
    monkeypatch.setenv("EXP_PATH", str(tmp_path))
    launcher = _new_launcher_for_experience_check()
    launcher._livestream = 0

    launcher._resolve_experience_file({"experience": str(experience)})

    assert launcher._sim_experience_file == str(experience)
