# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import types

import pytest

from isaaclab.sim.converters import _converter_cli as converter_cli


@pytest.fixture(autouse=True)
def _clear_kit_env(monkeypatch: pytest.MonkeyPatch):
    """Isolate tests from the ambient ``LIVESTREAM`` remote-preview request."""
    monkeypatch.delenv("LIVESTREAM", raising=False)


def _make_io_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    return parser


def _stub_carb(monkeypatch: pytest.MonkeyPatch, resolved: dict) -> None:
    carb_stub = types.SimpleNamespace(
        settings=types.SimpleNamespace(get_settings=lambda: types.SimpleNamespace(get=resolved.get))
    )
    monkeypatch.setitem(sys.modules, "carb", carb_stub)


def test_parse_args_launches_windowed_kit_for_viz_kit(monkeypatch: pytest.MonkeyPatch):
    recorded = {}
    simulation_app = object()

    class _FakeAppLauncher:
        def __init__(self, launcher_args: dict):
            recorded["launcher_args"] = launcher_args
            self.app = simulation_app

    monkeypatch.setattr(converter_cli.app_launcher_module, "SimulationApp", object())
    monkeypatch.setattr(converter_cli, "AppLauncher", _FakeAppLauncher)
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out", "--viz", "kit"])

    args_cli, selected_app = converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")

    assert selected_app is simulation_app
    assert recorded["launcher_args"] == {"visualizer": ["kit"]}
    assert args_cli.input == "robot.urdf"


def test_parse_args_launches_headless_kit_by_default(monkeypatch: pytest.MonkeyPatch):
    recorded = {}
    simulation_app = object()

    class _FakeAppLauncher:
        def __init__(self, launcher_args: dict):
            recorded["launcher_args"] = launcher_args
            self.app = simulation_app

    monkeypatch.setattr(converter_cli.app_launcher_module, "SimulationApp", object())
    monkeypatch.setattr(converter_cli, "AppLauncher", _FakeAppLauncher)
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out"])

    _, selected_app = converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")

    assert selected_app is simulation_app
    assert recorded["launcher_args"] == {}


def test_parse_args_uses_standalone_provider_without_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(converter_cli.app_launcher_module, "SimulationApp", None)
    monkeypatch.setattr(converter_cli.ImporterProvider, "is_standalone_available", lambda: True)
    monkeypatch.setattr(converter_cli.ImporterProvider, "validate_standalone_runtime", lambda importer_kind: None)
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out"])

    args_cli, simulation_app = converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")

    assert args_cli.input == "robot.urdf"
    assert args_cli.output == "out"
    assert simulation_app is None


def test_parse_args_rejects_viz_kit_without_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(converter_cli.app_launcher_module, "SimulationApp", None)
    monkeypatch.setattr(converter_cli.ImporterProvider, "is_standalone_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out", "--viz", "kit"])

    with pytest.raises(ImportError, match="Kit viewport preview requires the full Isaac Sim package"):
        converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")


def test_parse_args_reports_missing_providers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(converter_cli.app_launcher_module, "SimulationApp", None)
    monkeypatch.setattr(converter_cli.ImporterProvider, "is_standalone_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out"])

    with pytest.raises(ImportError, match="either the full Isaac Sim package or the standalone"):
        converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")


def test_parse_args_rejects_unknown_viz_choice(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out", "--viz", "web"])

    with pytest.raises(SystemExit):
        converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")


def test_parse_args_warns_on_ignored_livestream_kitless(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setenv("LIVESTREAM", "1")
    monkeypatch.setattr(converter_cli.app_launcher_module, "SimulationApp", None)
    monkeypatch.setattr(converter_cli.ImporterProvider, "is_standalone_available", lambda: True)
    monkeypatch.setattr(converter_cli.ImporterProvider, "validate_standalone_runtime", lambda importer_kind: None)
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out"])

    with caplog.at_level("WARNING"):
        _, simulation_app = converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")

    assert simulation_app is None
    assert "LIVESTREAM" in caplog.text


def test_maybe_preview_returns_without_kit(monkeypatch: pytest.MonkeyPatch):
    # kitless: must return before importing carb or omni (neither is stubbed here)
    args_cli = argparse.Namespace(viz="kit")

    converter_cli.ConverterCli.maybe_preview(args_cli, None, "/tmp/robot.usd")


def test_maybe_preview_returns_when_not_requested(monkeypatch: pytest.MonkeyPatch):
    # not requested: must return before importing carb (not stubbed here)
    args_cli = argparse.Namespace(viz="none")

    converter_cli.ConverterCli.maybe_preview(args_cli, object(), "/tmp/robot.usd")


def test_maybe_preview_defers_to_resolved_app_state(monkeypatch: pytest.MonkeyPatch):
    # e.g. HEADLESS=1 resolves the app windowless even though --viz kit was requested;
    # omni modules are not stubbed, so opening the stage would raise
    args_cli = argparse.Namespace(viz="kit")
    _stub_carb(monkeypatch, {"/app/window/enabled": False, "/app/livestream/enabled": False})

    converter_cli.ConverterCli.maybe_preview(args_cli, object(), "/tmp/robot.usd")


def _stub_omni_preview(monkeypatch: pytest.MonkeyPatch, opened: list) -> None:
    usd_stub = types.SimpleNamespace(
        get_context=lambda: types.SimpleNamespace(open_stage=lambda path: opened.append(path))
    )
    app_stub = types.SimpleNamespace(
        get_app_interface=lambda: types.SimpleNamespace(is_running=lambda: False, update=lambda: None)
    )
    kit_stub = types.SimpleNamespace(app=app_stub)
    monkeypatch.setitem(sys.modules, "omni", types.SimpleNamespace(usd=usd_stub, kit=kit_stub))
    monkeypatch.setitem(sys.modules, "omni.kit", kit_stub)
    monkeypatch.setitem(sys.modules, "omni.kit.app", app_stub)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_stub)


def test_maybe_preview_opens_stage_for_viz_kit(monkeypatch: pytest.MonkeyPatch):
    args_cli = argparse.Namespace(viz="kit")
    _stub_carb(monkeypatch, {"/app/window/enabled": True, "/app/livestream/enabled": False})
    opened: list = []
    _stub_omni_preview(monkeypatch, opened)

    converter_cli.ConverterCli.maybe_preview(args_cli, object(), "/tmp/robot.usd")

    assert opened == ["/tmp/robot.usd"]


def test_maybe_preview_counts_livestream_env_as_request(monkeypatch: pytest.MonkeyPatch):
    args_cli = argparse.Namespace(viz="none")
    monkeypatch.setenv("LIVESTREAM", "1")
    _stub_carb(monkeypatch, {"/app/window/enabled": False, "/app/livestream/enabled": True})
    opened: list = []
    _stub_omni_preview(monkeypatch, opened)

    converter_cli.ConverterCli.maybe_preview(args_cli, object(), "/tmp/robot.usd")

    assert opened == ["/tmp/robot.usd"]
