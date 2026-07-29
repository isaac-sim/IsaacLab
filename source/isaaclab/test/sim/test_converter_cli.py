# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys

import pytest

import isaaclab.sim.utils as sim_utils
from isaaclab.sim.converters import _converter_cli as converter_cli


def _make_io_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    return parser


def _make_fake_app_launcher(recorded: dict, simulation_app: object) -> type:
    class _FakeAppLauncher:
        is_available = staticmethod(lambda: True)

        def __init__(self, launcher_args: dict):
            recorded["launcher_args"] = launcher_args
            self.app = simulation_app

    return _FakeAppLauncher


def test_parse_args_launches_headless_kit_when_isaac_sim_available(monkeypatch: pytest.MonkeyPatch):
    # Isaac Sim present and no preview requested: Kit is launched for the importer extensions only,
    # so the launcher carries no visualizer.
    recorded = {}
    simulation_app = object()
    monkeypatch.setattr(converter_cli, "AppLauncher", _make_fake_app_launcher(recorded, simulation_app))
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out"])

    args_cli, selected_app = converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")

    assert selected_app is simulation_app
    assert recorded["launcher_args"] == {}
    assert args_cli.viz == "none"


def test_parse_args_rejects_kitless_viz_with_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    # a kitless visualizer cannot share the process with Kit, so the combination is rejected before
    # converting rather than converting and silently skipping the requested preview.
    recorded = {}
    monkeypatch.setattr(converter_cli, "AppLauncher", _make_fake_app_launcher(recorded, object()))
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out", "--viz", "newton"])

    with pytest.raises(SystemExit):
        converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")


def test_parse_args_uses_standalone_provider_without_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(converter_cli.AppLauncher, "is_available", lambda: False)
    monkeypatch.setattr(converter_cli.ImporterProvider, "is_standalone_available", lambda: True)
    monkeypatch.setattr(converter_cli.ImporterProvider, "validate_standalone_runtime", lambda importer_kind: None)
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out"])

    args_cli, simulation_app = converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")

    assert args_cli.input == "robot.urdf"
    assert simulation_app is None


def test_parse_args_reports_missing_providers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(converter_cli.AppLauncher, "is_available", lambda: False)
    monkeypatch.setattr(converter_cli.ImporterProvider, "is_standalone_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out"])

    with pytest.raises(ImportError, match="either the full Isaac Sim package or the standalone"):
        converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")


def test_parse_args_rejects_kit_viz_without_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    # the Kit viewport needs a full Isaac Sim installation, so '--viz kit' fails before converting.
    monkeypatch.setattr(converter_cli.AppLauncher, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out", "--viz", "kit"])

    with pytest.raises(SystemExit):
        converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")


def test_parse_args_accepts_kit_viz_with_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    # with Isaac Sim installed the conversion runs through Kit, so its viewport can host the preview.
    recorded = {}
    simulation_app = object()
    monkeypatch.setattr(converter_cli, "AppLauncher", _make_fake_app_launcher(recorded, simulation_app))
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out", "--viz", "kit"])

    args_cli, selected_app = converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")

    assert args_cli.viz == "kit"
    assert selected_app is simulation_app
    # the Kit viewport only exists when the launcher is told to create it
    assert recorded["launcher_args"] == {"visualizer": ["kit"]}


def test_preview_opens_kit_viewport(monkeypatch: pytest.MonkeyPatch):
    # '--viz kit' shows the converted asset in the viewport of the Kit app that ran the conversion.
    calls: list = []
    monkeypatch.setattr(converter_cli.AppLauncher, "has_gui", staticmethod(lambda: True))
    monkeypatch.setattr(sim_utils, "show_stage_in_viewport", lambda usd_path: calls.append(usd_path))

    converter_cli.ConverterCli.preview(argparse.Namespace(viz="kit"), object(), "/tmp/robot.usd")

    assert calls == ["/tmp/robot.usd"]


def test_preview_skips_when_not_requested(monkeypatch: pytest.MonkeyPatch):
    calls: list = []
    monkeypatch.setattr(converter_cli.ConverterCli, "_preview_kitless", lambda *args: calls.append(args))

    converter_cli.ConverterCli.preview(argparse.Namespace(viz="none"), None, "/tmp/robot.usd")

    assert calls == []


def test_preview_opens_kitless_visualizer(monkeypatch: pytest.MonkeyPatch):
    # kitless conversion (no SimulationApp): the requested visualizer opens the converted asset.
    calls: list = []
    monkeypatch.setattr(
        converter_cli.ConverterCli,
        "_preview_kitless",
        lambda usd_path, visualizer: calls.append((usd_path, visualizer)),
    )

    converter_cli.ConverterCli.preview(argparse.Namespace(viz="newton"), None, "/tmp/robot.usd")

    assert calls == [("/tmp/robot.usd", "newton")]


def test_parse_args_auto_viz_selects_kit_with_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    # bare '--viz' picks the backend that fits the runtime, so Isaac Sim gets the Kit viewport.
    recorded = {}
    monkeypatch.setattr(converter_cli, "AppLauncher", _make_fake_app_launcher(recorded, object()))
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out", "--viz"])

    args_cli, _ = converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")

    assert args_cli.viz == "kit"
    assert recorded["launcher_args"] == {"visualizer": ["kit"]}


def test_parse_args_auto_viz_selects_newton_without_isaac_sim(monkeypatch: pytest.MonkeyPatch):
    # kitless, the Newton viewer is the one that ships with the base install.
    monkeypatch.setattr(converter_cli.AppLauncher, "is_available", lambda: False)
    monkeypatch.setattr(converter_cli.ImporterProvider, "is_standalone_available", lambda: True)
    monkeypatch.setattr(converter_cli.ImporterProvider, "validate_standalone_runtime", lambda importer_kind: None)
    monkeypatch.setattr(sys, "argv", ["convert_urdf.py", "robot.urdf", "out", "--viz"])

    args_cli, simulation_app = converter_cli.ConverterCli.parse_args(_make_io_parser(), "urdf")

    assert args_cli.viz == "newton"
    assert simulation_app is None


def test_preview_skips_kit_without_gui(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    # a Kit app resolved without a GUI has no viewport to show or close, so the preview is skipped.
    calls: list = []
    monkeypatch.setattr(converter_cli.AppLauncher, "has_gui", staticmethod(lambda: False))
    monkeypatch.setattr(sim_utils, "show_stage_in_viewport", lambda usd_path: calls.append(usd_path))

    with caplog.at_level("WARNING"):
        converter_cli.ConverterCli.preview(argparse.Namespace(viz="kit"), object(), "/tmp/robot.usd")

    assert calls == []
    assert "without a GUI" in caplog.text


def test_preview_failure_does_not_fail_conversion(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    # the asset is already written, so a preview that cannot run is reported instead of raising.
    def _boom(usd_path, visualizer):
        raise ValueError("The model must have at least one joint")

    monkeypatch.setattr(converter_cli.ConverterCli, "_preview_kitless", _boom)

    with caplog.at_level("ERROR"):
        converter_cli.ConverterCli.preview(argparse.Namespace(viz="newton"), None, "/tmp/robot.usd")

    assert "at least one joint" in caplog.text
