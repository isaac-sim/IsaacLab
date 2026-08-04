# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

import pytest

from isaaclab.app import AppLauncher

pytestmark = pytest.mark.integration


@pytest.mark.usefixtures("mocker")
def test_livestream_launch_with_argparser(mocker):
    """Test launching with argparser arguments."""
    # Mock the parse_args method
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(livestream=1))
    # create argparser
    parser = argparse.ArgumentParser()
    # add app launcher arguments
    AppLauncher.add_app_launcher_args(parser)
    # check that argparser has the mandatory arguments
    for name in AppLauncher._APPLAUNCHER_CFG_INFO.keys() - {"headless", "enable_cameras"}:
        assert parser._option_string_actions[f"--{name}"]
    # parse args
    mock_args = parser.parse_args()
    # everything defaults to None
    app_launcher = AppLauncher(mock_args)
    app = app_launcher.app
    assert app_launcher._livestream == 1
    assert app_launcher._headless is True

    # close the app on exit
    app.close()


def test_visualizer_alias_parsing():
    """Test that --viz alias maps to visualizer values."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args(["--viz", "kit,newton"])
    assert args.visualizer == ["kit", "newton"]
    assert args.visualizer_explicit is True


@pytest.mark.parametrize("deprecated_arg", ["--headless", "--enable_cameras"])
def test_deprecated_render_flags_are_rejected(deprecated_arg: str):
    """Test that removed render flags are rejected by the parser."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([deprecated_arg])


def test_help_on_parser_with_required_positionals(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Launcher arguments reach the help output of a script that takes required positionals.

    ``add_app_launcher_args`` probes the command line to check for name collisions. That probe
    exits when a required argument is missing, which is the case for every tool script invoked
    with ``--help``, so it must not take the parser down before the arguments are added.
    """
    monkeypatch.setattr("sys.argv", ["convert_urdf.py", "--help"])
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    AppLauncher.add_app_launcher_args(parser)

    # the probe's own usage line must not leak to stderr ahead of the real help output
    assert capsys.readouterr().err == ""
    assert "--visualizer" in parser._option_string_actions
    assert "--device" in parser._option_string_actions

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
    assert "app_launcher arguments" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["none", "None"])
def test_visualizer_none_parsing(value: str):
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--viz", value])
    assert args.visualizer is None
    assert args.visualizer_explicit is True
