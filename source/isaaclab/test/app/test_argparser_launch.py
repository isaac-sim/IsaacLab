# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pytest

from isaaclab.app import AppLauncher


@pytest.mark.usefixtures("mocker")
def test_livestream_launch_with_argparser(mocker):
    """Test launching with argparser arguments."""
    # Mock the parse_args method
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(livestream=1, headless=True))
    # create argparser
    parser = argparse.ArgumentParser()
    # add app launcher arguments
    AppLauncher.add_app_launcher_args(parser)
    # check that argparser has the mandatory arguments
    for name in AppLauncher._APPLAUNCHER_CFG_INFO:
        assert parser._option_string_actions[f"--{name}"]
    # parse args
    mock_args = parser.parse_args()
    # everything defaults to None
    app = AppLauncher(mock_args).app

    from isaaclab.app.settings_manager import get_settings_manager

    settings = get_settings_manager()
    assert settings.get("/app/window/enabled") is False
    assert settings.get("/app/livestream/enabled") is True

    # close the app on exit
    app.close()


def test_headless_deprecated_arg_parsing():
    """Test deprecated --headless CLI argument parsing."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless"])

    assert args.headless is True
    assert args.headless_explicit is True


def test_visualizer_alias_parsing():
    """Test --viz alias parsing and explicit tracking."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--viz", "newton", "rerun"])

    assert args.visualizer == ["newton", "rerun"]
    assert args.visualizer_explicit is True


def test_visualizer_none_parsing():
    """Test --visualizer none parsing and explicit tracking."""
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--visualizer", "none"])

    assert args.visualizer == ["none"]
    assert args.visualizer_explicit is True


def test_visualizer_none_cannot_be_combined_with_others():
    """Test that 'none' cannot be combined with other visualizer tokens."""
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._livestream = 0
    launcher_args = {
        "headless": False,
        "headless_explicit": False,
        "visualizer": ["none", "kit"],
        "visualizer_explicit": True,
    }

    with pytest.raises(ValueError, match="cannot be combined"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("HEADLESS", "0")
            launcher._resolve_headless_settings(launcher_args, livestream_arg=-1, livestream_env=0)


def test_visualizer_none_forces_headless_without_headless_flag():
    """Test that explicit '--visualizer none' implies headless when livestream is disabled."""
    launcher = AppLauncher.__new__(AppLauncher)
    launcher._livestream = 0
    launcher_args = {
        "headless": False,
        "headless_explicit": False,
        "visualizer": ["none"],
        "visualizer_explicit": True,
    }

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("HEADLESS", "0")
        launcher._resolve_headless_settings(launcher_args, livestream_arg=-1, livestream_env=0)

    assert launcher._headless is True
    assert launcher_args["visualizer"] == []
