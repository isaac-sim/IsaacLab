# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pytest
from unittest import mock

from isaaclab.app import AppLauncher, run_tests

if AppLauncher.instance():
    AppLauncher.clear_instance()

@mock.patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(livestream=1))
def test_livestream_launch_with_argparser(mock_args):
    """Test launching with argparser arguments."""
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

    # import settings
    import carb

    # acquire settings interface
    carb_settings_iface = carb.settings.get_settings()
    # check settings
    # -- no-gui mode
    assert carb_settings_iface.get("/app/window/enabled") is False
    # -- livestream
    assert carb_settings_iface.get("/app/livestream/enabled") is True

    # close the app on exit
    app.close()


if __name__ == "__main__":
    run_tests()
