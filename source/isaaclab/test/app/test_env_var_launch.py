# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import pytest

from isaaclab.app import AppLauncher


@pytest.mark.usefixtures("mocker")
def test_livestream_launch_with_env_vars(mocker):
    """Test launching with environment variables."""
    # Mock the environment variables
    mocker.patch.dict(os.environ, {"LIVESTREAM": "1", "HEADLESS": "1"})
    # everything defaults to None
    app = AppLauncher().app

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
