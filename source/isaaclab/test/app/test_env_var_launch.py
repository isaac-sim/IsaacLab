# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
