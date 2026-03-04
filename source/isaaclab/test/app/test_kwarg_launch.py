# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

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
