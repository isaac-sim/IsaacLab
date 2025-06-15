# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import isaaclab.utils.assets as assets_utils


def test_nucleus_connection():
    """Test checking the Nucleus connection."""
    # check nucleus connection
    assert assets_utils.NUCLEUS_ASSET_ROOT_DIR is not None


def test_check_file_path_nucleus():
    """Test checking a file path on the Nucleus server."""
    # robot file path
    usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    # check file path
    assert assets_utils.check_file_path(usd_path) == 2


def test_check_file_path_invalid():
    """Test checking an invalid file path."""
    # robot file path
    usd_path = f"{assets_utils.ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_xyz.usd"
    # check file path
    assert assets_utils.check_file_path(usd_path) == 0
