# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for predefined GelSight sensor configurations."""

from isaaclab_assets.sensors import GELSIGHT_MINI_CFG


def test_gelsight_mini_uses_available_background_path():
    """Verify the GelSight Mini background resolves to the available render data."""
    assert f"{GELSIGHT_MINI_CFG.sensor_data_dir_name}/{GELSIGHT_MINI_CFG.background_path}" == (
        "gelsight_r15_data/bg.jpg"
    )
