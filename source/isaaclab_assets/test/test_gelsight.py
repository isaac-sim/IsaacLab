# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for predefined GelSight sensor configurations."""

import os

import pytest

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_assets.sensors import GELSIGHT_MINI_CFG


@pytest.mark.parametrize("file_name", [GELSIGHT_MINI_CFG.background_path, GELSIGHT_MINI_CFG.calib_path])
def test_gelsight_mini_render_data_is_available(file_name, tmp_path):
    """The GelSight Mini render data resolves to files on the asset server, as the renderer loads them."""
    remote_path = os.path.join(GELSIGHT_MINI_CFG.base_data_path, GELSIGHT_MINI_CFG.sensor_data_dir_name, file_name)

    assert os.path.isfile(retrieve_file_path(remote_path, download_dir=str(tmp_path)))
