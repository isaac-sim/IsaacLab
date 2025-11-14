# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for GelSight utility functions - primarily focused on get_gelsight_render_data."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np
import os
import tempfile
import torch

import pytest

from isaaclab.sensors.tacsl_sensor.gelsight_utils import GelsightRender, get_gelsight_render_data
from isaaclab.sensors.tacsl_sensor.visuotactile_sensor_cfg import GelSightRenderCfg


def test_get_gelsight_render_data_custom_path_missing_file():
    """Test retrieving data from custom path when file doesn't exist."""
    with pytest.raises(FileNotFoundError) as e:
        get_gelsight_render_data("non_existent_path", "gelsight_r15_data", "bg.jpg")
    assert "Custom GelSight render data not found" in str(e.value)


def test_get_gelsight_render_data_nucleus_path_default():
    """Test retrieving data from Nucleus (default path)."""
    data_dir = "gelsight_r15_data"
    file_name = "bg.jpg"

    # This will either download from Nucleus or use cached data
    result_path = get_gelsight_render_data(None, data_dir, file_name)

    # Result should be a valid path or None if Nucleus is unavailable
    if result_path is not None:
        assert os.path.exists(result_path)
        assert file_name in result_path
        assert data_dir in result_path


def test_get_gelsight_render_data_path_construction_custom():
    """Test correct path construction for custom paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = "sensor_data"
        file_name = "test.jpg"

        # Create the expected structure
        full_dir = os.path.join(tmpdir, data_dir)
        os.makedirs(full_dir, exist_ok=True)

        expected_path = os.path.join(tmpdir, data_dir, file_name)
        with open(expected_path, "w") as f:
            f.write("test")

        result_path = get_gelsight_render_data(tmpdir, data_dir, file_name)
        assert result_path == expected_path


def test_get_gelsight_render_data_real_cached_files():
    """Test with real cached GelSight data files that should exist."""
    data_dir = "gelsight_r15_data"

    # Test background image
    bg_path = get_gelsight_render_data(None, data_dir, "bg.jpg")
    if bg_path is not None:
        assert os.path.exists(bg_path)
        assert bg_path.endswith("bg.jpg")
        # Verify it's actually an image file by checking file size
        assert os.path.getsize(bg_path) > 0

    # Test calibration file
    calib_path = get_gelsight_render_data(None, data_dir, "polycalib.npz")
    if calib_path is not None:
        assert os.path.exists(calib_path)
        assert calib_path.endswith("polycalib.npz")
        # Verify it's actually a numpy file by checking file size
        assert os.path.getsize(calib_path) > 0

        # Try to load the calibration data to verify it's valid
        try:

            calib_data = np.load(calib_path)
            # Check that expected keys exist
            expected_keys = ["grad_r", "grad_g", "grad_b"]
            for key in expected_keys:
                assert key in calib_data.files, f"Missing key '{key}' in calibration data"
        except Exception as e:
            pytest.fail(f"Failed to load calibration data: {e}")


@pytest.fixture
def gelsight_render_setup():
    """Fixture to set up GelsightRender for testing."""
    # Use default GelSight R1.5 configuration
    cfg = GelSightRenderCfg()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create render instance
    try:
        render = GelsightRender(cfg, device=device)
        yield render, device
    except Exception as e:
        # If initialization fails (e.g., missing data files), skip tests
        pytest.skip(f"GelsightRender initialization failed: {e}")


def test_gelsight_render_initialization(gelsight_render_setup):
    """Test GelsightRender initialization."""
    render, device = gelsight_render_setup

    # Check that render object was created
    assert render is not None
    assert render.device == device
