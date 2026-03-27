# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for GelSight utility functions - primarily focused on GelsightRender."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import os
import tempfile

import cv2
import numpy as np
import pytest
import torch

from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import GelsightRender
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_cfg import GelSightRenderCfg


def test_gelsight_render_custom_path_missing_file():
    """Test initializing GelsightRender with custom path when file doesn't exist."""
    # Assuming 'non_existent_path' is treated as a local path or Nucleus path
    # If we pass a path that definitely doesn't exist locally or on Nucleus, it should fail
    cfg = GelSightRenderCfg(
        base_data_path="non_existent_path",
        sensor_data_dir_name="dummy",
        image_height=100,
        image_width=100,
        mm_per_pixel=0.1,
    )
    # This should raise FileNotFoundError because the directory/files won't exist
    with pytest.raises(FileNotFoundError):
        GelsightRender(cfg, device="cpu")


def test_gelsight_render_custom_path_success():
    """Test initializing GelsightRender with valid custom path and files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = "gelsight_r15_data"
        full_dir = os.path.join(tmpdir, data_dir)
        os.makedirs(full_dir, exist_ok=True)

        # Create dummy configuration
        width, height = 10, 10
        cfg = GelSightRenderCfg(
            base_data_path=tmpdir,
            sensor_data_dir_name=data_dir,
            image_width=width,
            image_height=height,
            num_bins=5,
            mm_per_pixel=0.1,
        )

        # 1. Create dummy background image
        bg_path = os.path.join(full_dir, cfg.background_path)
        dummy_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.imwrite(bg_path, dummy_img)

        # 2. Create dummy calibration file
        calib_path = os.path.join(full_dir, cfg.calib_path)
        # Calibration gradients shape: (num_bins, num_bins, 6)
        dummy_grad = np.zeros((cfg.num_bins, cfg.num_bins, 6), dtype=np.float32)
        np.savez(calib_path, grad_r=dummy_grad, grad_g=dummy_grad, grad_b=dummy_grad)

        # Test initialization
        try:
            device = torch.device("cpu")
            render = GelsightRender(cfg, device=device)
            assert render is not None
            assert render.device == device
            # Verify loaded background dimensions
            assert render.background.shape == (height, width, 3)
        except Exception as e:
            pytest.fail(f"GelsightRender initialization failed with valid custom files: {e}")


@pytest.fixture
def gelsight_render_setup():
    """Fixture to set up GelsightRender for testing with default (Nucleus/Cache) files."""
    # Use default GelSight R1.5 configuration
    cfg = GelSightRenderCfg(
        sensor_data_dir_name="gelsight_r15_data", image_height=320, image_width=240, mm_per_pixel=0.0877
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create render instance
    try:
        render = GelsightRender(cfg, device=device)
        yield render, device
    except Exception as e:
        # If initialization fails (e.g., missing data files), skip tests
        pytest.skip(f"GelsightRender initialization failed (likely network/Nucleus issue): {e}")


def test_gelsight_render_initialization(gelsight_render_setup):
    """Test GelsightRender initialization with default files."""
    render, device = gelsight_render_setup

    # Check that render object was created
    assert render is not None
    assert render.device == device

    # Check that background was loaded (non-empty)
    assert render.background is not None
    assert render.background.size > 0
    assert render.background.shape[2] == 3  # RGB


def test_gelsight_render_compute(gelsight_render_setup):
    """Test the render method of GelsightRender."""
    render, device = gelsight_render_setup

    # Create dummy height map
    height, width = render.cfg.image_height, render.cfg.image_width
    height_map = torch.zeros((1, height, width), device=device, dtype=torch.float32)

    # Add some features to height map
    height_map[0, height // 4 : height // 2, width // 4 : width // 2] = 0.001  # 1mm bump

    # Render
    output = render.render(height_map)

    # Check output
    assert output is not None
    assert output.shape == (1, height, width, 3)
    assert output.dtype == torch.uint8
