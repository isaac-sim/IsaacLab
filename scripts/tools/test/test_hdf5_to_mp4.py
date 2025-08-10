# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for HDF5 to MP4 conversion script."""

import h5py
import numpy as np
import os
import tempfile

import pytest

from scripts.tools.hdf5_to_mp4 import get_num_demos, main, write_demo_to_mp4


@pytest.fixture(scope="class")
def temp_hdf5_file():
    """Create temporary HDF5 file with test data."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    with h5py.File(temp_file.name, "w") as h5f:
        # Create test data structure
        for demo_id in range(2):  # Create 2 demos
            demo_group = h5f.create_group(f"data/demo_{demo_id}/obs")

            # Create RGB frames (2 frames per demo)
            rgb_data = np.random.randint(0, 255, (2, 704, 1280, 3), dtype=np.uint8)
            demo_group.create_dataset("table_cam", data=rgb_data)

            # Create segmentation frames
            seg_data = np.random.randint(0, 255, (2, 704, 1280, 4), dtype=np.uint8)
            demo_group.create_dataset("table_cam_segmentation", data=seg_data)

            # Create normal maps
            normals_data = np.random.rand(2, 704, 1280, 3).astype(np.float32)
            demo_group.create_dataset("table_cam_normals", data=normals_data)

            # Create depth maps
            depth_data = np.random.rand(2, 704, 1280, 1).astype(np.float32)
            demo_group.create_dataset("table_cam_depth", data=depth_data)

    yield temp_file.name
    # Cleanup
    os.remove(temp_file.name)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)


class TestHDF5ToMP4:
    """Test cases for HDF5 to MP4 conversion functionality."""

    def test_get_num_demos(self, temp_hdf5_file):
        """Test the get_num_demos function."""
        num_demos = get_num_demos(temp_hdf5_file)
        assert num_demos == 2

    def test_write_demo_to_mp4_rgb(self, temp_hdf5_file, temp_output_dir):
        """Test writing RGB frames to MP4."""
        write_demo_to_mp4(temp_hdf5_file, 0, "data/demo_0/obs", "table_cam", temp_output_dir, 704, 1280)

        output_file = os.path.join(temp_output_dir, "demo_0_table_cam.mp4")
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0

    def test_write_demo_to_mp4_segmentation(self, temp_hdf5_file, temp_output_dir):
        """Test writing segmentation frames to MP4."""
        write_demo_to_mp4(temp_hdf5_file, 0, "data/demo_0/obs", "table_cam_segmentation", temp_output_dir, 704, 1280)

        output_file = os.path.join(temp_output_dir, "demo_0_table_cam_segmentation.mp4")
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0

    def test_write_demo_to_mp4_normals(self, temp_hdf5_file, temp_output_dir):
        """Test writing normal maps to MP4."""
        write_demo_to_mp4(temp_hdf5_file, 0, "data/demo_0/obs", "table_cam_normals", temp_output_dir, 704, 1280)

        output_file = os.path.join(temp_output_dir, "demo_0_table_cam_normals.mp4")
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0

    def test_write_demo_to_mp4_shaded_segmentation(self, temp_hdf5_file, temp_output_dir):
        """Test writing shaded_segmentation frames to MP4."""
        write_demo_to_mp4(
            temp_hdf5_file,
            0,
            "data/demo_0/obs",
            "table_cam_shaded_segmentation",
            temp_output_dir,
            704,
            1280,
        )

        output_file = os.path.join(temp_output_dir, "demo_0_table_cam_shaded_segmentation.mp4")
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0

    def test_write_demo_to_mp4_depth(self, temp_hdf5_file, temp_output_dir):
        """Test writing depth maps to MP4."""
        write_demo_to_mp4(temp_hdf5_file, 0, "data/demo_0/obs", "table_cam_depth", temp_output_dir, 704, 1280)

        output_file = os.path.join(temp_output_dir, "demo_0_table_cam_depth.mp4")
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0

    def test_write_demo_to_mp4_invalid_demo(self, temp_hdf5_file, temp_output_dir):
        """Test writing with invalid demo ID."""
        with pytest.raises(KeyError):
            write_demo_to_mp4(
                temp_hdf5_file,
                999,  # Invalid demo ID
                "data/demo_999/obs",
                "table_cam",
                temp_output_dir,
                704,
                1280,
            )

    def test_write_demo_to_mp4_invalid_key(self, temp_hdf5_file, temp_output_dir):
        """Test writing with invalid input key."""
        with pytest.raises(KeyError):
            write_demo_to_mp4(temp_hdf5_file, 0, "data/demo_0/obs", "invalid_key", temp_output_dir, 704, 1280)

    def test_main_function(self, temp_hdf5_file, temp_output_dir):
        """Test the main function."""
        # Mock command line arguments
        import sys

        original_argv = sys.argv
        sys.argv = [
            "hdf5_to_mp4.py",
            "--input_file",
            temp_hdf5_file,
            "--output_dir",
            temp_output_dir,
            "--input_keys",
            "table_cam",
            "table_cam_segmentation",
            "--video_height",
            "704",
            "--video_width",
            "1280",
            "--framerate",
            "30",
        ]

        try:
            main()

            # Check if output files were created
            expected_files = [
                "demo_0_table_cam.mp4",
                "demo_0_table_cam_segmentation.mp4",
                "demo_1_table_cam.mp4",
                "demo_1_table_cam_segmentation.mp4",
            ]

            for file in expected_files:
                output_file = os.path.join(temp_output_dir, file)
                assert os.path.exists(output_file)
                assert os.path.getsize(output_file) > 0
        finally:
            # Restore original argv
            sys.argv = original_argv
