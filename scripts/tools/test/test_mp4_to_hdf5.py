# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for MP4 to HDF5 conversion script."""

import h5py
import numpy as np
import os
import tempfile

import cv2
import pytest

from scripts.tools.mp4_to_hdf5 import get_frames_from_mp4, main, process_video_and_demo


@pytest.fixture(scope="class")
def temp_hdf5_file():
    """Create temporary HDF5 file with test data."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    with h5py.File(temp_file.name, "w") as h5f:
        # Create test data structure for 2 demos
        for demo_id in range(2):
            demo_group = h5f.create_group(f"data/demo_{demo_id}")
            obs_group = demo_group.create_group("obs")

            # Create actions data
            actions_data = np.random.rand(10, 7).astype(np.float32)
            demo_group.create_dataset("actions", data=actions_data)

            # Create robot state data
            eef_pos_data = np.random.rand(10, 3).astype(np.float32)
            eef_quat_data = np.random.rand(10, 4).astype(np.float32)
            gripper_pos_data = np.random.rand(10, 1).astype(np.float32)
            obs_group.create_dataset("eef_pos", data=eef_pos_data)
            obs_group.create_dataset("eef_quat", data=eef_quat_data)
            obs_group.create_dataset("gripper_pos", data=gripper_pos_data)

            # Create camera data
            table_cam_data = np.random.randint(0, 255, (10, 704, 1280, 3), dtype=np.uint8)
            wrist_cam_data = np.random.randint(0, 255, (10, 704, 1280, 3), dtype=np.uint8)
            obs_group.create_dataset("table_cam", data=table_cam_data)
            obs_group.create_dataset("wrist_cam", data=wrist_cam_data)

            # Set attributes
            demo_group.attrs["num_samples"] = 10

    yield temp_file.name
    # Cleanup
    os.remove(temp_file.name)


@pytest.fixture(scope="class")
def temp_videos_dir():
    """Create temporary MP4 files."""
    temp_dir = tempfile.mkdtemp()
    video_paths = []

    for demo_id in range(2):
        video_path = os.path.join(temp_dir, f"demo_{demo_id}_table_cam.mp4")
        video_paths.append(video_path)

        # Create a test video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, 30, (1280, 704))

        # Write some random frames
        for _ in range(10):
            frame = np.random.randint(0, 255, (704, 1280, 3), dtype=np.uint8)
            video.write(frame)
        video.release()

    yield temp_dir, video_paths

    # Cleanup
    for video_path in video_paths:
        os.remove(video_path)
    os.rmdir(temp_dir)


@pytest.fixture
def temp_output_file():
    """Create temporary output file."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    yield temp_file.name
    # Cleanup
    os.remove(temp_file.name)


class TestMP4ToHDF5:
    """Test cases for MP4 to HDF5 conversion functionality."""

    def test_get_frames_from_mp4(self, temp_videos_dir):
        """Test extracting frames from MP4 video."""
        _, video_paths = temp_videos_dir
        frames = get_frames_from_mp4(video_paths[0])

        # Check frame properties
        assert frames.shape[0] == 10  # Number of frames
        assert frames.shape[1:] == (704, 1280, 3)  # Frame dimensions
        assert frames.dtype == np.uint8  # Data type

    def test_get_frames_from_mp4_resize(self, temp_videos_dir):
        """Test extracting frames with resizing."""
        _, video_paths = temp_videos_dir
        target_height, target_width = 352, 640
        frames = get_frames_from_mp4(video_paths[0], target_height, target_width)

        # Check resized frame properties
        assert frames.shape[0] == 10  # Number of frames
        assert frames.shape[1:] == (target_height, target_width, 3)  # Resized dimensions
        assert frames.dtype == np.uint8  # Data type

    def test_process_video_and_demo(self, temp_hdf5_file, temp_videos_dir, temp_output_file):
        """Test processing a single video and creating a new demo."""
        _, video_paths = temp_videos_dir
        with h5py.File(temp_hdf5_file, "r") as f_in, h5py.File(temp_output_file, "w") as f_out:
            process_video_and_demo(f_in, f_out, video_paths[0], 0, 2)

            # Check if new demo was created with correct data
            assert "data/demo_2" in f_out
            assert "data/demo_2/actions" in f_out
            assert "data/demo_2/obs/eef_pos" in f_out
            assert "data/demo_2/obs/eef_quat" in f_out
            assert "data/demo_2/obs/gripper_pos" in f_out
            assert "data/demo_2/obs/table_cam" in f_out
            assert "data/demo_2/obs/wrist_cam" in f_out

            # Check data shapes
            assert f_out["data/demo_2/actions"].shape == (10, 7)
            assert f_out["data/demo_2/obs/eef_pos"].shape == (10, 3)
            assert f_out["data/demo_2/obs/eef_quat"].shape == (10, 4)
            assert f_out["data/demo_2/obs/gripper_pos"].shape == (10, 1)
            assert f_out["data/demo_2/obs/table_cam"].shape == (10, 704, 1280, 3)
            assert f_out["data/demo_2/obs/wrist_cam"].shape == (10, 704, 1280, 3)

            # Check attributes
            assert f_out["data/demo_2"].attrs["num_samples"] == 10

    def test_main_function(self, temp_hdf5_file, temp_videos_dir, temp_output_file):
        """Test the main function."""
        # Mock command line arguments
        import sys

        original_argv = sys.argv
        sys.argv = [
            "mp4_to_hdf5.py",
            "--input_file",
            temp_hdf5_file,
            "--videos_dir",
            temp_videos_dir[0],
            "--output_file",
            temp_output_file,
        ]

        try:
            main()

            # Check if output file was created with correct data
            with h5py.File(temp_output_file, "r") as f:
                # Check if original demos were copied
                assert "data/demo_0" in f
                assert "data/demo_1" in f

                # Check if new demos were created
                assert "data/demo_2" in f
                assert "data/demo_3" in f

                # Check data in new demos
                for demo_id in [2, 3]:
                    assert f"data/demo_{demo_id}/actions" in f
                    assert f"data/demo_{demo_id}/obs/eef_pos" in f
                    assert f"data/demo_{demo_id}/obs/eef_quat" in f
                    assert f"data/demo_{demo_id}/obs/gripper_pos" in f
                    assert f"data/demo_{demo_id}/obs/table_cam" in f
                    assert f"data/demo_{demo_id}/obs/wrist_cam" in f
        finally:
            # Restore original argv
            sys.argv = original_argv
