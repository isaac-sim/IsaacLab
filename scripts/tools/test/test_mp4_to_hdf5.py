# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for MP4 to HDF5 conversion script."""

import h5py
import numpy as np
import os
import tempfile
import unittest

import cv2

from scripts.tools.mp4_to_hdf5 import get_frames_from_mp4, main, process_video_and_demo


class TestMP4ToHDF5(unittest.TestCase):
    """Test cases for MP4 to HDF5 conversion functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all test methods."""
        # Create temporary HDF5 file with test data
        cls.temp_hdf5_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        with h5py.File(cls.temp_hdf5_file.name, "w") as h5f:
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

        # Create temporary MP4 files
        cls.temp_videos_dir = tempfile.mkdtemp()
        cls.video_paths = []
        for demo_id in range(2):
            video_path = os.path.join(cls.temp_videos_dir, f"demo_{demo_id}_table_cam.mp4")
            cls.video_paths.append(video_path)

            # Create a test video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(video_path, fourcc, 30, (1280, 704))

            # Write some random frames
            for _ in range(10):
                frame = np.random.randint(0, 255, (704, 1280, 3), dtype=np.uint8)
                video.write(frame)
            video.release()

    def setUp(self):
        """Set up test fixtures that are created for each test method."""
        self.temp_output_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove the temporary output file
        os.remove(self.temp_output_file.name)

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures that are shared across all test methods."""
        # Remove the temporary HDF5 file
        os.remove(cls.temp_hdf5_file.name)

        # Remove temporary videos and directory
        for video_path in cls.video_paths:
            os.remove(video_path)
        os.rmdir(cls.temp_videos_dir)

    def test_get_frames_from_mp4(self):
        """Test extracting frames from MP4 video."""
        frames = get_frames_from_mp4(self.video_paths[0])

        # Check frame properties
        self.assertEqual(frames.shape[0], 10)  # Number of frames
        self.assertEqual(frames.shape[1:], (704, 1280, 3))  # Frame dimensions
        self.assertEqual(frames.dtype, np.uint8)  # Data type

    def test_get_frames_from_mp4_resize(self):
        """Test extracting frames with resizing."""
        target_height, target_width = 352, 640
        frames = get_frames_from_mp4(self.video_paths[0], target_height, target_width)

        # Check resized frame properties
        self.assertEqual(frames.shape[0], 10)  # Number of frames
        self.assertEqual(frames.shape[1:], (target_height, target_width, 3))  # Resized dimensions
        self.assertEqual(frames.dtype, np.uint8)  # Data type

    def test_process_video_and_demo(self):
        """Test processing a single video and creating a new demo."""
        with h5py.File(self.temp_hdf5_file.name, "r") as f_in, h5py.File(self.temp_output_file.name, "w") as f_out:
            process_video_and_demo(f_in, f_out, self.video_paths[0], 0, 2)

            # Check if new demo was created with correct data
            self.assertIn("data/demo_2", f_out)
            self.assertIn("data/demo_2/actions", f_out)
            self.assertIn("data/demo_2/obs/eef_pos", f_out)
            self.assertIn("data/demo_2/obs/eef_quat", f_out)
            self.assertIn("data/demo_2/obs/gripper_pos", f_out)
            self.assertIn("data/demo_2/obs/table_cam", f_out)
            self.assertIn("data/demo_2/obs/wrist_cam", f_out)

            # Check data shapes
            self.assertEqual(f_out["data/demo_2/actions"].shape, (10, 7))
            self.assertEqual(f_out["data/demo_2/obs/eef_pos"].shape, (10, 3))
            self.assertEqual(f_out["data/demo_2/obs/eef_quat"].shape, (10, 4))
            self.assertEqual(f_out["data/demo_2/obs/gripper_pos"].shape, (10, 1))
            self.assertEqual(f_out["data/demo_2/obs/table_cam"].shape, (10, 704, 1280, 3))
            self.assertEqual(f_out["data/demo_2/obs/wrist_cam"].shape, (10, 704, 1280, 3))

            # Check attributes
            self.assertEqual(f_out["data/demo_2"].attrs["num_samples"], 10)

    def test_main_function(self):
        """Test the main function."""
        # Mock command line arguments
        import sys

        original_argv = sys.argv
        sys.argv = [
            "mp4_to_hdf5.py",
            "--input_file",
            self.temp_hdf5_file.name,
            "--videos_dir",
            self.temp_videos_dir,
            "--output_file",
            self.temp_output_file.name,
        ]

        try:
            main()

            # Check if output file was created with correct data
            with h5py.File(self.temp_output_file.name, "r") as f:
                # Check if original demos were copied
                self.assertIn("data/demo_0", f)
                self.assertIn("data/demo_1", f)

                # Check if new demos were created
                self.assertIn("data/demo_2", f)
                self.assertIn("data/demo_3", f)

                # Check data in new demos
                for demo_id in [2, 3]:
                    self.assertIn(f"data/demo_{demo_id}/actions", f)
                    self.assertIn(f"data/demo_{demo_id}/obs/eef_pos", f)
                    self.assertIn(f"data/demo_{demo_id}/obs/eef_quat", f)
                    self.assertIn(f"data/demo_{demo_id}/obs/gripper_pos", f)
                    self.assertIn(f"data/demo_{demo_id}/obs/table_cam", f)
                    self.assertIn(f"data/demo_{demo_id}/obs/wrist_cam", f)
        finally:
            # Restore original argv
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
