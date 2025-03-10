# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

import os
import shutil
import tempfile
import torch
import unittest
import uuid

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler


def create_test_episode(device):
    """create a test episode with dummy data."""
    test_episode = EpisodeData()

    test_episode.seed = 0
    test_episode.success = True

    test_episode.add("initial_state", torch.tensor([1, 2, 3], device=device))

    test_episode.add("actions", torch.tensor([1, 2, 3], device=device))
    test_episode.add("actions", torch.tensor([4, 5, 6], device=device))
    test_episode.add("actions", torch.tensor([7, 8, 9], device=device))

    test_episode.add("obs/policy/term1", torch.tensor([1, 2, 3, 4, 5], device=device))
    test_episode.add("obs/policy/term1", torch.tensor([6, 7, 8, 9, 10], device=device))
    test_episode.add("obs/policy/term1", torch.tensor([11, 12, 13, 14, 15], device=device))

    return test_episode


class TestHDF5DatasetFileHandler(unittest.TestCase):
    """Test HDF5 dataset filer handler implementation."""

    """
    Test cases for HDF5DatasetFileHandler class.
    """

    def setUp(self):
        # create a temporary directory to store the test datasets
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # delete the temporary directory after the test
        shutil.rmtree(self.temp_dir)

    def test_create_dataset_file(self):
        """Test creating a new dataset file."""
        # create a dataset file given a file name with extension
        dataset_file_path = os.path.join(self.temp_dir, f"{uuid.uuid4()}.hdf5")
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.create(dataset_file_path, "test_env_name")
        dataset_file_handler.close()

        # check if the dataset is created
        self.assertTrue(os.path.exists(dataset_file_path))

        # create a dataset file given a file name without extension
        dataset_file_path = os.path.join(self.temp_dir, f"{uuid.uuid4()}")
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.create(dataset_file_path, "test_env_name")
        dataset_file_handler.close()

        # check if the dataset is created
        self.assertTrue(os.path.exists(dataset_file_path + ".hdf5"))

    def test_write_and_load_episode(self):
        """Test writing and loading an episode to and from the dataset file."""
        for device in ("cuda:0", "cpu"):
            with self.subTest(device=device):
                dataset_file_path = os.path.join(self.temp_dir, f"{uuid.uuid4()}.hdf5")
                dataset_file_handler = HDF5DatasetFileHandler()
                dataset_file_handler.create(dataset_file_path, "test_env_name")

                test_episode = create_test_episode(device)

                # write the episode to the dataset
                dataset_file_handler.write_episode(test_episode)
                dataset_file_handler.flush()

                self.assertEqual(dataset_file_handler.get_num_episodes(), 1)

                # write the episode again to test writing 2nd episode
                dataset_file_handler.write_episode(test_episode)
                dataset_file_handler.flush()

                self.assertEqual(dataset_file_handler.get_num_episodes(), 2)

                # close the dataset file to prepare for testing the load function
                dataset_file_handler.close()

                # load the episode from the dataset
                dataset_file_handler = HDF5DatasetFileHandler()
                dataset_file_handler.open(dataset_file_path)

                self.assertEqual(dataset_file_handler.get_env_name(), "test_env_name")

                loaded_episode_names = dataset_file_handler.get_episode_names()
                self.assertEqual(len(list(loaded_episode_names)), 2)

                for episode_name in loaded_episode_names:
                    loaded_episode = dataset_file_handler.load_episode(episode_name, device=device)
                    self.assertEqual(loaded_episode.env_id, "test_env_name")
                    self.assertEqual(loaded_episode.seed, test_episode.seed)
                    self.assertEqual(loaded_episode.success, test_episode.success)

                    self.assertTrue(torch.equal(loaded_episode.get_initial_state(), test_episode.get_initial_state()))

                    for action in test_episode.data["actions"]:
                        self.assertTrue(torch.equal(loaded_episode.get_next_action(), action))

                dataset_file_handler.close()


if __name__ == "__main__":
    run_tests()
