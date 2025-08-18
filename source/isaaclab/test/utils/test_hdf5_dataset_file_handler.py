# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

import os
import shutil
import tempfile
import torch
import uuid

import pytest

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


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test datasets."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # cleanup after tests
    shutil.rmtree(temp_dir)


def test_create_dataset_file(temp_dir):
    """Test creating a new dataset file."""
    # create a dataset file given a file name with extension
    dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.hdf5")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.create(dataset_file_path, "test_env_name")
    dataset_file_handler.close()

    # check if the dataset is created
    assert os.path.exists(dataset_file_path)

    # create a dataset file given a file name without extension
    dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.create(dataset_file_path, "test_env_name")
    dataset_file_handler.close()

    # check if the dataset is created
    assert os.path.exists(dataset_file_path + ".hdf5")


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_write_and_load_episode(temp_dir, device):
    """Test writing and loading an episode to and from the dataset file."""
    dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.hdf5")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.create(dataset_file_path, "test_env_name")

    test_episode = create_test_episode(device)

    # write the episode to the dataset
    dataset_file_handler.write_episode(test_episode)
    dataset_file_handler.flush()

    assert dataset_file_handler.get_num_episodes() == 1

    # write the episode again to test writing 2nd episode
    dataset_file_handler.write_episode(test_episode)
    dataset_file_handler.flush()

    assert dataset_file_handler.get_num_episodes() == 2

    # close the dataset file to prepare for testing the load function
    dataset_file_handler.close()

    # load the episode from the dataset
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(dataset_file_path)

    assert dataset_file_handler.get_env_name() == "test_env_name"

    loaded_episode_names = dataset_file_handler.get_episode_names()
    assert len(list(loaded_episode_names)) == 2

    for episode_name in loaded_episode_names:
        loaded_episode = dataset_file_handler.load_episode(episode_name, device=device)
        assert loaded_episode.env_id == "test_env_name"
        assert loaded_episode.seed == test_episode.seed
        assert loaded_episode.success == test_episode.success

        assert torch.equal(loaded_episode.get_initial_state(), test_episode.get_initial_state())

        for action in test_episode.data["actions"]:
            assert torch.equal(loaded_episode.get_next_action(), action)

    dataset_file_handler.close()
