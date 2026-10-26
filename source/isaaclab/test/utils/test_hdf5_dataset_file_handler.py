# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import json

import h5py
import pytest
import torch

from isaaclab.test.utils import test_devices
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

pytestmark = pytest.mark.unit


def test_create_dataset_file(tmp_path, monkeypatch):
    """Accept absolute paths, omitted extensions and bare filenames."""
    monkeypatch.chdir(tmp_path)
    for path in (str(tmp_path / "absolute.hdf5"), str(tmp_path / "no_extension"), "relative.hdf5"):
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.create(path, "test_env_name")
        dataset_file_handler.close()
    assert {path.name for path in tmp_path.iterdir()} == {"absolute.hdf5", "no_extension.hdf5", "relative.hdf5"}


def test_env_args_follow_active_file(tmp_path):
    """Preserve metadata on reopen, then discard it when reusing the handler for a new file."""
    dataset_file_path = str(tmp_path / "first.hdf5")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.create(dataset_file_path, "test_env_name")
    dataset_file_handler.close()

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(dataset_file_path, mode="r+")
    dataset_file_handler.add_env_args({"custom_arg": "custom_value"})
    dataset_file_handler.close()

    with h5py.File(dataset_file_path, "r") as dataset_file:
        env_args = json.loads(dataset_file["data"].attrs["env_args"])

    assert env_args == {"env_name": "test_env_name", "type": 2, "custom_arg": "custom_value"}

    second_dataset_path = str(tmp_path / "second.hdf5")
    dataset_file_handler.create(second_dataset_path, "second_env")
    dataset_file_handler.close()

    with h5py.File(second_dataset_path, "r") as dataset_file:
        env_args = json.loads(dataset_file["data"].attrs["env_args"])

    assert env_args == {"env_name": "second_env", "type": 2}


def test_convert_dataset_to_xyzw_updates_format_version(tmp_path):
    """Converted datasets should load their XYZW quaternions without converting them again."""
    input_path = str(tmp_path / "legacy.hdf5")
    output_path = str(tmp_path / "converted.hdf5")
    with h5py.File(input_path, "w") as dataset:
        dataset.attrs["format_version"] = 0
        data = dataset.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "test_env_name"})
        episode = data.create_group("demo_0")
        episode.create_dataset("root_pose", data=[[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])

    HDF5DatasetFileHandler.convert_dataset_to_xyzw(input_path, output_path)

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(output_path)
    try:
        assert dataset_file_handler.get_format_version() == 1
        loaded_episode = dataset_file_handler.load_episode("demo_0", device="cpu")
        torch.testing.assert_close(
            loaded_episode.data["root_pose"], torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        )
    finally:
        dataset_file_handler.close()


@pytest.mark.parametrize("device", test_devices())
def test_write_and_load_episode(tmp_path, device):
    """Round-trip episode data and metadata, then play every action through exhaustion."""
    dataset_file_path = str(tmp_path / "episodes.hdf5")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.create(dataset_file_path, "test_env_name")

    test_episode = EpisodeData()
    assert test_episode.is_empty()
    assert test_episode.get_next_action() is None
    test_episode.seed = 0
    test_episode.success = True
    test_episode.data = {
        "initial_state": torch.tensor([[1, 2, 3]], device=device),
        "actions": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device),
        "obs": {"policy": {"term1": torch.arange(1, 16, device=device).reshape(3, 5)}},
    }
    assert not test_episode.is_empty()
    # Reading an absent action must not advance the cursor before data becomes available.
    assert torch.equal(test_episode.get_next_action(), test_episode.data["actions"][0])

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

        assert torch.equal(loaded_episode.get_initial_state(), test_episode.data["initial_state"])

        assert torch.equal(loaded_episode.data["obs"]["policy"]["term1"], test_episode.data["obs"]["policy"]["term1"])

        for action in test_episode.data["actions"]:
            assert torch.equal(loaded_episode.get_next_action(), action)
        assert loaded_episode.get_next_action() is None

    dataset_file_handler.close()
