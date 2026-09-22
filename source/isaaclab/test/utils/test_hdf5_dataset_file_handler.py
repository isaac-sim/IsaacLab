# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import h5py
import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

pytestmark = pytest.mark.unit


def read_env_args(path) -> dict:
    with h5py.File(path, "r") as dataset_file:
        return json.loads(dataset_file["data"].attrs["env_args"])


def test_create_appends_extension_and_resets_env_args(tmp_path):
    handler = HDF5DatasetFileHandler()
    handler.create(str(tmp_path / "first"), "first_env")
    handler.add_env_args({"custom_arg": "custom_value"})
    handler.close()
    assert (tmp_path / "first.hdf5").is_file()
    assert read_env_args(tmp_path / "first.hdf5") == {"env_name": "first_env", "type": 2, "custom_arg": "custom_value"}

    # reusing the handler for a new dataset does not leak the previous env args
    handler.create(str(tmp_path / "second.hdf5"), "second_env")
    handler.close()
    assert read_env_args(tmp_path / "second.hdf5") == {"env_name": "second_env", "type": 2}

    # env args can be extended after reopening
    handler.open(str(tmp_path / "second.hdf5"), mode="r+")
    handler.add_env_args({"custom_arg": "custom_value"})
    handler.close()
    assert read_env_args(tmp_path / "second.hdf5") == {
        "env_name": "second_env",
        "type": 2,
        "custom_arg": "custom_value",
    }


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_write_and_load_episode(tmp_path, device):
    episode = EpisodeData()
    episode.seed = 0
    episode.success = True
    episode.add("initial_state", torch.tensor([1, 2, 3], device=device))
    actions = torch.arange(1, 10, device=device).reshape(3, 3)
    for action in actions:
        episode.add("actions", action)
        episode.add("obs/policy/term1", action.repeat(2))
    episode.pre_export()

    path = str(tmp_path / "dataset.hdf5")
    handler = HDF5DatasetFileHandler()
    handler.create(path, "test_env_name")
    for expected_count in (1, 2):
        handler.write_episode(episode)
        handler.flush()
        assert handler.get_num_episodes() == expected_count
    handler.close()

    handler.open(path)
    assert handler.get_env_name() == "test_env_name"
    episode_names = list(handler.get_episode_names())
    assert len(episode_names) == 2
    for name in episode_names:
        loaded = handler.load_episode(name, device=device)
        assert (loaded.env_id, loaded.seed, loaded.success) == ("test_env_name", 0, True)
        torch.testing.assert_close(loaded.get_initial_state(), episode.get_initial_state())
        for action in actions:
            torch.testing.assert_close(loaded.get_next_action(), action)
        assert loaded.get_next_action() is None
    handler.close()
