# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
from collections.abc import Iterable

import h5py
import numpy as np
import torch

from isaaclab.utils.math import convert_quat

from .dataset_file_handler_base import DatasetFileHandlerBase
from .episode_data import EpisodeData

# Current dataset format version
# Version 1: XYZW quaternion format (current)
# Version 0 (or missing): Legacy WXYZ quaternion format
DATASET_FORMAT_VERSION = 1


def convert_pose_quat_wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    """Convert pose quaternion from WXYZ format to XYZW format.

    The pose is expected to have shape (..., 7) where the first 3 elements are position
    and the last 4 elements are the quaternion.

    Args:
        pose: Pose array with shape (..., 7) where quaternion is in WXYZ format.

    Returns:
        Pose array with shape (..., 7) where quaternion is in XYZW format.
    """
    position = pose[..., :3]
    quat_wxyz = pose[..., 3:7]
    quat_xyzw = convert_quat(quat_wxyz, to="xyzw")
    return np.concatenate([position, quat_xyzw], axis=-1)


class HDF5DatasetFileHandler(DatasetFileHandlerBase):
    """HDF5 dataset file handler for storing and loading episode data."""

    def __init__(self):
        """Initializes the HDF5 dataset file handler."""
        self._hdf5_file_stream = None
        self._hdf5_data_group = None
        self._demo_count = 0
        self._env_args = {}

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._hdf5_file_stream = h5py.File(file_path, mode)
        self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._demo_count = len(self._hdf5_data_group)

    def create(self, file_path: str, env_name: str = None):
        """Create a new dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        self._hdf5_file_stream = h5py.File(file_path, "w")

        # Set the dataset format version
        self._hdf5_file_stream.attrs["format_version"] = DATASET_FORMAT_VERSION

        # set up a data group in the file
        self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
        self._hdf5_data_group.attrs["total"] = 0
        self._demo_count = 0

        # set environment arguments
        # the environment type (we use gym environment type) is set to be compatible with robomimic
        # Ref: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_base.py#L15
        env_name = env_name if env_name is not None else ""
        self.add_env_args({"env_name": env_name, "type": 2})

    def __del__(self):
        """Destructor for the file handler."""
        self.close()

    """
    Properties
    """

    def add_env_args(self, env_args: dict):
        """Add environment arguments to the dataset."""
        self._raise_if_not_initialized()
        self._env_args.update(env_args)
        self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def set_env_name(self, env_name: str):
        """Set the environment name."""
        self._raise_if_not_initialized()
        self.add_env_args({"env_name": env_name})

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        self._raise_if_not_initialized()
        env_args = json.loads(self._hdf5_data_group.attrs["env_args"])
        if "env_name" in env_args:
            return env_args["env_name"]
        return None

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        self._raise_if_not_initialized()
        return self._hdf5_data_group.keys()

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return self._demo_count

    @property
    def demo_count(self) -> int:
        """The number of demos collected so far."""
        return self._demo_count

    """
    Operations.
    """

    def get_format_version(self) -> int:
        """Get the dataset format version.

        Returns:
            The format version number. Returns 0 for legacy datasets without version info.
        """
        self._raise_if_not_initialized()
        if "format_version" in self._hdf5_file_stream.attrs:
            return int(self._hdf5_file_stream.attrs["format_version"])
        return 0  # Legacy format

    def is_legacy_quaternion_format(self) -> bool:
        """Check if the dataset uses the legacy WXYZ quaternion format.

        Returns:
            True if the dataset uses WXYZ format (version 0), False if it uses XYZW format.
        """
        return self.get_format_version() < DATASET_FORMAT_VERSION

    def load_episode(
        self, episode_name: str, device: str, convert_legacy_quat: bool | None = None
    ) -> EpisodeData | None:
        """Load episode data from the file.

        Args:
            episode_name: Name of the episode to load.
            device: Device to load tensors to.
            convert_legacy_quat: If True, convert quaternions from legacy WXYZ to XYZW format.
                If None (default), auto-detect based on dataset version.

        Returns:
            The loaded episode data, or None if the episode doesn't exist.
        """
        self._raise_if_not_initialized()
        if episode_name not in self._hdf5_data_group:
            return None

        # Auto-detect if conversion is needed
        if convert_legacy_quat is None:
            convert_legacy_quat = self.is_legacy_quaternion_format()

        episode = EpisodeData()
        h5_episode_group = self._hdf5_data_group[episode_name]

        def load_dataset_helper(group, path=""):
            """Helper method to load dataset that contains recursive dict objects."""
            data = {}
            for key in group:
                current_path = f"{path}/{key}" if path else key
                if isinstance(group[key], h5py.Group):
                    data[key] = load_dataset_helper(group[key], current_path)
                else:
                    # Converting group[key] to numpy array greatly improves the performance
                    # when converting to torch tensor
                    np_data = np.array(group[key])

                    # Convert legacy quaternions if needed
                    if convert_legacy_quat and key == "root_pose" and np_data.shape[-1] == 7:
                        np_data = convert_pose_quat_wxyz_to_xyzw(np_data)

                    data[key] = torch.tensor(np_data, device=device)
            return data

        episode.data = load_dataset_helper(h5_episode_group)

        if "seed" in h5_episode_group.attrs:
            episode.seed = h5_episode_group.attrs["seed"]

        if "success" in h5_episode_group.attrs:
            episode.success = h5_episode_group.attrs["success"]

        episode.env_id = self.get_env_name()

        return episode

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        """Add an episode to the dataset.

        Args:
            episode: The episode data to add.
            demo_id: Custom index for the episode. If None, uses default index.
        """
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        # Use custom demo id if provided, otherwise use default naming
        if demo_id is not None:
            episode_group_name = f"demo_{demo_id}"
        else:
            episode_group_name = f"demo_{self._demo_count}"

        # create episode group with the specified name
        if episode_group_name in self._hdf5_data_group:
            raise ValueError(f"Episode group '{episode_group_name}' already exists in the dataset")
        h5_episode_group = self._hdf5_data_group.create_group(episode_group_name)

        # store number of steps taken
        if "actions" in episode.data:
            h5_episode_group.attrs["num_samples"] = len(episode.data["actions"])
        else:
            h5_episode_group.attrs["num_samples"] = 0

        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed

        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        def create_dataset_helper(group, key, value):
            """Helper method to create dataset that contains recursive dict objects."""
            if isinstance(value, dict):
                key_group = group.create_group(key)
                for sub_key, sub_value in value.items():
                    create_dataset_helper(key_group, sub_key, sub_value)
            else:
                group.create_dataset(key, data=value.cpu().numpy(), compression="gzip")

        for key, value in episode.data.items():
            create_dataset_helper(h5_episode_group, key, value)

        # increment total step counts
        self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]

        # Only increment demo count if using default indexing
        if demo_id is None:
            # increment total demo counts
            self._demo_count += 1

    def flush(self):
        """Flush the episode data to disk."""
        self._raise_if_not_initialized()

        self._hdf5_file_stream.flush()

    def close(self):
        """Close the dataset file handler."""
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.close()
            self._hdf5_file_stream = None

    def _raise_if_not_initialized(self):
        """Raise an error if the dataset file handler is not initialized."""
        if self._hdf5_file_stream is None:
            raise RuntimeError("HDF5 dataset file stream is not initialized")

    @staticmethod
    def convert_dataset_to_xyzw(input_path: str, output_path: str | None = None) -> str:
        """Convert a legacy dataset from WXYZ to XYZW quaternion format.

        This method reads a dataset file, converts all quaternions from the legacy WXYZ format
        to the current XYZW format, and writes the result to a new file.

        Args:
            input_path: Path to the input dataset file (legacy WXYZ format).
            output_path: Path for the output dataset file. If None, appends '_xyzw' to input filename.

        Returns:
            Path to the converted dataset file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the dataset is already in XYZW format.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input dataset file not found: {input_path}")

        # Generate output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_xyzw{ext}"

        def convert_group_quaternions(src_group, dst_group):
            """Recursively copy and convert quaternions in groups."""
            # Copy attributes
            for attr_name, attr_value in src_group.attrs.items():
                dst_group.attrs[attr_name] = attr_value

            # Process items
            for key in src_group:
                if isinstance(src_group[key], h5py.Group):
                    # Recursively handle groups
                    dst_subgroup = dst_group.create_group(key)
                    convert_group_quaternions(src_group[key], dst_subgroup)
                else:
                    # Handle datasets
                    data = np.array(src_group[key])

                    # Convert root_pose quaternions
                    if key == "root_pose" and data.shape[-1] == 7:
                        data = convert_pose_quat_wxyz_to_xyzw(data)

                    # Preserve compression settings if possible
                    compression = src_group[key].compression
                    dst_group.create_dataset(key, data=data, compression=compression)

        with h5py.File(input_path, "r") as src_file:
            # Check if already converted
            if "format_version" in src_file.attrs and src_file.attrs["format_version"] >= DATASET_FORMAT_VERSION:
                raise ValueError(f"Dataset is already in XYZW format (version {src_file.attrs['format_version']})")

            with h5py.File(output_path, "w") as dst_file:
                # Set the new format version
                dst_file.attrs["format_version"] = DATASET_FORMAT_VERSION

                # Copy and convert all data
                convert_group_quaternions(src_file, dst_file)

        return output_path
