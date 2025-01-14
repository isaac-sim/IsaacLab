# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

from isaaclab_mimic.datagen.datagen_info import DatagenInfo

import isaaclab.utils.math as PoseUtils
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler


class DataGenInfoPool:
    """
    Pool of DatagenInfo for data generation.

    This class is a container for storing `DatagenInfo` objects that are extracted from episodes.
    The pool supports the use of an asyncio lock to safely add new episodes to the pool while
    consuming the data, so it can be shared across multiple mimic data generators.
    """

    def __init__(self, env, env_cfg, device, asyncio_lock: asyncio.Lock | None = None):
        """
        Args:
            env_cfg (dict): environment configuration
            device (torch.device): device to store the data
            asyncio_lock (asyncio.Lock or None): asyncio lock to use for thread safety
        """
        self._datagen_infos = []
        self._subtask_indices = []

        self.env = env
        self.env_cfg = env_cfg
        self.device = device

        self._asyncio_lock = asyncio_lock

        if len(env_cfg.subtask_configs) != 1:
            raise ValueError("Data generation currently supports only one end-effector.")

        (subtask_configs,) = env_cfg.subtask_configs.values()
        self.subtask_term_signals = [subtask_config.subtask_term_signal for subtask_config in subtask_configs]
        self.subtask_term_offset_ranges = [
            subtask_config.subtask_term_offset_range for subtask_config in subtask_configs
        ]

    @property
    def datagen_infos(self):
        """Returns the datagen infos."""
        return self._datagen_infos

    @property
    def subtask_indices(self):
        """Returns the subtask indices."""
        return self._subtask_indices

    @property
    def asyncio_lock(self):
        """Returns the asyncio lock."""
        return self._asyncio_lock

    @property
    def num_datagen_infos(self):
        """Returns the number of datagen infos."""
        return len(self._datagen_infos)

    async def add_episode(self, episode: EpisodeData):
        """
        Add a datagen info from the given episode.

        Args:
            episode (EpisodeData): episode to add
        """
        if self._asyncio_lock is not None:
            async with self._asyncio_lock:
                self._add_episode(episode)
        else:
            self._add_episode(episode)

    def _add_episode(self, episode: EpisodeData):
        """
        Add a datagen info from the given episode.

        Args:
            episode (EpisodeData): episode to add
        """
        ep_grp = episode.data
        eef_name = list(self.env.cfg.subtask_configs.keys())[0]

        # extract datagen info
        if "datagen_info" in ep_grp["obs"]:
            eef_pose = ep_grp["obs"]["datagen_info"]["eef_pose"][eef_name]
            object_poses_dict = ep_grp["obs"]["datagen_info"]["object_pose"]
            target_eef_pose = ep_grp["obs"]["datagen_info"]["target_eef_pose"][eef_name]
            subtask_term_signals_dict = ep_grp["obs"]["datagen_info"]["subtask_term_signals"]
        else:
            # Extract eef poses
            eef_pos = ep_grp["obs"]["eef_pos"]
            eef_quat = ep_grp["obs"]["eef_quat"]  # format (w, x, y, z)
            eef_rot_matrices = PoseUtils.matrix_from_quat(eef_quat)  # shape (N, 3, 3)
            # Create pose matrices for all environments
            eef_pose = PoseUtils.make_pose(eef_pos, eef_rot_matrices)  # shape (N, 4, 4)

            # Object poses
            object_poses_dict = dict()
            for object_name, value in ep_grp["obs"]["object_pose"].items():
                # object_pose
                value = value["root_pose"]
                # Root state ``[pos, quat, lin_vel, ang_vel]`` in simulation world frame. Shape is (num_steps, 13).
                # Quaternion ordering is wxyz

                # Convert to rotation matrices
                object_rot_matrices = PoseUtils.matrix_from_quat(value[:, 3:7])  # shape (N, 3, 3)
                object_rot_positions = value[:, 0:3]  # shape (N, 3)
                object_poses_dict[object_name] = PoseUtils.make_pose(object_rot_positions, object_rot_matrices)

            # Target eef pose
            target_eef_pose = ep_grp["obs"]["target_eef_pose"]

            # Subtask termination signalsS
            subtask_term_signals_dict = (ep_grp["obs"]["subtask_term_signals"],)

        # Extract gripper actions
        gripper_actions = self.env.actions_to_gripper_actions(ep_grp["actions"])[eef_name]

        ep_datagen_info_obj = DatagenInfo(
            eef_pose=eef_pose,
            object_poses=object_poses_dict,
            subtask_term_signals=subtask_term_signals_dict,
            target_eef_pose=target_eef_pose,
            gripper_action=gripper_actions,
        )
        self._datagen_infos.append(ep_datagen_info_obj)

        # parse subtask indices using subtask termination signals
        ep_subtask_indices = []
        prev_subtask_term_ind = 0
        for subtask_ind in range(len(self.subtask_term_signals)):
            subtask_term_signal = self.subtask_term_signals[subtask_ind]
            if subtask_term_signal is None:
                # final subtask, finishes at end of demo
                subtask_term_ind = ep_grp["actions"].shape[0]
            else:
                # trick to detect index where first 0 -> 1 transition occurs - this will be the end of the subtask
                subtask_indicators = ep_datagen_info_obj.subtask_term_signals[subtask_term_signal].flatten().int()
                diffs = subtask_indicators[1:] - subtask_indicators[:-1]
                end_ind = int(diffs.nonzero()[0][0]) + 1
                subtask_term_ind = end_ind + 1  # increment to support indexing like demo[start:end]
            ep_subtask_indices.append([prev_subtask_term_ind, subtask_term_ind])
            prev_subtask_term_ind = subtask_term_ind

        # run sanity check on subtask_term_offset_range in task spec to make sure we can never
        # get an empty subtask in the worst case when sampling subtask bounds:
        #
        #   end index of subtask i + max offset of subtask i < end index of subtask i + 1 + min offset of subtask i + 1
        #
        assert len(ep_subtask_indices) == len(
            self.subtask_term_signals
        ), "mismatch in length of extracted subtask info and number of subtasks"
        for i in range(1, len(ep_subtask_indices)):
            prev_max_offset_range = self.subtask_term_offset_ranges[i - 1][1]
            assert (
                ep_subtask_indices[i - 1][1] + prev_max_offset_range
                < ep_subtask_indices[i][1] + self.subtask_term_offset_ranges[i][0]
            ), (
                "subtask sanity check violation in demo with subtask {} end ind {}, subtask {} max offset {},"
                " subtask {} end ind {}, and subtask {} min offset {}".format(
                    i - 1,
                    ep_subtask_indices[i - 1][1],
                    i - 1,
                    prev_max_offset_range,
                    i,
                    ep_subtask_indices[i][1],
                    i,
                    self.subtask_term_offset_ranges[i][0],
                )
            )

        self._subtask_indices.append(ep_subtask_indices)

    def load_from_dataset_file(self, file_path, select_demo_keys: str | None = None):
        """
        Load from a dataset file.

        Args:
            file_path (str): path to the dataset file
            select_demo_keys (str or None): keys of the demos to load
        """
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(file_path)
        episode_names = dataset_file_handler.get_episode_names()

        if len(episode_names) == 0:
            return

        for episode_name in episode_names:
            if select_demo_keys is not None and episode_name not in select_demo_keys:
                continue
            episode = dataset_file_handler.load_episode(episode_name, self.device)
            self._add_episode(episode)
