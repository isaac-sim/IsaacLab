# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from isaaclab_mimic.datagen.datagen_info import DatagenInfo


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

        # Start and end step indices of each subtask in each episode for each eef
        self._subtask_boundaries: dict[str, list[list[tuple[int, int]]]] = {}

        self.env = env
        self.env_cfg = env_cfg
        self.device = device

        self._asyncio_lock = asyncio_lock

        # Subtask termination infos for the given environment
        self.subtask_term_signal_names: dict[str, list[str]] = {}
        self.subtask_term_offset_ranges: dict[str, list[tuple[int, int]]] = {}
        for eef_name, eef_subtask_configs in env_cfg.subtask_configs.items():
            self.subtask_term_signal_names[eef_name] = [
                subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs
            ]
            self.subtask_term_offset_ranges[eef_name] = [
                subtask_config.subtask_term_offset_range for subtask_config in eef_subtask_configs
            ]

    @property
    def datagen_infos(self):
        """Returns the datagen infos."""
        return self._datagen_infos

    @property
    def subtask_boundaries(self) -> dict[str, list[list[tuple[int, int]]]]:
        """Returns the subtask boundaries."""
        return self._subtask_boundaries

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

        # extract datagen info
        if "datagen_info" in ep_grp["obs"]:
            eef_pose = ep_grp["obs"]["datagen_info"]["eef_pose"]
            object_poses_dict = ep_grp["obs"]["datagen_info"]["object_pose"]
            target_eef_pose = ep_grp["obs"]["datagen_info"]["target_eef_pose"]
            subtask_term_signals_dict = ep_grp["obs"]["datagen_info"]["subtask_term_signals"]
        else:
            raise ValueError("Episode to be loaded to DatagenInfo pool lacks datagen_info annotations")

        # Extract gripper actions
        gripper_actions = self.env.actions_to_gripper_actions(ep_grp["actions"])

        ep_datagen_info_obj = DatagenInfo(
            eef_pose=eef_pose,
            object_poses=object_poses_dict,
            subtask_term_signals=subtask_term_signals_dict,
            target_eef_pose=target_eef_pose,
            gripper_action=gripper_actions,
        )
        self._datagen_infos.append(ep_datagen_info_obj)

        # parse subtask ranges using subtask termination signals and store
        # the start and end indices of each subtask for each eef
        for eef_name in self.subtask_term_signal_names.keys():
            if eef_name not in self._subtask_boundaries:
                self._subtask_boundaries[eef_name] = []
            prev_subtask_term_ind = 0
            eef_subtask_boundaries = []
            for subtask_term_signal_name in self.subtask_term_signal_names[eef_name]:
                if subtask_term_signal_name is None:
                    # None refers to the final subtask, so finishes at end of demo
                    subtask_term_ind = ep_grp["actions"].shape[0]
                else:
                    # trick to detect index where first 0 -> 1 transition occurs - this will be the end of the subtask
                    subtask_indicators = (
                        ep_datagen_info_obj.subtask_term_signals[subtask_term_signal_name].flatten().int()
                    )
                    diffs = subtask_indicators[1:] - subtask_indicators[:-1]
                    end_ind = int(diffs.nonzero()[0][0]) + 1
                    subtask_term_ind = end_ind + 1  # increment to support indexing like demo[start:end]

                if subtask_term_ind <= prev_subtask_term_ind:
                    raise ValueError(
                        f"subtask termination signal is not increasing: {subtask_term_ind} should be greater than"
                        f" {prev_subtask_term_ind}"
                    )
                eef_subtask_boundaries.append((prev_subtask_term_ind, subtask_term_ind))
                prev_subtask_term_ind = subtask_term_ind

            # run sanity check on subtask_term_offset_range in task spec to make sure we can never
            # get an empty subtask in the worst case when sampling subtask bounds:
            #
            #   end index of subtask i + max offset of subtask i < end index of subtask i + 1 + min offset of subtask i + 1
            #
            for i in range(1, len(eef_subtask_boundaries)):
                prev_max_offset_range = self.subtask_term_offset_ranges[eef_name][i - 1][1]
                assert (
                    eef_subtask_boundaries[i - 1][1] + prev_max_offset_range
                    < eef_subtask_boundaries[i][1] + self.subtask_term_offset_ranges[eef_name][i][0]
                ), (
                    "subtask sanity check violation in demo with subtask {} end ind {}, subtask {} max offset {},"
                    " subtask {} end ind {}, and subtask {} min offset {}".format(
                        i - 1,
                        eef_subtask_boundaries[i - 1][1],
                        i - 1,
                        prev_max_offset_range,
                        i,
                        eef_subtask_boundaries[i][1],
                        i,
                        self.subtask_term_offset_ranges[eef_name][i][0],
                    )
                )

            self._subtask_boundaries[eef_name].append(eef_subtask_boundaries)

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
