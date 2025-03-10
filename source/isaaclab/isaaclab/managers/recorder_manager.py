# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder manager for recording data produced from the given world."""

from __future__ import annotations

import enum
import os
import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RecorderTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DatasetExportMode(enum.IntEnum):
    """The mode to handle episode exports."""

    EXPORT_NONE = 0  # Export none of the episodes
    EXPORT_ALL = 1  # Export all episodes to a single dataset file
    EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES = 2  # Export succeeded and failed episodes in separate files
    EXPORT_SUCCEEDED_ONLY = 3  # Export only succeeded episodes to a single dataset file


@configclass
class RecorderManagerBaseCfg:
    """Base class for configuring recorder manager terms."""

    dataset_file_handler_class_type: type = HDF5DatasetFileHandler

    dataset_export_dir_path: str = "/tmp/isaaclab/logs"
    """The directory path where the recorded datasets are exported."""

    dataset_filename: str = "dataset"
    """Dataset file name without file extension."""

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL
    """The mode to handle episode exports."""

    export_in_record_pre_reset: bool = True
    """Whether to export episodes in the record_pre_reset call."""


class RecorderTerm(ManagerTermBase):
    """Base class for recorder terms.

    The recorder term is responsible for recording data at various stages of the environment's lifecycle.
    A recorder term is comprised of four user-defined callbacks to record data in the corresponding stages:

    * Pre-reset recording: This callback is invoked at the beginning of `env.reset()` before the reset is effective.
    * Post-reset recording: This callback is invoked at the end of `env.reset()`.
    * Pre-step recording: This callback is invoked at the beginning of `env.step()`, after the step action is processed
          and before the action is applied by the action manager.
    * Post-step recording: This callback is invoked at the end of `env.step()` when all the managers are processed.
    """

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        """Initialize the recorder term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # call the base class constructor
        super().__init__(cfg, env)

    """
    User-defined callbacks.
    """

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record data at the beginning of env.reset() before reset is effective.

        Args:
            env_ids: The environment ids. All environments should be considered when set to None.

        Returns:
            A tuple of key and value to be recorded.
            The key can contain nested keys separated by '/'. For example, "obs/joint_pos" would add the given
            value under ['obs']['policy'] in the underlying dictionary in the recorded episode data.
            The value can be a tensor or a nested dictionary of tensors. The shape of a tensor in the value
            is (env_ids, ...).
        """
        return None, None

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record data at the end of env.reset().

        Args:
            env_ids: The environment ids. All environments should be considered when set to None.

        Returns:
            A tuple of key and value to be recorded.
            Please refer to the `record_pre_reset` function for more details.
        """
        return None, None

    def record_pre_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record data in the beginning of env.step() after action is cached/processed in the ActionManager.

        Returns:
            A tuple of key and value to be recorded.
            Please refer to the `record_pre_reset` function for more details.
        """
        return None, None

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record data at the end of env.step() when all the managers are processed.

        Returns:
            A tuple of key and value to be recorded.
            Please refer to the `record_pre_reset` function for more details.
        """
        return None, None


class RecorderManager(ManagerBase):
    """Manager for recording data from recorder terms."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the recorder manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RecorderTermCfg]``).
            env: The environment instance.
        """
        self._term_names: list[str] = list()
        self._terms: dict[str, RecorderTerm] = dict()

        # Do nothing if cfg is None or an empty dict
        if not cfg:
            return

        super().__init__(cfg, env)

        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if not isinstance(cfg, RecorderManagerBaseCfg):
            raise TypeError("Configuration for the recorder manager is not of type RecorderManagerBaseCfg.")

        # create episode data buffer indexed by environment id
        self._episodes: dict[int, EpisodeData] = dict()
        for env_id in range(env.num_envs):
            self._episodes[env_id] = EpisodeData()

        env_name = getattr(env.cfg, "env_name", None)

        self._dataset_file_handler = None
        if cfg.dataset_export_mode != DatasetExportMode.EXPORT_NONE:
            self._dataset_file_handler = cfg.dataset_file_handler_class_type()
            self._dataset_file_handler.create(
                os.path.join(cfg.dataset_export_dir_path, cfg.dataset_filename), env_name=env_name
            )

        self._failed_episode_dataset_file_handler = None
        if cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
            self._failed_episode_dataset_file_handler = cfg.dataset_file_handler_class_type()
            self._failed_episode_dataset_file_handler.create(
                os.path.join(cfg.dataset_export_dir_path, f"{cfg.dataset_filename}_failed"), env_name=env_name
            )

        self._exported_successful_episode_count = {}
        self._exported_failed_episode_count = {}

    def __str__(self) -> str:
        """Returns: A string representation for recorder manager."""
        msg = f"<RecorderManager> contains {len(self._term_names)} active terms.\n"
        # create table for term information
        table = PrettyTable()
        table.title = "Active Recorder Terms"
        table.field_names = ["Index", "Name"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, name in enumerate(self._term_names):
            table.add_row([index, name])
        # convert table to string
        msg += table.get_string()
        msg += "\n"
        return msg

    def __del__(self):
        """Destructor for recorder."""
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if self._dataset_file_handler is not None:
            self._dataset_file_handler.close()

        if self._failed_episode_dataset_file_handler is not None:
            self._failed_episode_dataset_file_handler.close()

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active recorder terms."""
        return self._term_names

    @property
    def exported_successful_episode_count(self, env_id=None) -> int:
        """Number of successful episodes.

        Args:
            env_id: The environment id. Defaults to None, in which case all environments are considered.

        Returns:
            The number of successful episodes.
        """
        if env_id is not None:
            return self._exported_successful_episode_count.get(env_id, 0)
        return sum(self._exported_successful_episode_count.values())

    @property
    def exported_failed_episode_count(self, env_id=None) -> int:
        """Number of failed episodes.

        Args:
            env_id: The environment id. Defaults to None, in which case all environments are considered.

        Returns:
            The number of failed episodes.
        """
        if env_id is not None:
            return self._exported_failed_episode_count.get(env_id, 0)
        return sum(self._exported_failed_episode_count.values())

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the recorder data.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return {}

        # resolve environment ids
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        for term in self._terms.values():
            term.reset(env_ids=env_ids)

        for env_id in env_ids:
            self._episodes[env_id] = EpisodeData()

        # nothing to log here
        return {}

    def get_episode(self, env_id: int) -> EpisodeData:
        """Returns the episode data for the given environment id.

        Args:
            env_id: The environment id.

        Returns:
            The episode data for the given environment id.
        """
        return self._episodes.get(env_id, EpisodeData())

    def add_to_episodes(self, key: str, value: torch.Tensor | dict, env_ids: Sequence[int] | None = None):
        """Adds the given key-value pair to the episodes for the given environment ids.

        Args:
            key: The key of the given value to be added to the episodes. The key can contain nested keys
                separated by '/'. For example, "obs/joint_pos" would add the given value under ['obs']['policy']
                in the underlying dictionary in the episode data.
            value: The value to be added to the episodes. The value can be a tensor or a nested dictionary of tensors.
                The shape of a tensor in the value is (env_ids, ...).
            env_ids: The environment ids. Defaults to None, in which case all environments are considered.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        # resolve environment ids
        if key is None:
            return
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.add_to_episodes(f"{key}/{sub_key}", sub_value, env_ids)
            return

        for value_index, env_id in enumerate(env_ids):
            if env_id not in self._episodes:
                self._episodes[env_id] = EpisodeData()
                self._episodes[env_id].env_id = env_id
            self._episodes[env_id].add(key, value[value_index])

    def set_success_to_episodes(self, env_ids: Sequence[int] | None, success_values: torch.Tensor):
        """Sets the task success values to the episodes for the given environment ids.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all environments are considered.
            success_values: The task success values to be set to the episodes. The shape of the tensor is (env_ids, 1).
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        # resolve environment ids
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        for value_index, env_id in enumerate(env_ids):
            self._episodes[env_id].success = success_values[value_index].item()

    def record_pre_step(self) -> None:
        """Trigger recorder terms for pre-step functions."""
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        for term in self._terms.values():
            key, value = term.record_pre_step()
            self.add_to_episodes(key, value)

    def record_post_step(self) -> None:
        """Trigger recorder terms for post-step functions."""
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        for term in self._terms.values():
            key, value = term.record_post_step()
            self.add_to_episodes(key, value)

    def record_pre_reset(self, env_ids: Sequence[int] | None, force_export_or_skip=None) -> None:
        """Trigger recorder terms for pre-reset functions.

        Args:
            env_ids: The environment ids in which a reset is triggered.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        for term in self._terms.values():
            key, value = term.record_pre_reset(env_ids)
            self.add_to_episodes(key, value, env_ids)

        # Set task success values for the relevant episodes
        success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
        # Check success indicator from termination terms
        if "success" in self._env.termination_manager.active_terms:
            success_results |= self._env.termination_manager.get_term("success")[env_ids]
        self.set_success_to_episodes(env_ids, success_results)

        if force_export_or_skip or (force_export_or_skip is None and self.cfg.export_in_record_pre_reset):
            self.export_episodes(env_ids)

    def record_post_reset(self, env_ids: Sequence[int] | None) -> None:
        """Trigger recorder terms for post-reset functions.

        Args:
            env_ids: The environment ids in which a reset is triggered.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        for term in self._terms.values():
            key, value = term.record_post_reset(env_ids)
            self.add_to_episodes(key, value, env_ids)

    def export_episodes(self, env_ids: Sequence[int] | None = None) -> None:
        """Concludes and exports the episodes for the given environment ids.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()

        # Export episode data through dataset exporter
        need_to_flush = False
        for env_id in env_ids:
            if env_id in self._episodes and not self._episodes[env_id].is_empty():
                episode_succeeded = self._episodes[env_id].success
                target_dataset_file_handler = None
                if (self.cfg.dataset_export_mode == DatasetExportMode.EXPORT_ALL) or (
                    self.cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_ONLY and episode_succeeded
                ):
                    target_dataset_file_handler = self._dataset_file_handler
                elif self.cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
                    if episode_succeeded:
                        target_dataset_file_handler = self._dataset_file_handler
                    else:
                        target_dataset_file_handler = self._failed_episode_dataset_file_handler
                if target_dataset_file_handler is not None:
                    target_dataset_file_handler.write_episode(self._episodes[env_id])
                    need_to_flush = True
                # Update episode count
                if episode_succeeded:
                    self._exported_successful_episode_count[env_id] = (
                        self._exported_successful_episode_count.get(env_id, 0) + 1
                    )
                else:
                    self._exported_failed_episode_count[env_id] = self._exported_failed_episode_count.get(env_id, 0) + 1
            # Reset the episode buffer for the given environment after export
            self._episodes[env_id] = EpisodeData()

        if need_to_flush:
            if self._dataset_file_handler is not None:
                self._dataset_file_handler.flush()
            if self._failed_episode_dataset_file_handler is not None:
                self._failed_episode_dataset_file_handler.flush()

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of recorder terms."""
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            # skip non-term settings
            if term_name in [
                "dataset_file_handler_class_type",
                "dataset_filename",
                "dataset_export_dir_path",
                "dataset_export_mode",
                "export_in_record_pre_reset",
            ]:
                continue
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, RecorderTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type RecorderTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the recorder term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, RecorderTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type RecorderTerm.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term
