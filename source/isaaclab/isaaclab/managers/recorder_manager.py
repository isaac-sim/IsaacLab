# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Recorder manager for recording data produced from the given world."""

from __future__ import annotations

import enum
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from prettytable import PrettyTable

from ..utils import configclass, instantiate
from ..utils.datasets import EpisodeData, HDF5DatasetFileHandler
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import RecorderTermCfg

if TYPE_CHECKING:
    from ..envs import ManagerBasedEnv


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

    export_in_close: bool = False
    """Whether to export episodes in the close call."""

    buffer_initial_capacity: int = 64
    """Initial samples per world and field. Must be positive.

    Fields allocate on first use and double their capacity when necessary, retaining all samples.
    Resets reuse storage. Set this to the expected episode length (including decimation callbacks)
    to avoid growth during recording. Each field has independent write indices.
    """

    dataset_compression: bool = True
    """Enable dataset compression."""


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

    requires_post_step_observations: bool = True
    """Whether the RL environment must compute observations before this term's post-step callback.

    Terms that only read physical tensors should set this to False. The default preserves the
    observation access of existing terms; disabling it avoids extra noise draws and computation.
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

    def record_post_physics_decimation_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record data after the physics step is executed in the decimation loop.

        Returns:
            A tuple of key and value to be recorded.
            Please refer to the `record_pre_reset` function for more details.
        """
        return None, None

    def close(self, file_path: str):
        """Finalize and "clean up" the recorder term.

        This can include tasks such as appending metadata (e.g. labels) to a file
        and properly closing any associated file handles or resources.

        Args:
            file_path: the absolute path to the file
        """
        pass


class RecorderManager(ManagerBase):
    """Manager for recording data from recorder terms."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the recorder manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, RecorderTermCfg]``).
            env: The environment instance.
        """
        self._term_names: list[str] = []
        self._terms: dict[str, RecorderTerm] = {}
        self._dataset_file_handler = None
        self._failed_episode_dataset_file_handler = None

        # Do nothing if cfg is None or an empty dict
        if not cfg:
            return

        super().__init__(cfg, env)

        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if not isinstance(cfg, RecorderManagerBaseCfg):
            raise TypeError("Configuration for the recorder manager is not of type RecorderManagerBaseCfg.")

        if cfg.buffer_initial_capacity < 1:
            raise ValueError("buffer_initial_capacity must be positive.")
        self._fields: list[_RecorderField] = []
        self._env_ids = torch.arange(env.num_envs, device=env.device)
        # -1 denotes an episode whose success has not been set.
        self._success = torch.full((env.num_envs,), -1, dtype=torch.int8, device=env.device)
        self._exported_counts = torch.zeros(2, dtype=torch.long, device=env.device)

        env_name = getattr(env.cfg, "env_name", None)

        if cfg.dataset_export_mode != DatasetExportMode.EXPORT_NONE:
            self._dataset_file_handler = cfg.dataset_file_handler_class_type()
            self._dataset_file_handler.create(
                os.path.join(cfg.dataset_export_dir_path, cfg.dataset_filename), env_name=env_name
            )

        if cfg.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
            self._failed_episode_dataset_file_handler = cfg.dataset_file_handler_class_type()
            self._failed_episode_dataset_file_handler.create(
                os.path.join(cfg.dataset_export_dir_path, f"{cfg.dataset_filename}_failed"), env_name=env_name
            )

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
        self.close()

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active recorder terms."""
        return self._term_names

    @property
    def requires_post_step_observations(self) -> bool:
        """Whether any active recorder term requires post-step observations."""
        return any(term.requires_post_step_observations for term in self._terms.values())

    @property
    def exported_successful_episode_count(self) -> int:
        """Number of successful episodes.

        Returns:
            The number of successful episodes.
        """
        if not hasattr(self, "_exported_counts"):
            return 0
        return int(self._exported_counts[1].item())

    @property
    def exported_failed_episode_count(self) -> int:
        """Number of failed episodes.

        Returns:
            The number of failed episodes.
        """
        if not hasattr(self, "_exported_counts"):
            return 0
        return int(self._exported_counts[0].item())

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
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
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        self._reset_buffers(env_ids)
        return {}

    def get_term(self, name: str) -> RecorderTerm:
        """Return the configured recorder term with the given name."""
        return self._terms[name]

    def get_episode(self, env_id: int) -> EpisodeData:
        """Return an owned snapshot of a world's current episode.

        Leaves are tensors with a leading sample dimension, matching the exported dataset.
        Reading synchronizes write indices and copies the recorded samples. Later recording,
        reset, or snapshot mutation cannot affect the other copy. Add live data through
        :meth:`add_to_episodes`, and set success through :meth:`set_success_to_episodes`.
        """
        episode = EpisodeData()
        if not self.active_terms or not 0 <= env_id < self._env.num_envs:
            return episode
        for field in self._fields:
            length = int(field.lengths[env_id].item())
            if length:
                data = episode.data
                keys = field.name.split("/")
                for key in keys[:-1]:
                    data = data.setdefault(key, {})
                data[keys[-1]] = field.data[:length, env_id].clone()
        success = int(self._success[env_id].item())
        episode.success = None if success < 0 else bool(success)
        return episode

    def add_to_episodes(self, key: str, value: torch.Tensor | dict, env_ids: Sequence[int] | slice | None = None):
        """Append a batch to preallocated storage, advancing selected worlds' write indices.

        Args:
            key: Field path, with nested keys separated by '/'.
            value: Tensor of shape (selected worlds, ...) or nested dictionary of tensors.
                Warp arrays are also accepted. Storage copies the values and preserves dtype/device.
                A field's shape, dtype, and device must remain constant while it contains samples.
            env_ids: Unique world indices in the same order as the value batch. None selects all worlds.
        """
        if not self.active_terms or key is None:
            return
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.add_to_episodes(f"{key}/{sub_key}", sub_value, env_ids)
            return
        if isinstance(value, wp.array):
            value = wp.to_torch(value)
        num_selected = (
            self._env.num_envs
            if env_ids is None
            else (len(range(self._env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids))
        )
        if value.ndim < 1 or value.shape[0] != num_selected:
            raise ValueError(f"Field '{key}' requires {num_selected} rows, got {tuple(value.shape)}.")
        if not num_selected:
            return
        field = next((field for field in self._fields if field.name == key), None)
        if field is None:
            # Reject ambiguous leaf/subtree paths before allocating any storage.
            if any(key.startswith(f"{f.name}/") or f.name.startswith(f"{key}/") for f in self._fields):
                raise ValueError(f"Recorder field '{key}' conflicts with an existing field path.")
            field = _RecorderField(key, value, self._env.num_envs, self.cfg.buffer_initial_capacity)
            self._fields.append(field)
        field.append(value, env_ids)

    def set_success_to_episodes(self, env_ids: Sequence[int] | slice | None, success_values: torch.Tensor):
        """Set selected worlds' success flags from a tensor of shape (selected worlds,) or (selected worlds, 1)."""
        if not self.active_terms:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._success[env_ids] = success_values.reshape(-1).to(device=self._success.device, dtype=torch.int8)

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

    def record_post_physics_decimation_step(self) -> None:
        """Trigger recorder terms for post-physics step functions in the decimation loop."""
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        for term in self._terms.values():
            key, value = term.record_post_physics_decimation_step()
            self.add_to_episodes(key, value)

    def record_pre_reset(self, env_ids: Sequence[int] | slice | None, force_export_or_skip=None) -> None:
        """Trigger recorder terms for pre-reset functions.

        Args:
            env_ids: The environment ids in which a reset is triggered.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = self._env_ids
        elif isinstance(env_ids, slice):
            env_ids = self._env_ids[env_ids]

        for term in self._terms.values():
            key, value = term.record_pre_reset(env_ids)
            self.add_to_episodes(key, value, env_ids)

        # Set task success values for the relevant episodes
        success_results = torch.zeros(len(env_ids), dtype=bool, device=self._env.device)
        # Check success indicator from termination terms
        if hasattr(self._env, "termination_manager"):
            if "success" in self._env.termination_manager.active_terms:
                success_results |= self._env.termination_manager.get_term("success")[env_ids]
        self.set_success_to_episodes(env_ids, success_results)

        if force_export_or_skip or (force_export_or_skip is None and self.cfg.export_in_record_pre_reset):
            self.export_episodes(env_ids)

    def record_post_reset(self, env_ids: Sequence[int] | slice | None) -> None:
        """Trigger recorder terms for post-reset functions.

        Args:
            env_ids: The environment ids in which a reset is triggered.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if isinstance(env_ids, slice):
            env_ids = self._env_ids[env_ids]

        for term in self._terms.values():
            key, value = term.record_post_reset(env_ids)
            self.add_to_episodes(key, value, env_ids)

    def get_ep_meta(self) -> dict:
        """Get the episode metadata."""
        if not hasattr(self._env.cfg, "get_ep_meta"):
            # Add basic episode metadata
            ep_meta = dict()
            ep_meta["sim_args"] = {
                "dt": self._env.cfg.sim.dt,
                "decimation": self._env.cfg.decimation,
                "render_interval": self._env.cfg.sim.render_interval,
                "num_envs": self._env.cfg.scene.num_envs,
            }
            return ep_meta

        # Add custom episode metadata if available
        ep_meta = self._env.cfg.get_ep_meta()
        return ep_meta

    def export_episodes(
        self, env_ids: Sequence[int] | slice | None = None, demo_ids: Sequence[int] | None = None
    ) -> None:
        """Concludes and exports the episodes for the given environment ids.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
            demo_ids: Custom identifiers for the exported episodes.
                If provided, episodes will be named "demo_{demo_id}" in the dataset.
                Should have the same length as env_ids if both are provided.
                If None, uses the default sequential naming scheme. Defaults to None.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return

        if env_ids is None:
            env_ids = self._env_ids
        elif isinstance(env_ids, slice):
            env_ids = self._env_ids[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._env_ids.device)

        # Handle demo_ids processing
        if demo_ids is not None:
            if isinstance(demo_ids, torch.Tensor):
                demo_ids = demo_ids.tolist()
            if len(demo_ids) != len(env_ids):
                raise ValueError(f"Length of demo_ids ({len(demo_ids)}) must match length of env_ids ({len(env_ids)})")
            # Check for duplicate demo_ids
            if len(set(demo_ids)) != len(demo_ids):
                duplicates = [x for i, x in enumerate(demo_ids) if demo_ids.index(x) != i]
                raise ValueError(f"demo_ids must be unique. Found duplicates: {list(set(duplicates))}")

        present = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env_ids.device)
        for field in self._fields:
            present |= (field.lengths > 0).to(present.device)
        succeeded = self._success == 1

        # Only file export materializes per-world snapshots. EXPORT_NONE stays batched.
        need_to_flush = False
        mode = self.cfg.dataset_export_mode
        if mode != DatasetExportMode.EXPORT_NONE and present[env_ids].any().item():
            ep_meta = self.get_ep_meta()
            for handler in (self._dataset_file_handler, self._failed_episode_dataset_file_handler):
                if handler is not None:
                    handler.add_env_args(ep_meta)
            selected = env_ids.tolist()
            selected_present = present[env_ids].tolist()
            selected_success = succeeded[env_ids].tolist()
            for i, env_id in enumerate(selected):
                if not selected_present[i]:
                    continue
                handler = self._dataset_file_handler
                if mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
                    if not selected_success[i]:
                        handler = self._failed_episode_dataset_file_handler
                elif mode == DatasetExportMode.EXPORT_SUCCEEDED_ONLY and not selected_success[i]:
                    continue
                episode = self.get_episode(env_id)
                demo_id = demo_ids[i] if demo_ids is not None else None
                handler.write_episode(episode, demo_id, self.cfg.dataset_compression)
                need_to_flush = True

        self._exported_counts[0] += (present & ~succeeded)[env_ids].sum()
        self._exported_counts[1] += (present & succeeded)[env_ids].sum()
        self._reset_buffers(env_ids)
        if need_to_flush:
            for handler in (self._dataset_file_handler, self._failed_episode_dataset_file_handler):
                if handler is not None:
                    handler.flush()

    def close(self):
        """Closes the recorder manager by exporting any remaining data to file as well as properly
        closes the recorder terms.
        """
        # Do nothing if no active recorder terms are provided
        if len(self.active_terms) == 0:
            return
        if self._dataset_file_handler is not None:
            if self.cfg.export_in_close:
                self.export_episodes()
            self._dataset_file_handler.close()
        if self._failed_episode_dataset_file_handler is not None:
            self._failed_episode_dataset_file_handler.close()
        for term in self._terms.values():
            term.close(os.path.join(self.cfg.dataset_export_dir_path, self.cfg.dataset_filename))

    """
    Helper functions.
    """

    def _reset_buffers(self, env_ids):
        """Discard selected samples by resetting indices; retain allocations for the next episodes."""
        for field in self._fields:
            field.reset(env_ids)
        self._success[slice(None) if env_ids is None else env_ids] = -1

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
                "export_in_close",
                "dataset_compression",
                "buffer_initial_capacity",
            ]:
                continue
            # check if term config is None
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, RecorderTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type RecorderTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the recorder term
            term = instantiate(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, RecorderTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type RecorderTerm.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term


class _RecorderField:
    """One field's time-major allocation and independent per-world insertion indices.

    No per-world Python containers or tensor views are created during append. The host upper
    bound avoids reading device indices in steady state; only capacity exhaustion checks synchronize.
    """

    def __init__(self, name: str, value: torch.Tensor, num_envs: int, capacity: int):
        self.name = name
        # Retained storage outlives the caller's inference context: reset/export may run outside it.
        with torch.inference_mode(False):
            self.data = torch.empty((capacity, num_envs, *value.shape[1:]), dtype=value.dtype, device=value.device)
            self.lengths = torch.zeros(num_envs, dtype=torch.long, device=value.device)
            self.worlds = torch.arange(num_envs, device=value.device)
        self.upper_bound = 0

    @torch.no_grad()
    def append(self, value: torch.Tensor, env_ids):
        if value.shape[1:] != self.data.shape[2:] or value.dtype != self.data.dtype or value.device != self.data.device:
            if self.lengths.any().item():
                raise ValueError(f"Recorder field '{self.name}' changed shape, dtype, or device with active samples.")
            with torch.inference_mode(False):
                self.data = torch.empty(
                    (*self.data.shape[:2], *value.shape[1:]), dtype=value.dtype, device=value.device
                )
                self.lengths = self.lengths.to(value.device)
                self.worlds = self.worlds.to(value.device)
            self.upper_bound = 0
        if env_ids is None:
            worlds, indices = self.worlds, self.lengths
        else:
            if not isinstance(env_ids, slice):
                env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.data.device)
            worlds = self.worlds[env_ids]
            indices = self.lengths[worlds]
        capacity = self.data.shape[0]
        if self.upper_bound >= capacity:
            self.upper_bound = int(self.lengths.max().item())
            if self.upper_bound >= capacity:
                with torch.inference_mode(False):
                    data = self.data.new_empty((2 * capacity, *self.data.shape[1:]))
                data[:capacity].copy_(self.data)
                self.data = data
        self.data[indices, worlds] = value
        if env_ids is None:
            self.lengths.add_(1)
        else:
            self.lengths[worlds] = indices + 1
        self.upper_bound += 1

    def reset(self, env_ids):
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            self.lengths.zero_()
            self.upper_bound = 0
        else:
            if not isinstance(env_ids, slice):
                env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.data.device)
            self.lengths[env_ids] = 0
            if not isinstance(env_ids, slice) and len(env_ids) == len(self.lengths):
                self.upper_bound = 0
