# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit batched episode collection and dataset export."""

from __future__ import annotations

import enum
import os
from collections.abc import Sequence

import torch
import warp as wp

from ..configclass import configclass
from .episode_data import EpisodeData
from .hdf5_dataset_file_handler import HDF5DatasetFileHandler


class DatasetExportMode(enum.IntEnum):
    """The mode to handle episode exports."""

    EXPORT_NONE = 0  # Export none of the episodes
    EXPORT_ALL = 1  # Export all episodes to a single dataset file
    EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES = 2  # Export succeeded and failed episodes in separate files
    EXPORT_SUCCEEDED_ONLY = 3  # Export only succeeded episodes to a single dataset file


@configclass
class EpisodeRecorderCfg:
    """Output and buffer settings for explicitly collected episodes."""

    dataset_file_handler_class_type: type = HDF5DatasetFileHandler

    dataset_export_dir_path: str = "/tmp/isaaclab/logs"
    """The directory path where the recorded datasets are exported."""

    dataset_filename: str = "dataset"
    """Dataset file name without file extension."""

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL
    """The mode to handle episode exports."""

    export_in_close: bool = False
    """Whether to export episodes in the close call."""

    buffer_initial_capacity: int = 64
    """Initial sample capacity per environment and field; grows as needed. Must be positive."""

    dataset_compression: bool = True
    """Enable dataset compression."""


class EpisodeRecorder:
    """Batched episode storage and dataset export, independent of environments and capture functions."""

    def __init__(
        self,
        cfg: EpisodeRecorderCfg,
        num_envs: int,
        device: str,
        env_name: str | None = None,
        env_metadata: dict | None = None,
    ) -> None:
        """Create episode storage and optional dataset files.

        Args:
            cfg: Buffer and output settings.
            num_envs: Number of independent episodes collected together.
            device: Device for episode indices and success flags.
            env_name: Environment identifier stored in the dataset.
            env_metadata: Explicit simulation or task metadata for dataset exports.
        """
        if cfg.buffer_initial_capacity < 1:
            raise ValueError("buffer_initial_capacity must be positive.")
        self.cfg = cfg
        self.num_envs = num_envs
        self.env_metadata = env_metadata or {}
        self._fields: list[_RecorderField] = []
        self._env_ids = torch.arange(num_envs, device=device)
        self._success = torch.full((num_envs,), -1, dtype=torch.int8, device=device)
        self._exported_counts = torch.zeros(2, dtype=torch.long, device=device)
        self._dataset_file_handler = None
        self._failed_episode_dataset_file_handler = None
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

    @property
    def exported_successful_episode_count(self) -> int:
        """Number of successful episodes."""
        return int(self._exported_counts[1].item())

    @property
    def exported_failed_episode_count(self) -> int:
        """Number of failed episodes."""
        return int(self._exported_counts[0].item())

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Resets the recorder data.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        if isinstance(env_ids, Sequence):
            env_ids = list(env_ids)
        self._reset_buffers(env_ids)

    def get_episode(self, env_id: int) -> EpisodeData:
        """Return an owned snapshot of the selected environment's current episode.

        Reading copies data and may synchronize the device. For live writes, use
        :meth:`add_to_episodes` and :meth:`set_success_to_episodes`.

        Args:
            env_id: The environment index.

        Returns:
            An episode with tensor leaves and a leading sample dimension.
        """
        episode = EpisodeData()
        if not 0 <= env_id < self.num_envs:
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

    def add_to_episodes(
        self, key: str, value: torch.Tensor | dict, env_ids: Sequence[int] | slice | None = None
    ) -> None:
        """Append a batch to the selected episodes.

        Args:
            key: Field path, with nested keys separated by '/'.
            value: Tensor of shape (selected environments, ...) or nested dictionary of tensors.
                Warp arrays are also accepted. Storage copies the values and preserves dtype/device.
                A field's shape, dtype, and device must remain constant while it contains samples.
            env_ids: Unique environment indices in batch order. None selects all environments.
        """
        if key is None:
            return
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.add_to_episodes(f"{key}/{sub_key}", sub_value, env_ids)
            return
        if isinstance(value, wp.array):
            value = wp.to_torch(value)
        num_selected = (
            self.num_envs
            if env_ids is None
            else (len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids))
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
            field = _RecorderField(key, value, self.num_envs, self.cfg.buffer_initial_capacity)
            self._fields.append(field)
        field.append(value, env_ids)

    def set_success_to_episodes(self, env_ids: Sequence[int] | slice | None, success_values: torch.Tensor) -> None:
        """Set success flags for selected environments.

        Args:
            env_ids: The environment indices. None selects all environments.
            success_values: Flags in selection order, with shape (num_selected,) or (num_selected, 1).
        """
        if env_ids is None:
            env_ids = slice(None)
        elif isinstance(env_ids, Sequence):
            env_ids = list(env_ids)
        self._success[env_ids] = success_values.reshape(-1).to(device=self._success.device, dtype=torch.int8)

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

        present = torch.zeros(self.num_envs, dtype=torch.bool, device=self._env_ids.device)
        for field in self._fields:
            present |= (field.lengths > 0).to(present.device)
        succeeded = self._success == 1

        # Only file export materializes per-world snapshots. EXPORT_NONE stays batched.
        need_to_flush = False
        mode = self.cfg.dataset_export_mode
        if mode != DatasetExportMode.EXPORT_NONE and present[env_ids].any().item():
            ep_meta = self.env_metadata
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
        """Export remaining episodes if configured and close dataset files."""
        if self._dataset_file_handler is not None:
            if self.cfg.export_in_close:
                self.export_episodes()
            self._dataset_file_handler.close()
        if self._failed_episode_dataset_file_handler is not None:
            self._failed_episode_dataset_file_handler.close()

    def _reset_buffers(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> None:
        """Discard selected samples by resetting indices; retain allocations for the next episodes."""
        for field in self._fields:
            field.reset(env_ids)
        self._success[slice(None) if env_ids is None else env_ids] = -1


class _RecorderField:
    """One field's time-major allocation and independent per-environment insertion indices.

    No per-environment Python containers or tensor views are created during append. The host upper
    bound avoids reading device indices in steady state; only capacity exhaustion checks synchronize.
    """

    def __init__(self, name: str, value: torch.Tensor, num_envs: int, capacity: int) -> None:
        self.name = name
        # Retained storage outlives the caller's inference context: reset/export may run outside it.
        with torch.inference_mode(False):
            self.data = torch.empty((capacity, num_envs, *value.shape[1:]), dtype=value.dtype, device=value.device)
            self.lengths = torch.zeros(num_envs, dtype=torch.long, device=value.device)
            self.worlds = torch.arange(num_envs, device=value.device)
        self.upper_bound = 0

    @torch.no_grad()
    def append(self, value: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | slice | None) -> None:
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

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> None:
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            self.lengths.zero_()
            self.upper_bound = 0
        else:
            if not isinstance(env_ids, slice):
                env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.data.device)
            self.lengths[env_ids] = 0
            if not isinstance(env_ids, slice) and len(env_ids) == len(self.lengths):
                self.upper_bound = 0
