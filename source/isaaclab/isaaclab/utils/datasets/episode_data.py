# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


class EpisodeData:
    """Class to store episode data."""

    def __init__(self) -> None:
        """Initializes episode data class."""
        self._data = {}
        self._next_action_index = 0
        self._next_state_index = 0
        self._next_joint_target_index = 0
        self._seed = None
        self._env_id = None
        self._success = None

    @property
    def data(self):
        """Returns the episode data."""
        return self._data

    @data.setter
    def data(self, data: dict):
        """Set the episode data."""
        self._data = data

    @property
    def seed(self):
        """Returns the random number generator seed."""
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        """Set the random number generator seed."""
        self._seed = seed

    @property
    def env_id(self):
        """Returns the environment ID."""
        return self._env_id

    @env_id.setter
    def env_id(self, env_id: int):
        """Set the environment ID."""
        self._env_id = env_id

    @property
    def next_action_index(self):
        """Returns the next action index."""
        return self._next_action_index

    @next_action_index.setter
    def next_action_index(self, index: int):
        """Set the next action index."""
        self._next_action_index = index

    @property
    def next_state_index(self):
        """Returns the next state index."""
        return self._next_state_index

    @next_state_index.setter
    def next_state_index(self, index: int):
        """Set the next state index."""
        self._next_state_index = index

    @property
    def success(self):
        """Returns the success value."""
        return self._success

    @success.setter
    def success(self, success: bool):
        """Set the success value."""
        self._success = success

    def is_empty(self):
        """Check if the episode data is empty."""
        return not self._data

    def add(self, key: str, value: torch.Tensor | dict, clone: bool = True):
        """Add a key-value pair to the dataset.

        The key can be nested by using the "/" character.
        For example: "obs/joint_pos".

        Args:
            key: The key name.
            value: The corresponding value of tensor type or of dict type.
            clone: Whether to clone the tensor value before storing it in the episode data.
        """
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.add(f"{key}/{sub_key}", sub_value, clone=clone)
            return

        stored = value.clone() if (clone and isinstance(value, torch.Tensor)) else value
        sub_keys = key.split("/")
        data = self._data
        for sub_key in sub_keys[:-1]:
            if sub_key not in data:
                data[sub_key] = {}
            data = data[sub_key]
        # Accumulate in lists to avoid copying tensors on each append.
        if sub_keys[-1] not in data:
            data[sub_keys[-1]] = []
        data[sub_keys[-1]].append(stored)

    def get_initial_state(self) -> torch.Tensor | None:
        """Get the initial state from the dataset."""
        return self._data.get("initial_state")

    def get_action(self, action_index) -> torch.Tensor | None:
        """Get the action of the specified index from the dataset."""
        if "actions" not in self._data:
            return None
        if action_index >= len(self._data["actions"]):
            return None
        return self._data["actions"][action_index]

    def get_next_action(self) -> torch.Tensor | None:
        """Get the next action from the dataset."""
        action = self.get_action(self._next_action_index)
        if action is not None:
            self._next_action_index += 1
        return action

    def get_state(self, state_index) -> dict | None:
        """Get the state of the specified index from the dataset."""
        if "states" not in self._data:
            return None
        return _index_nested(self._data["states"], state_index, keep_dim=True)

    def get_next_state(self) -> dict | None:
        """Get the next state from the dataset."""
        state = self.get_state(self._next_state_index)
        if state is not None:
            self._next_state_index += 1
        return state

    def get_joint_target(self, joint_target_index) -> dict | torch.Tensor | None:
        """Get the joint target of the specified index from the dataset."""
        if "joint_targets" not in self._data:
            return None
        return _index_nested(self._data["joint_targets"], joint_target_index, keep_dim=False)

    def get_next_joint_target(self) -> dict | torch.Tensor | None:
        """Get the next joint target from the dataset."""
        joint_target = self.get_joint_target(self._next_joint_target_index)
        if joint_target is not None:
            self._next_joint_target_index += 1
        return joint_target

    def pre_export(self):
        """Prepare data for export by converting lists to tensors."""

        def pre_export_helper(data):
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = torch.stack(value)
                elif isinstance(value, dict):
                    pre_export_helper(value)

        pre_export_helper(self._data)


def _index_nested(data: dict | torch.Tensor, index: int, keep_dim: bool) -> dict | torch.Tensor | None:
    """Select ``index`` along the first dimension of every tensor in a (nested) dict.

    Returns None when the index is out of range for any tensor. With ``keep_dim`` the selected
    slice keeps a leading dimension of size one.
    """
    if isinstance(data, dict):
        output = {}
        for key, value in data.items():
            output[key] = _index_nested(value, index, keep_dim)
            if output[key] is None:
                return None
        return output
    if isinstance(data, torch.Tensor):
        if index >= len(data):
            return None
        return data[index, None] if keep_dim else data[index]
    raise ValueError(f"Invalid data type: {type(data)}")
