# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

from .episode_data import EpisodeData


class DatasetFileHandlerBase(ABC):
    """Abstract class for handling dataset files."""

    def __init__(self):
        """Initializes the dataset file handler."""
        pass

    @abstractmethod
    def open(self, file_path: str, mode: str = "r"):
        """Open a file."""
        return NotImplementedError

    @abstractmethod
    def create(self, file_path: str, env_name: str = None):
        """Create a new file."""
        return NotImplementedError

    @abstractmethod
    def get_env_name(self) -> str | None:
        """Get the environment name."""
        return NotImplementedError

    @abstractmethod
    def write_episode(self, episode: EpisodeData):
        """Write episode data to the file."""
        return NotImplementedError

    @abstractmethod
    def flush(self):
        """Flush the file."""
        return NotImplementedError

    @abstractmethod
    def close(self):
        """Close the file."""
        return NotImplementedError

    @abstractmethod
    def load_episode(self, episode_name: str) -> EpisodeData | None:
        """Load episode data from the file."""
        return NotImplementedError

    @abstractmethod
    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return NotImplementedError
