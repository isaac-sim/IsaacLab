# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test LeRobot dataset file handler functionality."""

import os
import shutil
import tempfile
import torch
import uuid
import pytest
from typing import TYPE_CHECKING

from isaaclab.utils.datasets import EpisodeData

if TYPE_CHECKING:
    from isaaclab.utils.datasets import LeRobotDatasetFileHandler

try:
    from isaaclab.utils.datasets import LeRobotDatasetFileHandler  # type: ignore
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


@pytest.mark.skipif(not LEROBOT_AVAILABLE, reason="LeRobot dependencies not available")
class TestLeRobotDatasetFileHandler:
    """Test LeRobot dataset file handler."""

    def create_test_episode(self, device):
        """Create a test episode with dummy data."""
        test_episode = EpisodeData()

        test_episode.seed = 42
        test_episode.success = True

        # Add some dummy observations and actions
        test_episode.add("obs/joint_pos", torch.tensor([1.0, 2.0, 3.0], device=device))
        test_episode.add("obs/joint_pos", torch.tensor([1.1, 2.1, 3.1], device=device))
        test_episode.add("obs/joint_pos", torch.tensor([1.2, 2.2, 3.2], device=device))

        test_episode.add("actions", torch.tensor([0.1, 0.2], device=device))
        test_episode.add("actions", torch.tensor([0.3, 0.4], device=device))
        test_episode.add("actions", torch.tensor([0.5, 0.6], device=device))

        return test_episode

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test datasets."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # cleanup after tests
        shutil.rmtree(temp_dir)

    def test_import_available(self):
        """Test that LeRobot handler can be imported."""
        assert LEROBOT_AVAILABLE, "LeRobot dependencies should be available for testing"
        handler = LeRobotDatasetFileHandler()
        assert handler is not None

    def test_create_dataset_file(self, temp_dir):
        """Test creating a new LeRobot dataset file."""
        dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.lerobot")
        handler = LeRobotDatasetFileHandler()
        
        # Test creating with .lerobot extension
        handler.create(dataset_file_path, "test_env_name")
        assert handler.get_env_name() == "test_env_name"
        handler.close()

        # Test creating without extension (should add .lerobot)
        dataset_file_path_no_ext = os.path.join(temp_dir, f"{uuid.uuid4()}")
        handler = LeRobotDatasetFileHandler()
        handler.create(dataset_file_path_no_ext, "test_env_name")
        assert handler.get_env_name() == "test_env_name"
        handler.close()

    @pytest.mark.parametrize("device", ["cpu"])  # Only test CPU for CI compatibility
    def test_write_episode(self, temp_dir, device):
        """Test writing an episode to the LeRobot dataset."""
        dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.lerobot")
        handler = LeRobotDatasetFileHandler()
        handler.create(dataset_file_path, "test_env_name")

        test_episode = self.create_test_episode(device)

        # Write the episode to the dataset
        handler.write_episode(test_episode)
        assert handler.get_num_episodes() == 1

        # Write another episode
        handler.write_episode(test_episode)
        assert handler.get_num_episodes() == 2

        handler.flush()
        handler.close()

    def test_get_properties(self, temp_dir):
        """Test getting dataset properties."""
        dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.lerobot")
        handler = LeRobotDatasetFileHandler()
        handler.create(dataset_file_path, "test_env_name")

        # Test environment name
        assert handler.get_env_name() == "test_env_name"
        
        # Test episode count (should be 0 for new dataset)
        assert handler.get_num_episodes() == 0
        
        # Test episode names (should be empty for new dataset)
        episode_names = list(handler.get_episode_names())
        assert len(episode_names) == 0

        handler.close()


if __name__ == "__main__":
    pytest.main([__file__]) 