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
        test_episode.add("obs/policy/joint_pos", torch.tensor([1.0, 2.0, 3.0], device=device))
        test_episode.add("obs/policy/joint_pos", torch.tensor([1.1, 2.1, 3.1], device=device))
        test_episode.add("obs/policy/joint_pos", torch.tensor([1.2, 2.2, 3.2], device=device))

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
        
        # Create handler with required configuration
        from isaaclab.managers import RecorderManagerBaseCfg
        config = RecorderManagerBaseCfg()
        config.observation_keys_to_record = [("policy", "joint_pos")]
        config.state_observation_keys = [("policy", "joint_vel")]
        
        handler = LeRobotDatasetFileHandler(config=config)
        assert handler is not None

    def test_create_dataset_file(self, temp_dir):
        """Test creating a new LeRobot dataset file."""
        dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.lerobot")
        
        # Create handler with required configuration
        from isaaclab.managers import RecorderManagerBaseCfg
        config = RecorderManagerBaseCfg()
        config.observation_keys_to_record = [("policy", "joint_pos")]
        config.state_observation_keys = [("policy", "joint_vel")]
        
        handler = LeRobotDatasetFileHandler(config=config)
        
        # Test creating with .lerobot extension
        handler.create(dataset_file_path, "test_env_name")
        assert handler.get_env_name() == "test_env_name"
        handler.close()

        # Test creating without extension (should add .lerobot)
        dataset_file_path_no_ext = os.path.join(temp_dir, f"{uuid.uuid4()}")
        handler = LeRobotDatasetFileHandler(config=config)
        handler.create(dataset_file_path_no_ext, "test_env_name")
        assert handler.get_env_name() == "test_env_name"
        handler.close()

    @pytest.mark.parametrize("device", ["cpu"])  # Only test CPU for CI compatibility
    def test_write_episode(self, temp_dir, device):
        """Test writing an episode to the LeRobot dataset."""
        dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.lerobot")
        
        # Create handler with required configuration
        from isaaclab.managers import RecorderManagerBaseCfg
        config = RecorderManagerBaseCfg()
        config.observation_keys_to_record = [("policy", "joint_pos"), ("policy", "camera_rgb")]
        config.state_observation_keys = []
        
        handler = LeRobotDatasetFileHandler(config=config)
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

    @pytest.mark.parametrize("device", ["cpu"])  # Only test CPU for CI compatibility
    def test_state_observations(self, temp_dir, device):
        """Test that state observations are properly handled."""
        dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.lerobot")
        
        # Create handler with state observation configuration
        from isaaclab.managers import RecorderManagerBaseCfg
        config = RecorderManagerBaseCfg()
        config.state_observation_keys = [("policy", "joint_pos"), ("policy", "joint_vel")]
        config.observation_keys_to_record = [("policy", "camera_rgb")]
        
        handler = LeRobotDatasetFileHandler(config=config)
        handler.create(dataset_file_path, "test_env_name")

        # Create test episode with state observations
        test_episode = EpisodeData()
        test_episode.seed = 42
        test_episode.success = True

        # Add state observations
        test_episode.add("obs/policy/joint_pos", torch.tensor([1.0, 2.0, 3.0], device=device))
        test_episode.add("obs/policy/joint_pos", torch.tensor([1.1, 2.1, 3.1], device=device))
        test_episode.add("obs/policy/joint_vel", torch.tensor([0.1, 0.2, 0.3], device=device))
        test_episode.add("obs/policy/joint_vel", torch.tensor([0.11, 0.21, 0.31], device=device))

        # Add regular observations
        test_episode.add("obs/policy/camera_rgb", torch.tensor([[[[0.1, 0.2, 0.3]]]], device=device))
        test_episode.add("obs/policy/camera_rgb", torch.tensor([[[[0.11, 0.21, 0.31]]]], device=device))

        # Add actions
        test_episode.add("actions", torch.tensor([0.1, 0.2], device=device))
        test_episode.add("actions", torch.tensor([0.3, 0.4], device=device))

        # Write the episode to the dataset
        handler.write_episode(test_episode)
        assert handler.get_num_episodes() == 1

        handler.flush()
        handler.close()

    def test_get_properties(self, temp_dir):
        """Test getting dataset properties."""
        dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.lerobot")
        
        # Create handler with required configuration
        from isaaclab.managers import RecorderManagerBaseCfg
        config = RecorderManagerBaseCfg()
        config.observation_keys_to_record = [("policy", "joint_pos")]
        config.state_observation_keys = []
        
        handler = LeRobotDatasetFileHandler(config=config)
        handler.create(dataset_file_path, "test_env_name")

        # Test environment name
        assert handler.get_env_name() == "test_env_name"
        
        # Test episode count (should be 0 for new dataset)
        assert handler.get_num_episodes() == 0
        
        # Test episode names (should be empty for new dataset)
        episode_names = list(handler.get_episode_names())
        assert len(episode_names) == 0

        handler.close()

    def test_missing_configuration_error(self, temp_dir):
        """Test that appropriate errors are raised when configuration is missing."""
        dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.lerobot")
        
        # Test with both observation_keys_to_record and state_observation_keys empty (should cause an error)
        from isaaclab.managers import RecorderManagerBaseCfg
        config = RecorderManagerBaseCfg()
        config.observation_keys_to_record = []  # Empty list
        config.state_observation_keys = []  # Empty list
        
        handler = LeRobotDatasetFileHandler(config=config)
        
        # Create a mock environment for testing
        class MockEnv:
            def __init__(self):
                self.step_dt = 0.01
                self.action_manager = type('ActionManager', (), {
                    'total_action_dim': 7,
                    '_terms': {}
                })()
                self.observation_manager = type('ObservationManager', (), {
                    'compute': lambda: {'policy': {'joint_pos': torch.tensor([[1.0, 2.0, 3.0]])}}
                })()
        
        mock_env = MockEnv()
        
        # This should raise an error since both lists are empty
        with pytest.raises(ValueError, match="must have at least one observation configured"):
            handler.create(dataset_file_path, "test_env_name", env=mock_env)
        
        # Test with only observation_keys_to_record set (should work)
        config = RecorderManagerBaseCfg()
        config.observation_keys_to_record = [("policy", "joint_pos")]
        config.state_observation_keys = []  # Empty list should work if other is set
        
        handler = LeRobotDatasetFileHandler(config=config)
        
        # This should work since we have at least one observation configured
        handler.create(dataset_file_path, "test_env_name", env=mock_env)
        handler.close()
        
        # Test with only state_observation_keys set (should work)
        config = RecorderManagerBaseCfg()
        config.observation_keys_to_record = []  # Empty list
        config.state_observation_keys = [("policy", "joint_pos")]  # Should work if other is set
        
        handler = LeRobotDatasetFileHandler(config=config)
        
        # This should work since we have at least one observation configured
        handler.create(dataset_file_path, "test_env_name", env=mock_env)
        handler.close()

    @pytest.mark.parametrize("device", ["cpu"])  # Only test CPU for CI compatibility
    def test_multi_group_observations(self, temp_dir, device):
        """Test that observations from multiple groups are properly handled."""
        dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.lerobot")
        
        # Create handler with multi-group observation configuration
        from isaaclab.managers import RecorderManagerBaseCfg
        config = RecorderManagerBaseCfg()
        config.observation_keys_to_record = [
            ("policy", "joint_pos"), 
            ("policy", "camera_rgb"), 
            ("critic", "joint_vel")
        ]
        config.state_observation_keys = [("policy", "joint_pos"), ("critic", "joint_vel")]
        
        handler = LeRobotDatasetFileHandler(config=config)
        handler.create(dataset_file_path, "test_env_name")

        # Create test episode with observations from multiple groups
        test_episode = EpisodeData()
        test_episode.seed = 42
        test_episode.success = True

        # Add observations from policy group
        test_episode.add("obs/policy/joint_pos", torch.tensor([1.0, 2.0, 3.0], device=device))
        test_episode.add("obs/policy/joint_pos", torch.tensor([1.1, 2.1, 3.1], device=device))
        test_episode.add("obs/policy/camera_rgb", torch.tensor([[[[0.1, 0.2, 0.3]]]], device=device))
        test_episode.add("obs/policy/camera_rgb", torch.tensor([[[[0.11, 0.21, 0.31]]]], device=device))

        # Add observations from critic group
        test_episode.add("obs/critic/joint_vel", torch.tensor([0.1, 0.2, 0.3], device=device))
        test_episode.add("obs/critic/joint_vel", torch.tensor([0.11, 0.21, 0.31], device=device))

        # Add actions
        test_episode.add("actions", torch.tensor([0.1, 0.2], device=device))
        test_episode.add("actions", torch.tensor([0.3, 0.4], device=device))

        # Write the episode to the dataset
        handler.write_episode(test_episode)
        assert handler.get_num_episodes() == 1

        handler.flush()
        handler.close()


if __name__ == "__main__":
    pytest.main([__file__]) 