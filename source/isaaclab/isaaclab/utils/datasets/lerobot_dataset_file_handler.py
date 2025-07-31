# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
LeRobot Dataset File Handler

This module provides a configuration-driven LeRobot dataset file handler that works with
all manager-based environments in Isaac Lab.

DESIGN OVERVIEW:
==================

The LeRobotDatasetFileHandler is designed to automatically extract and record episode data
from Isaac Lab environments to LeRobot dataset format. It uses a configuration-based approach
to determine which observations and actions to record.

The Lerobot format expects the dataset to contain
- observation.example_1
- observation.example_2
- observation.example_3
- observation.example_4
- observation.state
- action

The action is extract using the action manager.

The observation that correspondes to the state is configured using the state_observation_keys attribute.

The state is extract using the observation manager.
The state is a concatenation of the state_observation_keys.

The observation that are not part of the state are configured using the observation_keys_to_record attribute.

The task description is configured using the task_description attribute.


KEY FEATURES:
============

1. CONFIGURATION-DRIVEN:
   - Uses LeRobotDatasetCfg to specify which observations to record
   - Supports both regular observations and state observations

2. AUTOMATIC FEATURE EXTRACTION:
   - Analyzes environment's observation and action managers automatically
   - Handles nested observation structures with group-based access
   - Automatically detects and processes video/image features
   - Supports different action term types

3. FLEXIBLE OBSERVATION HANDLING:
   - Regular observations: saved as "observation.{key}"
   - State observations: combined into "observation.state"
   - Support for observations from different groups (policy, critic, etc.)
   - Automatic tensor shape analysis and feature specification

4. UNIVERSAL COMPATIBILITY:
   - Works with any manager-based environment
   - No hardcoded assumptions about observation or action structure
   - Adapts to different environment types automatically

USAGE PATTERNS:
==============

1. Basic Configuration:
   ```python
   # Configure the environment first
   env.cfg.lerobot_dataset.observation_keys_to_record = [("policy", "joint_pos"), ("policy", "camera_rgb")]
   env.cfg.lerobot_dataset.state_observation_keys = [("policy", "joint_vel")]
   env.cfg.lerobot_dataset.task_description = "Stack the red cube on top of the blue cube"
   
   handler = LeRobotDatasetFileHandler()
   handler.create("dataset.lerobot", env=env)
   ```

2. State Observations:
   ```python
   # Configure state observations (combined into "observation.state")
   env.cfg.lerobot_dataset.state_observation_keys = [("policy", "joint_pos"), ("policy", "joint_vel")]
   env.cfg.lerobot_dataset.observation_keys_to_record = [("policy", "camera_rgb"), ("policy", "end_effector_pos")]
   ```

3. Multi-Group Observations:
   ```python
   # Configure observations from different groups
   env.cfg.lerobot_dataset.observation_keys_to_record = [
       ("policy", "joint_pos"), 
       ("policy", "camera_rgb"), 
       ("critic", "joint_vel")
   ]
   ```

4. Video/Image Support:
   ```python
   # Automatically detects and processes video/image data
   env.cfg.lerobot_dataset.observation_keys_to_record = [("policy", "camera_rgb")]
   # Handles [B, H, W, C] format and converts to [C, H, W] for LeRobot
   ```

REQUIRED CONFIGURATION:
=====================

The environment must have a LeRobotDatasetCfg configuration with at least one observation configured:

```python
env.cfg.lerobot_dataset.observation_keys_to_record = [("policy", "camera_rgb"), ("policy", "end_effector_pos")]
env.cfg.lerobot_dataset.state_observation_keys = [("policy", "joint_pos"), ("policy", "joint_vel")]
env.cfg.lerobot_dataset.task_description = "Custom task description"
```

This handler provides a streamlined way to record Isaac Lab environments to LeRobot datasets
with minimal configuration and maximum flexibility.
"""

import shutil
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from collections.abc import Iterable

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

from .dataset_file_handler_base import DatasetFileHandlerBase
from .episode_data import EpisodeData


class LeRobotDatasetFileHandler(DatasetFileHandlerBase):
    """LeRobot dataset file handler for storing and loading episode data.
    
    This handler is designed to work with all manager-based environments by automatically
    extracting features from the environment's observation and action managers. It provides
    flexible configuration options for customizing how observations and actions are mapped
    to LeRobot dataset features.
    
    Key Features:
    - Automatic feature extraction from observation and action managers
    - Support for nested observation groups and terms
    - Flexible video/image detection and processing
    - Customizable feature mapping
    - Support for different action term types
    - Configurable task description generation
    - Support for state observations (saved as "observation.state")
    - Configurable observation group selection
    
    Configuration Options:
    - observation_keys_to_record: List of (group_name, observation_key) tuples to save as "observation.{key}"
    - state_observation_keys: List of (group_name, observation_key) tuples to combine into "observation.state"
    - task_description: String to use as the task description for all episodes
    
    Example Usage:
    
    ```python
    # Basic usage with automatic feature extraction
    handler = LeRobotDatasetFileHandler()
    handler.create("my_dataset.lerobot", env_name="my_env", env=env)
    
    # Configure task description
    env.cfg.lerobot_dataset.task_description = "Stack the red cube on top of the blue cube"
    
    # Configure state observations
    env.cfg.lerobot_dataset.state_observation_keys = [("policy", "joint_pos"), ("policy", "joint_vel")]
    env.cfg.lerobot_dataset.observation_keys_to_record = [("policy", "camera_rgb"), ("policy", "end_effector_pos")]
    
    # Configure observations from different groups
    env.cfg.lerobot_dataset.observation_keys_to_record = [
        ("policy", "joint_pos"), 
        ("policy", "camera_rgb"), 
        ("critic", "joint_vel")
    ]
    ```
    """

    def __init__(self, 
                 config: Optional[Any] = None):
        """Initialize the LeRobot dataset file handler.
        
        Args:
            config: Optional configuration object from the environment.
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "LeRobot dependencies not available. Please install following the documentation here: "
                "https://github.com/huggingface/lerobot"
            )
        
        self._dataset = None
        self._dataset_path = None
        self._env_name = None
        self._episode_count = 0
        
        # Store configuration from environment
        self._config = config

    def create(self, file_path: str, env_name: str | None = None, env = None):
        """Create a new dataset file by automatically extracting features from environment.
        
        This method analyzes the environment's observation and action managers to automatically
        create a comprehensive feature schema for the LeRobot dataset.
        
        Args:
            file_path: Path to the dataset file (will be created with .lerobot extension if not present)
            env_name: Optional name for the environment (used in metadata)
            env: The manager-based environment instance
        """
        if not file_path.endswith(".lerobot"):
            # add .lerobot extension
            file_path += ".lerobot"
        
        self._dataset_path = Path(file_path)
        
        # Delete existing dataset if it exists
        if self._dataset_path.exists():
            # get confirmation from user
            confirm = input(f"Dataset at {self._dataset_path} already exists. Do you want to remove it? (y/n): ")
            if confirm != "y":
                raise ValueError("Dataset already exists. Please remove it or use a different file path.")
            print(f"Removing existing dataset at {self._dataset_path}")
            shutil.rmtree(self._dataset_path)
        
        # Extract repo_id from file path
        repo_id = self._dataset_path.name.replace('.lerobot', '')
        
        # Initialize environment name
        self._env_name = env_name or "isaac_lab_env"
        
        # Get configuration from environment's recorder manager if available
        if env is not None and hasattr(env, 'cfg') and hasattr(env.cfg, 'recorders'):
            recorder_config = env.cfg.recorders
            
            # Check if this is a RecorderManagerBaseCfg with LeRobot configuration
            if hasattr(recorder_config, 'observation_keys_to_record') and hasattr(recorder_config, 'state_observation_keys'):
                # Store the configuration from recorder manager
                self._config = recorder_config
            else:
                # Error out if configuration does not exist
                raise ValueError(
                    "LeRobot dataset configuration not found in recorder manager. "
                    "The recorder manager must have 'observation_keys_to_record' and 'state_observation_keys' "
                    "attributes. Please ensure the recorder manager is properly configured with LeRobot dataset settings."
                )
        else:
            # Error out if environment or recorder configuration does not exist
            raise ValueError(
                "Environment or recorder configuration not found. "
                "The environment must have a 'recorders' configuration with LeRobot dataset settings. "
                "Please ensure the environment is properly configured."
            )
        
        # Extract features from environment
        features = self._extract_features_from_env(env)
        
        # Calculate FPS from environment timestep
        fps = int(1 / env.step_dt)
        
        # Create LeRobot dataset
        try:
            self._dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=self._dataset_path,
                robot_type="isaac_lab_robot",
                use_videos=True,
                tolerance_s=1e-4
            )
            
            # Add environment name to metadata
            self._dataset.meta.info["env_name"] = self._env_name
            
        except Exception as e:
            raise RuntimeError(f"Failed to create LeRobot dataset: {e}")
        
        self._episode_count = 0

    def _extract_features_from_env(self, env) -> Dict[str, Dict]:
        """Extract features schema from environment observations and actions.
        
        This method automatically analyzes the environment's observation and action managers
        to create a comprehensive feature schema for the LeRobot dataset.
        
        Args:
            env: The manager-based environment instance
            
        Returns:
            Dictionary mapping feature names to their specifications
        """
        if env is None:
            raise ValueError("Environment must be provided to extract features")
        
        features = {}

        # Extract action features
        features.update(self._extract_action_features(env))
        
        # Extract observation features
        features.update(self._extract_observation_features(env))
        
        # Add annotation features
        features.update(self._extract_annotation_features(env))

        return features

    def _extract_action_features(self, env) -> Dict[str, Dict]:
        """Extract action features from the action manager.
        
        This method handles both the main action tensor and individual action terms.
        """
        features = {}
        
        # Add main action feature
        features["action"] = {
            "dtype": "float32",
            "shape": (env.action_manager.total_action_dim,),
            "names": None
        }
        
        return features

    def _extract_observation_features(self, env) -> Dict[str, Dict]:
        """Extract observation features from the observation manager.
        
        This method handles both concatenated observation groups and individual terms
        within groups, automatically detecting video/image features.
        Uses configuration to determine which observation keys to record.
        """
        features = {}
        
        # Get observation sample to analyze structure
        obs_sample = env.observation_manager.compute()
        
        # Get the lists of observation keys to record from configuration
        observation_keys_to_record = self._config.observation_keys_to_record
        state_observation_keys = self._config.state_observation_keys

        print(f"observation_keys_to_record: {observation_keys_to_record}")
        print(f"state_observation_keys: {state_observation_keys}")
        
        # Validate configuration - ensure both observation types are configured
        if not observation_keys_to_record:
            raise ValueError(
                "RecorderManagerBaseCfg must have observation_keys_to_record configured. "
                "Please set observation_keys_to_record with format: [('group_name', 'observation_key'), ...]"
            )
        
        if not state_observation_keys:
            raise ValueError(
                "RecorderManagerBaseCfg must have state_observation_keys configured. "
                "Please set state_observation_keys with format: [('group_name', 'observation_key'), ...]"
            )
        
        # Track state observations to combine them
        state_observations = []
        
        # Process each (group_name, observation_key) tuple
        for group_name, obs_key in observation_keys_to_record:
            # Validate that the group exists
            if group_name not in obs_sample:
                available_groups = list(obs_sample.keys())
                raise ValueError(
                    f"Observation group '{group_name}' not found. "
                    f"Available groups: {available_groups}"
                )
            
            # Validate that the observation key exists in the group
            if obs_key not in obs_sample[group_name]:
                available_keys = list(obs_sample[group_name].keys())
                raise ValueError(
                    f"Observation key '{obs_key}' not found in group '{group_name}'. "
                    f"Available keys: {available_keys}"
                )
            
            value = obs_sample[group_name][obs_key]
            if isinstance(value, torch.Tensor):
                print(f"Processing observation: {group_name}.{obs_key}")
                feature_name = f"observation.{obs_key}"
                features[feature_name] = self._analyze_tensor_feature(value, env)
            else:
                raise ValueError(f"Observation {group_name}.{obs_key} is not a tensor")
        
        # Process state observations
        for group_name, obs_key in state_observation_keys:
            # Validate that the group exists
            if group_name not in obs_sample:
                available_groups = list(obs_sample.keys())
                raise ValueError(
                    f"State observation group '{group_name}' not found. "
                    f"Available groups: {available_groups}"
                )
            
            # Validate that the observation key exists in the group
            if obs_key not in obs_sample[group_name]:
                available_keys = list(obs_sample[group_name].keys())
                raise ValueError(
                    f"State observation key '{obs_key}' not found in group '{group_name}'. "
                    f"Available keys: {available_keys}"
                )
            
            value = obs_sample[group_name][obs_key]
            if isinstance(value, torch.Tensor):
                state_observations.append((obs_key, value))
            else:
                raise ValueError(f"State observation {group_name}.{obs_key} is not a tensor")
        
        # Create combined state feature if we have state observations
        if len(state_observations) == 1:
            # Single state observation
            key, value = state_observations[0]
            features["observation.state"] = self._analyze_tensor_feature(value, env)
        else:
            # Multiple state observations - combine their features
            total_dim = 0
            for key, value in state_observations:
                # Calculate the flattened dimension for this state observation
                if value.ndim > 0:
                    dim = value.shape[1] if value.ndim > 1 else 1
                else:
                    dim = 1
                total_dim += dim
            
            # Create combined state feature
            features["observation.state"] = {
                "dtype": "float32",
                "shape": (total_dim,),
                "names": None
            }
            print(f"Combined {len(state_observations)} state observations into single 'observation.state' feature with {total_dim} dimensions")
        
        return features

    def _extract_annotation_features(self, env) -> Dict[str, Dict]:
        """Extract annotation features."""
        return {
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": (1,),
                "names": None
            }
        }

    def _analyze_tensor_feature(self, tensor: torch.Tensor, env) -> Dict[str, Any]:
        """Analyze a tensor to determine its LeRobot feature specification.
        
        Automatically detects video/image features and handles different tensor shapes.
        """
        # Remove batch dimension for feature specification
        if tensor.ndim > 0:
            shape = tensor.shape[1:] if tensor.ndim > 1 else (1,)
        else:
            shape = (1,)
        
        # Determine if this is a video/image feature
        if self._is_video_feature(tensor):
            return {
                "dtype": "video",
                "shape": self._get_video_shape(tensor),
                "names": ["channel", "height", "width"],
                "video_info": {
                    "video.fps": int(1 / env.step_dt)
                }
            }
        else:
            return {
                "dtype": "float32",
                "shape": shape,
                "names": None
            }

    def _is_video_feature(self, tensor: torch.Tensor) -> bool:
        """Determine if a tensor represents video/image data.
        
        Image data is expected to be exactly 4 dimensions in [B, H, W, C] format.
        Raises an error if tensor is not exactly 4D or not in [B, H, W, C] format.
        """
        # Check if tensor has exactly 4 dimensions
        if tensor.ndim == 4:
            # Check if the last 2-3 dimensions could be spatial (height, width, channels)
            spatial_dims = tensor.shape[-2:]
            # Assume it's video if spatial dimensions are reasonable for images
            is_likely_image = all(dim > 1 and dim < 10000 for dim in spatial_dims)
            
            if is_likely_image:
                # Check if format is [B, H, W, C]
                if not (tensor.shape[1] > 1 and tensor.shape[2] > 1 and tensor.shape[3] <= 4):
                    raise ValueError(
                        f"Image data must be in [B, H, W, C] format, but got shape {tensor.shape}. "
                        f"Expected format: batch_size > 0, height > 1, width > 1, channels <= 4"
                    )
                
                return True
        return False

    def _get_video_shape(self, tensor: torch.Tensor) -> tuple:
        """Get the video shape in [C, H, W] format for LeRobot.
        
        Expects exactly 4D tensors in [B, H, W, C] format.
        """
        if tensor.ndim != 4:
            raise ValueError(
                f"Video tensor must have exactly 4 dimensions, but got {tensor.ndim} "
                f"dimensions with shape {tensor.shape}"
            )
        
        # Check if format is [B, H, W, C]
        if not (tensor.shape[1] > 1 and tensor.shape[2] > 1 and tensor.shape[3] <= 4):
            raise ValueError(
                f"Video tensor must be in [B, H, W, C] format, but got shape {tensor.shape}. "
                f"Expected format: batch_size > 0, height > 1, width > 1, channels <= 4"
            )
        
        # Convert from [B, H, W, C] to [C, H, W] for LeRobot
        return (tensor.shape[3], tensor.shape[1], tensor.shape[2])

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        raise NotImplementedError("Open not implemented for LeRobot handler")

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        return self._env_name

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        if self._dataset is None:
            return []
        return [f"episode_{i:06d}" for i in range(self._episode_count)]

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return self._episode_count

    def write_episode(self, episode: EpisodeData):
        """Add an episode to the dataset.
        
        Converts Isaac Lab episode data to LeRobot format and saves it to the dataset.
        """
        if self._dataset is None or episode.is_empty():
            return

        # Convert Isaac Lab episode data to LeRobot format and save
        self._convert_and_save_episode(episode)
        self._episode_count += 1

    def _convert_and_save_episode(self, episode: EpisodeData):
        """Convert Isaac Lab episode data to LeRobot format and save it.
        
        This method processes each frame of the episode, converting observations and actions
        to the appropriate LeRobot format.
        """
        episode_dict = episode.data
        
        # Determine number of frames from actions
        if "actions" not in episode_dict:
            raise ValueError("No actions found in episode data")
        if "obs" not in episode_dict:
            raise ValueError("No observations found in episode data")
        
        # Get the number of frames from the actions tensor
        actions_tensor = episode_dict["actions"]
        num_frames = actions_tensor.shape[0]

        # Generate task description
        task = self._config.task_description or "Isaac Lab task"
        
        # Add frames one by one to the LeRobot dataset
        for frame_idx in range(num_frames):
            try:
                frame_data = {}
                
                # Process actions if available
                if "actions" in episode_dict:
                    actions_tensor = episode_dict["actions"]
                    frame_action = actions_tensor[frame_idx]
                    frame_data.update(self._process_actions(frame_action))
                
                # Process observations if available
                if "obs" in episode_dict:
                    obs_dict = episode_dict["obs"]
                    # Extract frame-specific observations
                    frame_obs = self._extract_frame_observations(obs_dict, frame_idx)
                    frame_data.update(self._process_observations(frame_obs))
                
                # Add annotation data
                frame_data["annotation.human.action.task_description"] = np.array([0], dtype=np.int64)
                
                # Add frame to the dataset
                self._dataset.add_frame(frame_data, task)
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                print(f"Frame data keys: {list(frame_data.keys()) if 'frame_data' in locals() else 'N/A'}")
                continue
        
        # Save the episode
        try:
            self._dataset.save_episode()
        except Exception as e:
            print(f"Warning: Failed to save episode: {e}")

    def _extract_frame_observations(self, obs_dict: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
        """Extract observations for a specific frame from the batch tensor.
        
        Args:
            obs_dict: Dictionary containing observation tensors with batch dimension
            frame_idx: Index of the frame to extract
            
        Returns:
            Dictionary containing observations for the specific frame
        """
        frame_obs = {}
        
        # Get the lists of observation keys to record from configuration
        observation_keys_to_record = self._config.observation_keys_to_record
        state_observation_keys = self._config.state_observation_keys
        
        # Extract observations from the correct groups
        for group_name, obs_key in observation_keys_to_record + state_observation_keys:
            if obs_key in obs_dict:
                try:
                    value = obs_dict[obs_key]
                    # Extract the frame from the batch dimension
                    frame_obs[obs_key] = value[frame_idx]

                except Exception as e:
                    print(f"Error extracting observation for key '{obs_key}' at frame {frame_idx}: {e}")
                    print(f"Value shape: {value.shape}")
                    raise Exception(f"Error extracting observation for key '{obs_key}' at frame {frame_idx}: {e}")
            else:
                print(f"Warning: Observation key '{obs_key}' not found in episode data")
        
        return frame_obs

    def _extract_frame_states(self, states_dict: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
        """Extract states for a specific frame from the batch tensor.
        
        Args:
            states_dict: Dictionary containing state tensors with batch dimension
            frame_idx: Index of the frame to extract
            
        Returns:
            Dictionary containing states for the specific frame
        """
        frame_states = {}
        
        for key, value in states_dict.items():
            try:
                if value.ndim > 0 and frame_idx < value.shape[0]:
                    # Extract the frame from the batch dimension
                    frame_states[key] = value[frame_idx]
                else:
                    # Handle 0D tensors or tensors without batch dimension
                    frame_states[key] = value
            except Exception as e:
                print(f"Error extracting state for key '{key}' at frame {frame_idx}: {e}")
                print(f"Value shape: {value.shape}")
                # Skip this state if there's an error
                continue
        
        return frame_states

    def _process_actions(self, action_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Process actions for a single frame.
        
        Handles both the main action tensor and individual action terms.
        """
        frame_data = {}
        
        # Convert tensor to numpy array
        frame_data["action"] = action_tensor.cpu().numpy()
        
        # Process individual action terms if available
        # Note: This would require access to the action manager to split actions
        # For now, we'll just add the main action
        
        return frame_data

    def _process_observations(self, obs_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process observations for a single frame.
        
        Uses configuration to determine which observation keys to record.
        """
        frame_data = {}
        
        # Get the lists of observation keys to record from configuration
        observation_keys_to_record = self._config.observation_keys_to_record
        state_observation_keys = self._config.state_observation_keys
        
        # Track state observations to combine them
        state_observations = []
        
        # Process regular observations
        for group_name, obs_key in observation_keys_to_record:
            if obs_key in obs_dict:
                try:
                    feature_name = f"observation.{obs_key}"
                    processed_value = self._process_observation_term(obs_key, obs_dict[obs_key])
                    frame_data[feature_name] = processed_value
                except Exception as e:
                    print(f"Error processing observation '{obs_key}': {e}")
                    continue
            else:
                print(f"Warning: Observation key '{obs_key}' not found in frame data'")
        
        # Process state observations
        for group_name, obs_key in state_observation_keys:
            if obs_key in obs_dict:
                try:
                    processed_value = self._process_observation_term(obs_key, obs_dict[obs_key])
                    state_observations.append(processed_value)
                except Exception as e:
                    print(f"Error processing state observation '{group_name}.{obs_key}': {e}")
                    continue
            else:
                print(f"Warning: State observation key '{obs_key}' not found in frame data'")
        
        # Combine state observations into a single "observation.state" feature
        if state_observations:
            if len(state_observations) == 1:
                # Single state observation
                frame_data["observation.state"] = state_observations[0]
            else:
                # Multiple state observations - concatenate them
                # Assuming all state observations are 1D arrays
                concatenated_state = np.concatenate(state_observations)
                frame_data["observation.state"] = concatenated_state
                print(f"Combined {len(state_observations)} state observations into single 'observation.state' feature")
        
        return frame_data

    def _process_states(self, states_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process states for a single frame.
        
        States are typically privileged information that may not be available
        in real-world scenarios.
        """
        frame_data = {}
        
        for key, value in states_dict.items():
            try:
                feature_name = f"state.{key}"
                frame_data[feature_name] = value.cpu().numpy()
            except Exception as e:
                print(f"Error processing state '{key}': {e}")
                continue
        
        return frame_data

    def _process_observation_term(self, term_name: str, tensor: torch.Tensor) -> np.ndarray:
        """Process a single observation term.
        
        Uses default processing for all observation terms.
        """
        # Default processing
        numpy_array = tensor.cpu().numpy()
        
        # Handle video/image data
        if self._is_video_feature(tensor):
            # _is_video_feature already ensures tensor is 4D in [B, H, W, C] format
            if numpy_array.ndim == 4:  # [B, H, W, C]
                numpy_array = numpy_array.transpose(0, 3, 1, 2)  # Convert to [B, C, H, W]
            else:
                # This should never happen since _is_video_feature ensures 4D
                raise ValueError(f"Unexpected video tensor dimensions for {term_name}: {numpy_array.ndim}, expected 4")
        
        return numpy_array

    def load_episode(self, episode_name: str) -> EpisodeData | None:
        """Load episode data from the file."""
        raise NotImplementedError("Load episode not implemented for LeRobot handler")

    def flush(self):
        """Flush any pending data to disk."""
        # LeRobot dataset handles flushing automatically when save_episode() is called
        pass

    def close(self):
        """Close the dataset file handler."""
        # Stop any async image writers
        if self._dataset and hasattr(self._dataset, 'image_writer') and self._dataset.image_writer:
            self._dataset.stop_image_writer()
        
        # Clear references
        self._dataset = None
