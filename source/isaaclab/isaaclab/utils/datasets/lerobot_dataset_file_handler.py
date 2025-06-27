# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import shutil
import tempfile
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections.abc import Iterable

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.utils import (
        create_empty_dataset_info,
        get_hf_features_from_features, 
        DEFAULT_FEATURES
    )
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

from .dataset_file_handler_base import DatasetFileHandlerBase
from .episode_data import EpisodeData


class LeRobotDatasetFileHandler(DatasetFileHandlerBase):
    """LeRobot dataset file handler for storing and loading episode data."""

    def __init__(self):
        """Initializes the LeRobot dataset file handler."""
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "LeRobot dependencies not available. Please install following the documentation here: "
                "https://github.com/huggingface/lerobot"
            )
        
        self._dataset = None
        self._dataset_path = None
        self._env_name = None
        self._episode_count = 0

    def create(self, file_path: str, env_name: str | None = None, env = None):
        """Create a new dataset file by automatically extracting features from environment."""
        if not file_path.endswith(".lerobot"):
            file_path += ".lerobot"
        
        self._dataset_path = Path(file_path)
        
        # Delete existing dataset if it exists
        if self._dataset_path.exists():
            print(f"Removing existing dataset at {self._dataset_path}")
            shutil.rmtree(self._dataset_path)
        
        # Extract repo_id from file path
        repo_id = self._dataset_path.name.replace('.lerobot', '')
        
        # Initialize environment name
        self._env_name = env_name or "isaac_lab_env"
        
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
        """Extract features schema from environment observations and actions."""
        if env is None:
            raise ValueError("Environment must be provided to extract features")
        
        features = {}

        features["action"] = {
            "dtype": "float32",
            "shape": (env.action_manager.total_action_dim,),
            "names": None
        }
        
        # Add hardcoded features
        features["annotation.human.action.task_description"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        }
        
        # Get observation features from observation manager active terms
        obs_sample = env.observation_manager.compute()
        
        print("obs_sample", obs_sample)
        # Extract features from nested observation structure
        for group_key, group_data in obs_sample.items():
            if isinstance(group_data, dict):
                for obs_key, obs_value in group_data.items():
                    if isinstance(obs_value, torch.Tensor):
                        if "cam" in obs_key:
                            print(f"Processing camera observation: {obs_key}")
                            print(f"Camera observation shape: {obs_value.shape}")
                            # For camera observations, remove batch dimension and use [C, H, W] format
                            # obs_value shape is typically [batch_size, H, W, C] or [H, W, C]
                            if obs_value.ndim == 4:  # [batch_size, H, W, C]
                                height, width, channels = obs_value.shape[1], obs_value.shape[2], obs_value.shape[3]
                                print(f"4D camera observation - batch size: {obs_value.shape[0]}")
                            else:  # [H, W, C]
                                height, width, channels = obs_value.shape[0], obs_value.shape[1], obs_value.shape[2]
                                print("3D camera observation - no batch dimension")
                            
                            print(f"Extracted dimensions - channels: {channels}, height: {height}, width: {width}")
                            features[f"observation.{obs_key}"] = {
                                "dtype": "video",
                                "shape": (channels, height, width),  # LeRobot expects [C, H, W]
                                "names": ["channel", "height", "width"],
                                "video_info": {
                                    "video.fps": int(1 / env.step_dt)
                                }
                            }
                            print(f"Added video feature for {obs_key} with shape {(channels, height, width)}")
                        elif "joint_pos" in obs_key:
                            print(f"Processing joint position observation: {obs_key}")
                            print(f"Joint position shape: {obs_value.shape}")
                            # State observation - remove batch dimension
                            if obs_value.ndim > 1:
                                shape = obs_value.shape[1:]  # Remove batch dimension
                                print(f"Multi-dimensional joint position - original shape: {obs_value.shape}, reduced shape: {shape}")
                            else:
                                shape = obs_value.shape
                                print(f"Single-dimensional joint position - shape: {shape}")
                            features["observation.state"] = {
                                "dtype": "float32",
                                "shape": shape,
                                "names": None
                            }

        return features

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
        """Add an episode to the dataset."""
        if self._dataset is None or episode.is_empty():
            return

        # Convert Isaac Lab episode data to LeRobot format and save
        self._convert_and_save_episode(episode)
        self._episode_count += 1

    def _convert_and_save_episode(self, episode: EpisodeData):
        """Convert Isaac Lab episode data to LeRobot format and save it."""
        episode_dict = episode.data

        print("episode_dict", episode_dict)
        
        # Determine number of frames
        num_frames = len(episode_dict["actions"])


        # Generate a task description
        task = f"Pick the red cube and place it on the blue cube"
        
        # Add frames one by one to the LeRobot dataset
        for frame_idx in range(num_frames):
            frame_data = {}
            
            # Process episode data
            for key, value in episode_dict.items():
                if key == "actions":
                    # Handle actions
                    frame_data["action"] = value[frame_idx].cpu().numpy()
                elif key == "obs":
                    # Handle observations - value is already the obs dictionary
                    for obs_key, obs_value in value.items():
                        if "cam" in obs_key:
                            # Convert camera data from [H, W, C] to [C, H, W] format for LeRobot
                            camera_frame = obs_value[frame_idx].cpu().numpy()
                            if camera_frame.ndim == 3:  # [H, W, C]
                                camera_frame = camera_frame.transpose(2, 0, 1)  # Convert to [C, H, W]
                            frame_data[f"observation.{obs_key}"] = camera_frame
                        elif "joint_pos" in obs_key:
                            frame_data["observation.state"] = obs_value[frame_idx].cpu().numpy()

            # Add hardcoded feature data
            frame_data["annotation.human.action.task_description"] = np.array([0], dtype=np.int64)

            # Add frame to the dataset
            self._dataset.add_frame(frame_data, task)
        
        # Save the episode
        try:
            self._dataset.save_episode()
        except Exception as e:
            print(f"Warning: Failed to save episode: {e}")

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
