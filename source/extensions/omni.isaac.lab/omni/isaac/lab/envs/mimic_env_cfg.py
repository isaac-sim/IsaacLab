# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Base MimicEnvCfg object for Isaac Lab Mimic data generation.
"""
from omni.isaac.lab.utils import configclass


@configclass
class DataGenConfig:
    """Configuration settings for data generation processes within the Isaac Lab Mimic environment."""

    name: str = "demo"  # The name of the datageneration, default is "demo"
    source_dataset_path: str = None  # Path to the source dataset for mimic generation
    generation_path: str = None  # Path where the generated data will be saved
    generation_guarantee: bool = False  # Whether to guarantee generation of data (e.g., retry until successful)
    generation_keep_failed: bool = True  # Whether to keep failed generation trials
    generation_num_trials: int = 10  # Number of trial to be generated
    generation_select_src_per_subtask: bool = False  # Whether to select source data per subtask
    generation_transform_first_robot_pose: bool = False  # Whether to transform the first robot pose during generation
    generation_interpolate_from_last_target_pose: bool = True  # Whether to interpolate from last target pose
    task_name: str = None  # Name of the task being configured
    max_num_failures: int = 50  # Maximum number of failures allowed before stopping generation
    num_demo_to_render: int = 50  # Number of demonstrations to render
    num_fail_demo_to_render: int = 50  # Number of failed demonstrations to render
    seed: int = 1  # Seed for randomization to ensure reproducibility


@configclass
class SubTaskConfig:
    """Configuration settings specific to the management of individual subtasks."""

    object_ref: str = None  # Reference to the object involved in this subtask
    subtask_term_signal: str = None  # Signal for subtask termination
    subtask_term_offset_range: tuple = (0, 0)  # Range for offsetting subtask termination
    selection_strategy: str = None  # Strategy for selecting subtask
    selection_strategy_kwargs: dict = {}  # Keyword arguments for the selection strategy
    action_noise: float = 0.03  # Amplitude of action noise applied
    num_interpolation_steps: int = 5  # Number of steps for interpolation between waypoints
    num_fixed_steps: int = 0  # Number of fixed steps for the subtask
    apply_noise_during_interpolation: bool = False  # Whether to apply noise during interpolation


@configclass
class MimicEnvCfg:
    """Configuration class for the Mimic environment integration.

    This class consolidates various configuration aspects for the Isaac Lab Mimic data generation pipeline.
    """

    datagen_config: DataGenConfig = DataGenConfig()  # Configuration for the data generation
    subtask_configs: list[SubTaskConfig] = []  # List of configurations for each subtask
