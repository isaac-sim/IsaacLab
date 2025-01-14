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
from isaaclab.utils import configclass


@configclass
class DataGenConfig:
    """Configuration settings for data generation processes within the Isaac Lab Mimic environment."""

    # The name of the datageneration, default is "demo"
    name: str = "demo"

    # If set to True, generation will be retried until
    # generation_num_trials successful demos have been generated.
    # If set to False, generation will stop after generation_num_trails,
    # independent of whether they were all successful or not.
    generation_guarantee: bool = True

    ##############################################################
    # Debugging parameters, which can help determining low success
    # rates.

    # Whether to keep failed generation trials. Keeping failed
    # demonstrations is useful for visualizing and debugging low
    # success rates.
    generation_keep_failed: bool = False

    # Maximum number of failures allowed before stopping generation
    max_num_failures: int = 50

    # Seed for randomization to ensure reproducibility
    seed: int = 1

    ##############################################################
    # The following values can be changed on the command line, and
    # only serve as defaults.

    # Path to the source dataset for mimic generation
    source_dataset_path: str = None

    # Path where the generated data will be saved
    generation_path: str = None

    # Number of trial to be generated
    generation_num_trials: int = 10

    # Name of the task being configured
    task_name: str = None

    ##############################################################
    # Advanced configuration, does not usually need to be changed

    # Whether to select source data per subtask
    # Note: this requires subtasks to be properly temporally
    #       constrained, and may require additional subtasks to allow
    #       for time synchronization.
    generation_select_src_per_subtask: bool = False

    # Whether to transform the first robot pose during generation
    generation_transform_first_robot_pose: bool = False

    # Whether to interpolate from last target pose
    generation_interpolate_from_last_target_pose: bool = True


@configclass
class SubTaskConfig:
    """
    Configuration settings specific to the management of individual
    subtasks.
    """

    ##############################################################
    # Mandatory options that should be defined for every subtask

    # Reference to the object involved in this subtask, None if no
    # object is involved (this is rarely the case).
    object_ref: str = None

    # Signal for subtask termination
    subtask_term_signal: str = None

    ##############################################################
    # Advanced options for tuning the generation results

    # Strategy on how to select a subtask segment. Can be either
    # 'random', 'nearest_neighbor_object' or
    # 'nearest_neighbor_robot_distance'. Details can be found in
    # source/isaaclab_mimic/isaaclab_mimic/datagen/selection_strategy.py
    #
    # Note: for 'nearest_neighbor_object' and
    #       'nearest_neighbor_robot_distance', the subtask needs to have
    #       'object_ref' set to a value other than 'None' above. At the
    #       same time, if 'object_ref' is not 'None', then either of
    #       those strategies will usually yield higher success rates
    #       than the default 'random' strategy.
    selection_strategy: str = "random"

    # Additional arguments to the selected strategy. See details on
    # each strategy in
    # source/isaaclab_mimic/isaaclab_mimic/datagen/selection_strategy.py
    # Arguments will be passed through to the `select_source_demo`
    # method.
    selection_strategy_kwargs: dict = {}

    # Range for start offset of the first subtask
    first_subtask_start_offset_range: tuple = (0, 0)

    # Range for offsetting subtask termination
    subtask_term_offset_range: tuple = (0, 0)

    # Amplitude of action noise applied
    action_noise: float = 0.03

    # Number of steps for interpolation between waypoints
    num_interpolation_steps: int = 5

    # Number of fixed steps for the subtask
    num_fixed_steps: int = 0

    # Whether to apply noise during interpolation
    apply_noise_during_interpolation: bool = False


@configclass
class MimicEnvCfg:
    """
    Configuration class for the Mimic environment integration.

    This class consolidates various configuration aspects for the
    Isaac Lab Mimic data generation pipeline.
    """

    # Overall configuration for the data generation
    datagen_config: DataGenConfig = DataGenConfig()

    # Dictionary of list of subtask configurations for each end-effector.
    # Keys are end-effector names.
    # Currently, only a single end-effector is supported by Isaac Lab Mimic
    # so `subtask_configs` must always be of size 1.
    subtask_configs: dict[str, list[SubTaskConfig]] = {}
