# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Base MimicEnvCfg object for Isaac Lab Mimic data generation.
"""
import enum

from isaaclab.utils import configclass


@configclass
class DataGenConfig:
    """Configuration settings for data generation processes within the Isaac Lab Mimic environment."""

    name: str = "demo"
    """The name of the data generation process. Defaults to "demo"."""

    generation_guarantee: bool = True
    """Whether to retry generation until generation_num_trials successful demos have been generated.

    If True, generation will be retried until generation_num_trials successful demos are created.
    If False, generation will stop after generation_num_trails, regardless of success.
    """

    generation_keep_failed: bool = False
    """Whether to keep failed generation trials.

    Keeping failed demonstrations is useful for visualizing and debugging low success rates.
    """

    max_num_failures: int = 50
    """Maximum number of failures allowed before stopping generation."""

    seed: int = 1
    """Seed for randomization to ensure reproducibility."""

    """The following configuration values can be changed on the command line, and only serve as defaults."""

    source_dataset_path: str = None
    """Path to the source dataset for mimic generation."""

    generation_path: str = None
    """Path where the generated data will be saved."""

    generation_num_trials: int = 10
    """Number of trials to be generated."""

    task_name: str = None
    """Name of the task being configured."""

    """The following configurations are advanced and do not usually need to be changed."""

    generation_select_src_per_subtask: bool = False
    """Whether to select source data per subtask.

    Note:
        This requires subtasks to be properly temporally constrained, and may require
        additional subtasks to allow for time synchronization.
    """

    generation_select_src_per_arm: bool = False
    """Whether to select source data per arm."""

    generation_transform_first_robot_pose: bool = False
    """Whether to transform the first robot pose during generation."""

    generation_interpolate_from_last_target_pose: bool = True
    """Whether to interpolate from last target pose."""


@configclass
class SubTaskConfig:
    """
    Configuration settings for specifying subtasks used in Mimic environments.
    """

    """Mandatory options that should be defined for every subtask."""

    object_ref: str = None
    """Reference to the object involved in this subtask.

    Set to None if no object is involved (this is rarely the case).
    """

    subtask_term_signal: str = None
    """Subtask termination signal name."""

    """Advanced options for tuning the generation results."""

    selection_strategy: str = "random"
    """Strategy for selecting a subtask segment.

    Can be one of:
        * 'random'
        * 'nearest_neighbor_object'
        * 'nearest_neighbor_robot_distance'

    Note:
        For 'nearest_neighbor_object' and 'nearest_neighbor_robot_distance', the subtask needs
        to have 'object_ref' set to a value other than 'None'. These strategies typically yield
        higher success rates than the default 'random' strategy when object_ref is set.
    """

    selection_strategy_kwargs: dict = {}
    """Additional arguments to the selected strategy. See details on each strategy in
    source/isaaclab_mimic/isaaclab_mimic/datagen/selection_strategy.py
    Arguments will be passed through to the `select_source_demo` method."""

    first_subtask_start_offset_range: tuple = (0, 0)
    """Range for start offset of the first subtask."""

    subtask_term_offset_range: tuple = (0, 0)
    """Range for offsetting subtask termination."""

    action_noise: float = 0.03
    """Amplitude of action noise applied."""

    num_interpolation_steps: int = 5
    """Number of steps for interpolation between waypoints."""

    num_fixed_steps: int = 0
    """Number of fixed steps for the subtask."""

    apply_noise_during_interpolation: bool = False
    """Whether to apply noise during interpolation."""

    description: str = ""
    """Description of the subtask"""

    next_subtask_description: str = ""
    """Instructions for the next subtask"""


class SubTaskConstraintType(enum.IntEnum):
    """Enum for subtask constraint types."""

    SEQUENTIAL = 0
    COORDINATION = 1

    _SEQUENTIAL_FORMER = 2
    _SEQUENTIAL_LATTER = 3


class SubTaskConstraintCoordinationScheme(enum.IntEnum):
    """Enum for coordination schemes."""

    REPLAY = 0
    TRANSFORM = 1
    TRANSLATE = 2


@configclass
class SubTaskConstraintConfig:
    """
    Configuration settings for specifying subtask constraints used in multi-eef Mimic environments.
    """

    eef_subtask_constraint_tuple: list[tuple[str, int]] = (("", 0), ("", 0))
    """List of associated subtasks tuples in order.

    The first element of the tuple refers to the eef name.
    The second element of the tuple refers to the subtask index of the eef.
    """

    constraint_type: SubTaskConstraintType = None
    """Type of constraint to apply between subtasks."""

    sequential_min_time_diff: int = -1
    """Minimum time difference between two sequential subtasks finishing.

    The second subtask will execute until sequential_min_time_diff steps left in its subtask trajectory
    and wait until the first (preconditioned) subtask is finished to continue executing the rest.
    If set to -1, the second subtask will start only after the first subtask is finished.
    """

    coordination_scheme: SubTaskConstraintCoordinationScheme = SubTaskConstraintCoordinationScheme.REPLAY
    """Scheme to use for coordinating subtasks."""

    coordination_scheme_pos_noise_scale: float = 0.0
    """Scale of position noise to apply during coordination."""

    coordination_scheme_rot_noise_scale: float = 0.0
    """Scale of rotation noise to apply during coordination."""

    coordination_synchronize_start: bool = False
    """Whether subtasks should start at the same time."""

    def generate_runtime_subtask_constraints(self):
        """
        Populate expanded task constraints dictionary based on the task constraint config.
        The task constraint config contains the configurations set by the user. While the
        task_constraints_dict contains flags used to implement the constraint logic in this class.

        The task_constraint_configs may include the following types:
        - "sequential"
        - "coordination"

        For a "sequential" constraint:
            - Data from task_constraint_configs is added to task_constraints_dict as "sequential former" task constraint.
            - The opposite constraint, of type "sequential latter", is also added to task_constraints_dict.
            - Additionally, a ("fulfilled", Bool) key-value pair is added to task_constraints_dict.
            - This is used to check if the precondition (i.e., the sequential former task) has been met.
            - Until the "fulfilled" flag in "sequential latter" is set by "sequential former",
                the "sequential latter" subtask will remain paused.

        For a "coordination" constraint:
            - Data from task_constraint_configs is added to task_constraints_dict.
            - The opposite constraint, of type "coordination", is also added to task_constraints_dict.
            - The number of synchronous steps is set to the minimum of subtask_len and concurrent_subtask_len.
            - This ensures both concurrent tasks end at the same time step.
            - A "selected_src_demo_ind" and "transform" field are used to ensure the transforms used by both subtasks are the same.
        """
        task_constraints_dict = dict()
        if self.constraint_type == SubTaskConstraintType.SEQUENTIAL:
            constrained_task_spec_key, constrained_subtask_ind = self.eef_subtask_constraint_tuple[1]
            assert isinstance(constrained_subtask_ind, int)
            pre_condition_task_spec_key, pre_condition_subtask_ind = self.eef_subtask_constraint_tuple[0]
            assert isinstance(pre_condition_subtask_ind, int)
            assert (
                constrained_task_spec_key,
                constrained_subtask_ind,
            ) not in task_constraints_dict, "only one constraint per subtask allowed"
            task_constraints_dict[(constrained_task_spec_key, constrained_subtask_ind)] = dict(
                type=SubTaskConstraintType._SEQUENTIAL_LATTER,
                pre_condition_task_spec_key=pre_condition_task_spec_key,
                pre_condition_subtask_ind=pre_condition_subtask_ind,
                min_time_diff=self.sequential_min_time_diff,
                fulfilled=False,
            )
            task_constraints_dict[(pre_condition_task_spec_key, pre_condition_subtask_ind)] = dict(
                type=SubTaskConstraintType._SEQUENTIAL_FORMER,
                constrained_task_spec_key=constrained_task_spec_key,
                constrained_subtask_ind=constrained_subtask_ind,
            )
        elif self.constraint_type == SubTaskConstraintType.COORDINATION:
            constrained_task_spec_key, constrained_subtask_ind = self.eef_subtask_constraint_tuple[0]
            assert isinstance(constrained_subtask_ind, int)
            concurrent_task_spec_key, concurrent_subtask_ind = self.eef_subtask_constraint_tuple[1]
            assert isinstance(concurrent_subtask_ind, int)
            if self.coordination_scheme is None:
                raise ValueError("Coordination scheme must be specified.")
            assert (
                constrained_task_spec_key,
                constrained_subtask_ind,
            ) not in task_constraints_dict, "only one constraint per subtask allowed"
            task_constraints_dict[(constrained_task_spec_key, constrained_subtask_ind)] = dict(
                concurrent_task_spec_key=concurrent_task_spec_key,
                concurrent_subtask_ind=concurrent_subtask_ind,
                type=SubTaskConstraintType.COORDINATION,
                fulfilled=False,
                finished=False,
                selected_src_demo_ind=None,
                coordination_scheme=self.coordination_scheme,
                coordination_scheme_pos_noise_scale=self.coordination_scheme_pos_noise_scale,
                coordination_scheme_rot_noise_scale=self.coordination_scheme_rot_noise_scale,
                coordination_synchronize_start=self.coordination_synchronize_start,
                synchronous_steps=None,  # to be calculated at runtime
            )
            task_constraints_dict[(concurrent_task_spec_key, concurrent_subtask_ind)] = dict(
                concurrent_task_spec_key=constrained_task_spec_key,
                concurrent_subtask_ind=constrained_subtask_ind,
                type=SubTaskConstraintType.COORDINATION,
                fulfilled=False,
                finished=False,
                selected_src_demo_ind=None,
                coordination_scheme=self.coordination_scheme,
                coordination_scheme_pos_noise_scale=self.coordination_scheme_pos_noise_scale,
                coordination_scheme_rot_noise_scale=self.coordination_scheme_rot_noise_scale,
                coordination_synchronize_start=self.coordination_synchronize_start,
                synchronous_steps=None,  # to be calculated at runtime
            )
        else:
            raise ValueError("Constraint type not supported.")

        return task_constraints_dict


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
    subtask_configs: dict[str, list[SubTaskConfig]] = {}

    # List of configurations for subtask constraints
    task_constraint_configs: list[SubTaskConstraintConfig] = []
