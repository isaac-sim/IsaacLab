# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.so_100.stack_pink_ik_abs_visuomotor_env_cfg import SO100CubeStackPinkIKAbsVisuomotorEnvCfg


@configclass
class StackSO100MimicEnvCfg(SO100CubeStackPinkIKAbsVisuomotorEnvCfg, MimicEnvCfg):

    def __post_init__(self):
        # Calling post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_so100_stack_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the stack task.
        # For SO100, we only have one "gripper" arm, so all subtasks are for this single arm
        subtask_configs = []
        
        # Subtask 1: Pick up cube_2
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                # For picking up cube_2, the object of interest is cube_2
                object_ref="cube_2",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_1",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        
        # Subtask 2: Place cube_2 on cube_1
        subtask_configs.append(
            SubTaskConfig(
                # For placing cube_2 on cube_1, the object of interest is cube_1 (target location)
                object_ref="cube_1",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal=None,  # This is the final subtask, so no termination signal
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=3,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        
        # For SO100, we only have one arm "gripper", so we assign all subtasks to it
        self.subtask_configs["gripper"] = subtask_configs 