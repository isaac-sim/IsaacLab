# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.bin_stack_ik_rel_env_cfg import FrankaBinStackEnvCfg


@configclass
class FrankaBinStackIKRelMimicEnvCfg(FrankaBinStackEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Cube Stack IK Rel env.
    """

    def __post_init__(self):

        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_stack_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the stack task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp red cube",
                next_subtask_description="Stack red cube on top of blue cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="stack_1",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Grasp green cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                subtask_term_signal="grasp_2",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                next_subtask_description="Stack green cube on top of red cube",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="stack_2",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["franka"] = subtask_configs
