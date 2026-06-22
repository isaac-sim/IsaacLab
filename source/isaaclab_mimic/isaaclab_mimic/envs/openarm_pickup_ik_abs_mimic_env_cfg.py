# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Mimic env config for the OpenArm pick-up task (IK relative)."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.openarm.pickup_ik_abs_env_cfg import (
    OpenarmPickUpRedCubeEnvCfg,
)


@configclass
class OpenArmPickUpIKAbsMimicEnvCfg(OpenarmPickUpRedCubeEnvCfg, MimicEnvCfg):
    """Isaac Lab Mimic environment config for Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0.

    Two subtasks:
      1. grasp  — left gripper closes around cube_2  (term signal: "grasp")
      2. lift   — cube_2 rises above 0.30 m          (last subtask, no term signal)
    """

    def __post_init__(self):
        super().__post_init__()

        # Dataset generation settings (relative mode = delta pose controller)
        self.datagen_config.name = "demo_src_pickup_openarm_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = []

        # Subtask 1: reach and grasp cube_2 (red cube)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Reach and grasp the red cube",
                next_subtask_description="Lift the red cube",
            )
        )

        # Subtask 2 (last): lift cube_2 above the table — no explicit term signal
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Lift the red cube above the table",
            )
        )

        # Key must match what get_robot_eef_pose / target_eef_pose_to_action use;
        # "left_eef" identifies the left end-effector that Mimic will control.
        self.subtask_configs["left_eef"] = subtask_configs
