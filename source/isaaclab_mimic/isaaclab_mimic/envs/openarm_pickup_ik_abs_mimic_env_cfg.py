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
        # KEY: False = both subtasks come from ONE source demo → no seam discontinuity.
        # True would pick a different source demo for each subtask, creating a hard
        # EEF-pose jump at the boundary that is the main cause of jerkiness.
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_transform_first_robot_pose = False
        # True = interpolation at subtask seams starts from the LAST COMMANDED pose
        # (smoother when IK tracking is good). Set False if the robot often lags
        # behind its command, which would cause a jump from actual→commanded position.
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
                # Shorter tail: 2-5 extra steps keeps the seam boundary tight
                # and reduces the random positional spread at transition.
                subtask_term_offset_range=(2, 5),
                selection_strategy="nearest_neighbor_object",
                # nn_k=5: more candidates → pick the closest source demo geometry
                selection_strategy_kwargs={"nn_k": 5},
                # Near-zero noise: clean data for policy training.
                # Add noise only if you explicitly want data augmentation.
                action_noise=0.005,
                # 25 interp steps ≈ ~0.8 s at 30 Hz to bridge from robot start
                # pose to the first waypoint of the source segment. Smooth enough
                # to remove the "snap" at the beginning of each subtask.
                num_interpolation_steps=25,
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
                selection_strategy_kwargs={"nn_k": 5},
                action_noise=0.005,
                # Since generation_select_src_per_subtask=False, subtask 2 is always
                # from the same demo as subtask 1 → the seam is naturally tight and
                # fewer interp steps are needed here.
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Lift the red cube above the table",
            )
        )

        # Key must match what get_robot_eef_pose / target_eef_pose_to_action use;
        # "left_eef" identifies the left end-effector that Mimic will control.
        self.subtask_configs["left_eef"] = subtask_configs
