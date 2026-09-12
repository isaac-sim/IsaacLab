# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL teacher-student distillation for camera-based Franka stacking."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationRunnerCfg, RslRlMLPModelCfg

from .rsl_rl_ppo_cfg import CAMERA_POLICY_CFG, STATE_POLICY_CFG


@configclass
class FrankaStackCameraDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distill the state policy into the deployable RGB-plus-proprio policy."""

    num_steps_per_env = 8
    # Reset-table curricula need a complete first outcome for every row.
    init_at_random_ep_len = False
    max_iterations = 3000
    save_interval = 50
    experiment_name = "franka_stack_camera_distillation"
    run_name = ""
    clip_actions = 1.0
    obs_groups = {
        "student": ["policy", "base_image"],
        "teacher": ["privileged"],
    }
    # A small Gaussian makes RSL-RL collect teacher labels at states visited
    # by the student while keeping the architecture identical to camera PPO.
    student = CAMERA_POLICY_CFG.replace(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.1,
            std_range=(0.05, 0.3),
            std_type="log",
        ),
    )
    # Keep this architecture identical to the state actor so a PPO checkpoint
    # can be loaded as the frozen teacher with strict state-dict validation.
    teacher = STATE_POLICY_CFG
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name=("isaaclab_tasks.contrib.stack.config.franka.agents.distillation:ClippedTeacherDistillation"),
        num_learning_epochs=2,
        learning_rate=1.0e-4,
        gradient_length=1,
        max_grad_norm=1.0,
        loss_type="mse",
    )
