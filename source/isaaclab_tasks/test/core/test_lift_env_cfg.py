# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration gates for the unified dexterous Lift and Reorient tasks."""

import torch

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FrankaLiftEnvCfg
from isaaclab_tasks.core.lift.config.kuka_allegro.agents.rsl_rl_ppo_cfg import KukaAllegroPPORunnerCfg
from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_env_cfg import (
    KukaAllegroLiftEnvCfg,
    KukaAllegroReorientEnvCfg,
)
from isaaclab_tasks.utils import resolve_presets


def test_lift_keeps_the_upstream_reset_and_progress_training_terms() -> None:
    """The package move must retain the rewritten reset bank and progress rewards."""
    for raw_env_cfg in (FrankaLiftEnvCfg(), KukaAllegroLiftEnvCfg()):
        env_cfg = resolve_presets(raw_env_cfg)
        reset_params = env_cfg.events.conditional_reset.params

        assert reset_params["success_monitor"].target_success_rate == 0.5
        assert reset_params["diversity_feature"] is not None
        assert env_cfg.rewards.position_tracking.func is mdp.position_command_progress
        assert env_cfg.rewards.orientation_tracking is None
        assert len(env_cfg.scene.object.spawn.assets_cfg) > 1


def test_reorient_keeps_orientation_progress_separate_from_lift() -> None:
    """Reorient must retain orientation tracking while Lift disables only that term."""
    env_cfg = KukaAllegroReorientEnvCfg()

    assert env_cfg.rewards.position_tracking.func is mdp.position_command_progress
    assert env_cfg.rewards.orientation_tracking.func is mdp.orientation_command_progress
    assert env_cfg.commands.object_pose.position_only is False


def test_camera_runner_keeps_the_stable_encoder_settings() -> None:
    """Camera presets must retain the fixed-rate spatial-softmax training configuration."""
    runner_cfg = KukaAllegroPPORunnerCfg()

    assert runner_cfg.single_camera.algorithm.schedule == "fixed"
    assert runner_cfg.single_camera.algorithm.learning_rate == 7.0e-5
    assert runner_cfg.duo_camera.actor.class_name == (
        "isaaclab_tasks.core.lift.config.kuka_allegro.agents.models:SpatialSoftmaxCNNModel"
    )
    assert runner_cfg.duo_camera.actor.cnn_cfg.output_channels == [16, 32, 32]


def test_camera_normalization_is_stationary() -> None:
    """RGB and depth normalization must not depend on per-frame statistics."""
    rgb = torch.tensor([0.0, 127.5, 255.0])
    depth = torch.tensor([0.0, 2.0])

    assert torch.allclose(mdp.vision_camera._rgb_norm(None, rgb), torch.tensor([-0.5, 0.0, 0.5]))
    assert torch.allclose(mdp.vision_camera._depth_norm(None, depth), torch.tanh(depth / 2) - 0.5)
