# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the UR10 particle-push task configuration."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.ur10_particle_push.mdp.curriculums import SinglePushCurriculum
from isaaclab_tasks.contrib.ur10_particle_push.ur10_particle_push_env_cfg import UR10ParticlePushEnvCfg


def test_final_validation_checks_post_construction_overrides():
    """The standard config hook validates values after Hydra-style updates."""
    cfg = UR10ParticlePushEnvCfg()
    cfg.from_dict({"reset_pose_count": 191})

    with pytest.raises(ValueError, match="reset_pose_count must be positive and divisible"):
        cfg.validate()


def test_curriculum_preserves_reset_mixture_and_tracks_level_success():
    """Successful episodes must not eliminate easier reset distributions."""
    torch.manual_seed(0)
    env_count = 10_000
    env = SimpleNamespace(
        common_step_counter=1,
        device="cpu",
        num_envs=env_count,
        cfg=SimpleNamespace(
            reset_level_probabilities=(0.2, 0.4, 0.4),
            reset_randomization_scales=(0.35, 0.65, 1.0),
        ),
        success_this_step=torch.zeros(env_count, dtype=torch.bool),
    )
    curriculum = SinglePushCurriculum(SimpleNamespace(params={}), env)
    level_0_ids = torch.nonzero(curriculum.levels == 0).flatten()
    level_1_ids = torch.nonzero(curriculum.levels == 1).flatten()
    env.success_this_step[level_0_ids[: level_0_ids.numel() // 2]] = True
    env.success_this_step[level_1_ids[: level_1_ids.numel() // 4]] = True

    state = curriculum(env, torch.arange(env_count))

    expected = torch.tensor(env.cfg.reset_level_probabilities)
    actual = torch.bincount(curriculum.levels, minlength=3).float() / env_count
    assert torch.allclose(actual, expected, atol=0.02)
    assert state["level_0_success_rate"].item() == pytest.approx((level_0_ids.numel() // 2) / level_0_ids.numel())
    assert state["level_1_success_rate"].item() == pytest.approx((level_1_ids.numel() // 4) / level_1_ids.numel())
    assert state["level_2_success_rate"].item() == pytest.approx(0.0)
