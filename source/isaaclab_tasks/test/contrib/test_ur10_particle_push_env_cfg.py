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


def test_default_config_is_valid():
    """The authored reset envelope and runtime bounds are internally consistent."""
    UR10ParticlePushEnvCfg().validate()


def test_final_validation_checks_post_construction_overrides():
    """The standard config hook validates values after Hydra-style updates."""
    cfg = UR10ParticlePushEnvCfg()
    cfg.from_dict({"reset_pose_count": 191})

    with pytest.raises(ValueError, match="reset_pose_count must be positive and divisible"):
        cfg.validate()


def test_curriculum_success_at_maximum_level_stays_capped():
    """Repeated successes must not advance past the reset-scale table."""
    curriculum = SinglePushCurriculum.__new__(SinglePushCurriculum)
    curriculum._initial_level = 1
    curriculum._levels = torch.tensor([1])
    curriculum._maximum_level = 2
    curriculum._scales = torch.tensor([0.35, 0.65, 1.0])
    env = SimpleNamespace(
        common_step_counter=1,
        device="cpu",
        num_envs=1,
        success_this_step=torch.tensor([True]),
    )

    curriculum(env, torch.tensor([0]), initial_level=1)
    state = curriculum(env, torch.tensor([0]), initial_level=1)

    assert curriculum.levels.item() == 2
    assert state["randomization_scale"].item() == 1.0
