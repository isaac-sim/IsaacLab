# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminal-success gating for optional MPM resolution stages."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumTermCfg

from isaaclab_tasks.contrib.franka_pour.mdp.mpm_resolution import PourMPMResolutionCurriculum
from isaaclab_tasks.contrib.franka_pour.pour_env_cfg import FrankaPourResetDatasetEnvCfg, PourMPMResolutionStageCfg


def test_resolution_requires_completed_success_cohorts_and_respects_freeze():
    """Initial resets, small cohorts, and failed cohorts cannot change global physics."""
    cfg = FrankaPourResetDatasetEnvCfg()
    cfg.mpm_resolution_min_episodes = 4
    cfg.mpm_resolution_stages = (
        PourMPMResolutionStageCfg(success_rate=0.75, voxel_size=0.020),
        PourMPMResolutionStageCfg(success_rate=1.0, particle_radius=0.003),
    )
    env = SimpleNamespace(
        cfg=cfg,
        num_envs=4,
        device="cpu",
        common_step_counter=7,
        episode_length_buf=torch.zeros(4, dtype=torch.int64),
        episode_succeeded=torch.ones(4, dtype=torch.bool),
    )
    changes = []

    def record_voxel(value, *, entry_name):
        changes.append(("voxel", value, entry_name))

    def record_radius(value, *, entry_name):
        changes.append(("radius", value, entry_name))

    env.sim = SimpleNamespace(
        backend_manager=SimpleNamespace(
            set_voxel_size=record_voxel,
            set_particle_radius=record_radius,
        )
    )
    curriculum = PourMPMResolutionCurriculum(CurriculumTermCfg(func=PourMPMResolutionCurriculum), env)
    assert curriculum(env, slice(None))["stage"] == 0.0
    env.episode_length_buf.fill_(1)
    env.episode_succeeded[:] = torch.tensor([True, False, False, False])
    assert curriculum(env, [0, 1])["stage"] == 0.0
    assert curriculum(env, [2, 3])["success_rate"] == 0.25
    assert not changes
    env.episode_succeeded[:] = torch.tensor([True, True, True, False])
    assert curriculum(env, slice(None))["stage"] == 1.0
    assert curriculum(env, slice(None))["last_transition_policy_step"] == 7.0
    assert changes == [("voxel", 0.020, "media")]
    cfg.curriculum_freeze = True
    env.episode_succeeded.fill_(True)
    assert curriculum(env, slice(None))["stage"] == 1.0
    assert len(changes) == 1
    cfg.curriculum_freeze = False
    assert curriculum(env, slice(None))["stage"] == 2.0
    assert changes == [("voxel", 0.020, "media"), ("radius", 0.003, "media")]
    assert curriculum(env, slice(None))["stage"] == 2.0
    assert len(changes) == 2


@pytest.mark.parametrize("values", ({}, {"voxel_size": 0.0}, {"voxel_size": 0.02, "success_rate": float("nan")}))
def test_resolution_rejects_invalid_stage_before_simulation(values):
    cfg = FrankaPourResetDatasetEnvCfg(mpm_resolution_stages=(PourMPMResolutionStageCfg(**values),))
    with pytest.raises(ValueError, match="MPM resolution"):
        cfg.validate_config()
