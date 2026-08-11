# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the UR10 particle-push task configuration."""

import pytest

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
