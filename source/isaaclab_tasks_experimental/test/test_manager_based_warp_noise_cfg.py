# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp-native noise selection in built-in manager-based tasks."""

from isaaclab_experimental.utils.noise import NoiseCfg
from isaaclab_tasks_experimental.manager_based.locomotion.velocity.velocity_env_cfg import (
    PolicyCfg as VelocityPolicyCfg,
)
from isaaclab_tasks_experimental.manager_based.manipulation.reach.reach_env_cfg import (
    ObservationsCfg as ReachObservationsCfg,
)

from isaaclab.utils.noise import UniformNoiseCfg as StableUniformNoiseCfg


def test_stable_uniform_noise_is_not_a_warp_noise_config():
    """Stable Torch noise should not satisfy the experimental manager's Warp type check."""
    assert not isinstance(StableUniformNoiseCfg(), NoiseCfg)


def test_builtin_manager_tasks_use_warp_uniform_noise():
    """Every configured velocity and Reach observation corruption should use Warp noise."""
    velocity_policy = VelocityPolicyCfg()
    reach_policy = ReachObservationsCfg.PolicyCfg()
    noisy_terms = [
        velocity_policy.base_lin_vel,
        velocity_policy.base_ang_vel,
        velocity_policy.projected_gravity,
        velocity_policy.joint_pos,
        velocity_policy.joint_vel,
        reach_policy.joint_pos,
        reach_policy.joint_vel,
    ]

    assert all(isinstance(term.noise, NoiseCfg) for term in noisy_terms)
