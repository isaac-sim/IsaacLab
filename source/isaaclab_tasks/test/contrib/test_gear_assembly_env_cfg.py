# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for GearAssembly environment configuration defaults."""

import pytest

from isaaclab_tasks.contrib.deploy.gear_assembly.config.rizon_4s.ros_inference_env_cfg import (
    Rizon4sGearAssemblyROSInferenceEnvCfg,
)
from isaaclab_tasks.contrib.deploy.gear_assembly.config.ur_10e.joint_pos_env_cfg import UR10e2F140GearAssemblyEnvCfg


@pytest.mark.parametrize(
    "env_cfg_cls",
    # Every class inherits the base scene default; Rizon 4s ROS inference is the only one that writes num_envs.
    (UR10e2F140GearAssemblyEnvCfg, Rizon4sGearAssemblyROSInferenceEnvCfg),
)
def test_gear_assembly_defaults_limit_parallel_environments(env_cfg_cls):
    """GearAssembly defaults must fit recurrent PPO training on development GPUs."""
    assert env_cfg_cls().scene.num_envs == 1024
