# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_newton.sim.schemas import MujocoJointDrivePropertiesCfg

from isaaclab_tasks.contrib.reach.config.franka.ik_abs_env_cfg import FrankaReachEnvCfg as FrankaReachAbsEnvCfg
from isaaclab_tasks.contrib.reach.config.franka.ik_rel_env_cfg import FrankaReachEnvCfg as FrankaReachRelEnvCfg
from isaaclab_tasks.utils import resolve_presets


@pytest.mark.parametrize("env_cfg_type", [FrankaReachAbsEnvCfg, FrankaReachRelEnvCfg])
def test_newton_mjwarp_preset_enables_native_gravity_compensation(env_cfg_type):
    env_cfg = resolve_presets(env_cfg_type(), selected=("newton_mjwarp",))

    joint_drive_props = env_cfg.scene.robot.spawn.joint_drive_props
    assert isinstance(joint_drive_props, MujocoJointDrivePropertiesCfg)
    assert joint_drive_props.actuatorgravcomp is True


@pytest.mark.parametrize("env_cfg_type", [FrankaReachAbsEnvCfg, FrankaReachRelEnvCfg])
@pytest.mark.parametrize("selected", [(), ("physx",)])
def test_default_and_physx_presets_leave_native_gravity_compensation_unset(env_cfg_type, selected):
    env_cfg = resolve_presets(env_cfg_type(), selected=selected)

    assert env_cfg.scene.robot.spawn.joint_drive_props is None
