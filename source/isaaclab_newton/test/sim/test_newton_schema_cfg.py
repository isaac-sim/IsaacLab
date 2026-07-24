# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests for Newton schema configuration contracts."""

from isaaclab_newton.sim.schemas import (
    MujocoJointDrivePropertiesCfg,
    MujocoRigidBodyPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonRigidBodyPropertiesCfg,
)


def test_mujoco_isinstance_newton():
    """Mujoco configuration instances inherit their Newton configuration parents."""
    mjc_rigid = MujocoRigidBodyPropertiesCfg(gravcomp=0.5)
    assert isinstance(mjc_rigid, NewtonRigidBodyPropertiesCfg)

    mjc_joint = MujocoJointDrivePropertiesCfg(actuatorgravcomp=True)
    assert isinstance(mjc_joint, NewtonJointDrivePropertiesCfg)
