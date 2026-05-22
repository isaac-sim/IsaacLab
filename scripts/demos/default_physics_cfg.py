# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default physics backend presets for the demo scripts."""

from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils.hydra import PresetCfg


@configclass
class DefaultPhysicsCfgs(PresetCfg):
    """Default physics backend presets for the demo scripts."""

    default: PhysxCfg = PhysxCfg()
    physx: PhysxCfg = PhysxCfg()
    newton_mjwarp: NewtonCfg = NewtonCfg()
