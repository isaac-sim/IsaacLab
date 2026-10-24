# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend selection for humanoid teleoperation tasks with task-owned preparation."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class TeleopPhysicsCfg(PresetCfg):
    """Keep Kit as the default and expose the supported standalone backends."""

    isaacsim_physx = PhysxCfg()
    ovphysx = OvPhysxCfg()
    newton_mjwarp = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = isaacsim_physx
