# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations shared by the direct and manager-based Humanoid environments."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

JOINT_GEARS: dict[str, float] = {
    ".*_waist.*": 67.5,
    ".*_upper_arm.*": 67.5,
    "pelvis": 67.5,
    ".*_lower_arm": 45.0,
    ".*_thigh:0": 45.0,
    ".*_thigh:1": 135.0,
    ".*_thigh:2": 45.0,
    ".*_shin": 90.0,
    ".*_foot.*": 22.5,
}
"""Effort scale per joint [N·m], keyed by joint name expression."""

JOINT_EFFORT_LIMITS = {name: (-gear, gear) for name, gear in JOINT_GEARS.items()}
"""Effort clip per joint [N·m], i.e. the effort produced by a unit action."""

FEET_BODY_NAMES: list[str] = ["left_foot", "right_foot"]
"""Bodies whose incoming wrench is observed."""

WALK_TARGET_POS: tuple[float, float, float] = (1000.0, 0.0, 0.0)
"""Walk target [m] relative to the environment origin, far enough away that it is never reached."""


@configclass
class HumanoidPhysicsCfg(PresetCfg):
    """Physics backend presets for the Humanoid environments."""

    isaacsim_physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=80,
            nconmax=25,
            cone="pyramidal",
            update_data_interval=2,
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default: NewtonCfg = newton_mjwarp
