# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .flat_env_cfg import G1FlatEnvCfg
from .rough_29dof_env_cfg import MINIMUM_PELVIS_HEIGHT, retarget_g1_rewards_to_29dof

##
# Pre-defined configs
##
from isaaclab_assets import G1_29DOF_VELOCITY_CFG  # isort: skip


@configclass
class G129DofFlatEnvCfg(G1FlatEnvCfg):
    """Flat-terrain velocity tracking for the Unitree G1 on the current 29-DoF asset."""

    def __post_init__(self):
        super().__post_init__()

        # physics
        # ``G1FlatEnvCfg`` sizes these for ``g1_minimal.usd``, whose whole collision set is three
        # boxes -- one per foot plus the torso. This robot keeps four spheres per foot and the
        # hands, so a standing robot alone exceeds ``nconmax = 10``; the observation goes NaN
        # within a few hundred iterations rather than failing at startup.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 200
        newton_mjwarp.solver_cfg.nconmax = 70
        self.sim.physics.default = newton_mjwarp
        # scene
        self.scene.robot = G1_29DOF_VELOCITY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # rewards
        retarget_g1_rewards_to_29dof(self)
        # terminations -- see MINIMUM_PELVIS_HEIGHT. World-frame is correct here; the ground is a
        # plane, so there is no terrain to measure against.
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": MINIMUM_PELVIS_HEIGHT, "asset_cfg": SceneEntityCfg("robot")},
        )
