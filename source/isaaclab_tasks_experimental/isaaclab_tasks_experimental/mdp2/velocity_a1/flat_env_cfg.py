# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree A1 velocity tracking on flat ground."""

from isaaclab_experimental import mdp2

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from ..velocity_flat import FlatVelocityEnvCfg, FlatVelocityPhysicsCfg, FlatVelocitySceneCfg, flat_velocity_task

_FLAT_DEFAULTS = FlatVelocityEnvCfg()
_FLAT_MJWARP = _FLAT_DEFAULTS.sim.physics.newton_mjwarp
_A1_MJWARP = replace(_FLAT_MJWARP, solver_cfg=replace(_FLAT_MJWARP.solver_cfg, njmax=60, nconmax=30))


@configclass
class UnitreeA1FlatEnvCfg(FlatVelocityEnvCfg):
    """Configuration of the Unitree A1 flat velocity environment."""

    sim: SimulationCfg = replace(
        _FLAT_DEFAULTS.sim, physics=FlatVelocityPhysicsCfg(newton_mjwarp=_A1_MJWARP, default=_A1_MJWARP)
    )
    scene: FlatVelocitySceneCfg = replace(
        _FLAT_DEFAULTS.scene, robot=replace(UNITREE_A1_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    )
    task: mdp2.TaskCfg = flat_velocity_task("trunk", action_rate_weight=-0.01)
