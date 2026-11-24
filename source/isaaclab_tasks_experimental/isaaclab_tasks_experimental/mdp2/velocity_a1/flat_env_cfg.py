# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree A1 velocity tracking on flat ground."""

from isaaclab_experimental import mdp2

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from ..velocity_flat import (
    FLAT_VELOCITY_TASK,
    FlatVelocityEnvCfg,
    FlatVelocityPhysicsCfg,
    FlatVelocitySceneCfg,
    contact_forces,
    robot,
)

base = robot.bodies("trunk")
_ACTION_RATE = FLAT_VELOCITY_TASK.rewards["action_rate_l2"]
_BASE_CONTACT = FLAT_VELOCITY_TASK.terminations["base_contact"]
_BASE_MASS = FLAT_VELOCITY_TASK.events["add_base_mass"]
_BASE_COM = FLAT_VELOCITY_TASK.events["base_com"]

TASK = replace(
    FLAT_VELOCITY_TASK,
    rewards={**FLAT_VELOCITY_TASK.rewards, "action_rate_l2": replace(_ACTION_RATE, weight=-0.01)},
    terminations={
        **FLAT_VELOCITY_TASK.terminations,
        "base_contact": replace(
            _BASE_CONTACT, inputs={**_BASE_CONTACT.inputs, "body_ids": contact_forces.bodies("trunk")}
        ),
    },
    events={
        **FLAT_VELOCITY_TASK.events,
        "add_base_mass": replace(_BASE_MASS, inputs={**_BASE_MASS.inputs, "body_ids": base}),
        "base_com": replace(_BASE_COM, inputs={**_BASE_COM.inputs, "body_ids": base}),
    },
)
"""The MDP of the Unitree A1 task."""

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
    task: mdp2.TaskCfg = TASK
