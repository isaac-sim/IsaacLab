# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility configuration for absolute DiffIK Franka Reach."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.config.franka import franka_reach_env_cfg


@configclass
class FrankaReachEnvCfg(franka_reach_env_cfg.FrankaReachEnvCfg):
    """Compatibility configuration that selects the canonical absolute DiffIK preset."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.physics = self.sim.physics.isaacsim_physx
        self.scene.robot.spawn.rigid_props.disable_gravity = (
            self.scene.robot.spawn.rigid_props.disable_gravity.diffik_abs
        )
        self.actions.arm_action = self.actions.arm_action.diffik_abs
