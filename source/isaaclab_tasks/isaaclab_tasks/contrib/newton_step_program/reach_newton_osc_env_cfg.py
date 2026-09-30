# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka reach driven by the Newton operational-space controller inside the Newton step program."""

from isaaclab_newton.controllers import NewtonOperationalSpaceControllerCfg
from isaaclab_newton.envs.mdp.actions import NewtonOperationalSpaceControllerActionCfg

from isaaclab.utils import configclass

from isaaclab_tasks.core.reach.config.franka import franka_reach_osc_env_cfg


@configclass
class FrankaReachNewtonOSCEnvCfg(franka_reach_osc_env_cfg.FrankaReachEnvCfg):
    """Franka reach with the Newton operational-space controller.

    On Newton physics the controller computes arm efforts before every physics step inside the captured step
    program, so the decimation loop stays folded into one physics call. Other backends run the same controller on
    the host every physics step. Gains match ``Isaac-Reach-Franka-OSC``'s default impedance with critical damping.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = NewtonOperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            target_type="pose_abs",
            null_space_joint_pos_target="center",
            controller=NewtonOperationalSpaceControllerCfg(
                motion_stiffness=100.0,
                motion_damping=20.0,
                use_inertia_decoupling=True,
                # The parent task compensates gravity through the solver.
                use_gravity_compensation=False,
                use_null_space_control=True,
                null_space_stiffness=10.0,
                null_space_damping=6.3,
            ),
        )
