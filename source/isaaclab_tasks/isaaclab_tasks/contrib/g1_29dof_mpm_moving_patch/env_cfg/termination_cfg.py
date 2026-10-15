# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .. import mdp as g1_mdp
from .. import mdp as mpm_mdp
from .. import mdp as vel_mdp


@configclass
class G1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_too_low = TerminationTermCfg(
        func=g1_mdp.root_height_below_minimum_adaptive,
        params={
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.pi / 4})
    terrain_out_of_bounds = TerminationTermCfg(
        func=mpm_mdp.root_outside_workspace,
        params={"margin": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )

    extreme_action = TerminationTermCfg(
        func=vel_mdp.ExtremeJointPositionAction,  # type: ignore
        params={
            "action_name": "joint_pos",
            "torque_fraction": 5.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # a diverged granular solve leaves the world NaN, where every thresholded term stays false
    solver_diverged = TerminationTermCfg(
        func=mpm_mdp.root_state_infinite,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
