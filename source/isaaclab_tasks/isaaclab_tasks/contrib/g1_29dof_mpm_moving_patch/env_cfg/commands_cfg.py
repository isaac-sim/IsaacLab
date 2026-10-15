# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .. import mdp as g1_mdp


@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = g1_mdp.UniformVelocityYawCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=g1_mdp.UniformVelocityYawCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )
