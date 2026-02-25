# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the locomotion environments."""

import lazy_loader as lazy

_lazy_getattr, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "curriculums": ["terrain_levels_vel"],
        "rewards": [
            "feet_air_time",
            "feet_air_time_positive_biped",
            "feet_slide",
            "track_lin_vel_xy_yaw_frame_exp",
            "track_ang_vel_z_world_exp",
            "stand_still_joint_deviation_l1",
        ],
        "terminations": ["terrain_out_of_bounds"],
    },
)


def __getattr__(name):
    try:
        return _lazy_getattr(name)
    except AttributeError:
        pass
    import isaaclab.envs.mdp as _parent_mdp

    return getattr(_parent_mdp, name)
