# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Built-in terms, the MDP2 counterparts of :mod:`isaaclab.envs.mdp`.

Terms keep the names of their MDP1 functions, and bind scene data through inputs instead of ``asset_cfg``::

    from isaaclab_experimental import mdp2
    from isaaclab_experimental.mdp2 import mdp

    mdp2.Observation(
        mdp.joint_pos_rel,
        inputs={
            "joint_pos": robot.joint_pos,
            "default_joint_pos": robot.default_joint_pos,
            "joint_ids": robot.joints(".*"),
        },
    )
"""

from isaaclab.utils.module import lazy_export

lazy_export()
