# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp MDP route for the stable Ant task.

There is no separate task registration: run the stable ``Isaac-Ant`` task with
``--frontend warp`` and ``presets=newton_mjwarp``.
"""

from isaaclab_experimental.envs.frontend import register_mdp_route

# The stable Ant task borrows Humanoid MDP terms, so its warp twins live in the
# experimental humanoid package.
register_mdp_route(
    "isaaclab_tasks.core.locomotion.ant",
    "isaaclab_tasks_experimental.manager_based.classic.humanoid.mdp",
)
