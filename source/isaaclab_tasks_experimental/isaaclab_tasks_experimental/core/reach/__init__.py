# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp MDP twins for the stable Reach tasks.

There is no separate task registration: run the stable joint-position reach
tasks (``Isaac-Reach-Franka``, ``Isaac-Reach-UR10``) with ``--frontend warp``
and ``presets=newton_mjwarp``. The IK and OSC variants require controller
action twins that do not exist yet.
"""

from isaaclab_experimental.envs.frontend import register_mdp_route

# Warp twins for the stable reach MDP terms live in this package's ``mdp``.
register_mdp_route("isaaclab_tasks.core.reach", f"{__name__}.mdp")
