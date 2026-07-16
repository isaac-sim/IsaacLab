# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp MDP twins for the stable Cartpole task.

There is no separate task registration: run the stable ``Isaac-Cartpole`` task
with ``--frontend warp`` and ``presets=newton_mjwarp``.
"""

from isaaclab_experimental.envs.frontend import register_mdp_route

# Warp twins for the stable cartpole MDP terms live in this package's ``mdp``.
register_mdp_route("isaaclab_tasks.core.cartpole", f"{__name__}.mdp")
