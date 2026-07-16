# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp implementations for the stable Humanoid tasks.

Holds the MDP twins for the manager-based ``Isaac-Humanoid`` task (also
serving the Ant task, which borrows them) and the direct
:class:`~isaaclab_tasks_experimental.core.locomotion.humanoid.humanoid_warp_env.HumanoidWarpEnv`
declared by ``Isaac-Humanoid-Direct``. There is no separate task registration:
run the stable ids with ``--frontend warp`` and ``presets=newton_mjwarp``.
"""

from isaaclab_experimental.envs.frontend import register_mdp_route

# Warp twins for the stable humanoid MDP terms live in this package's ``mdp``.
register_mdp_route("isaaclab_tasks.core.locomotion.humanoid", f"{__name__}.mdp")
