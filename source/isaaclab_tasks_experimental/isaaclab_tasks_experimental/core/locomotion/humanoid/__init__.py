# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp implementation of the stable direct Humanoid task.

The manager-based MDP twins live in the shared
:mod:`isaaclab_tasks_experimental.core.locomotion.mdp` package (mirroring the
stable layout). ``Isaac-Humanoid-Direct`` resolves
:class:`~isaaclab_tasks_experimental.core.locomotion.humanoid.humanoid_warp_env.HumanoidWarpEnv`
by name from the stable env class (``HumanoidEnv`` → ``HumanoidWarpEnv`` on the
mirrored package); run the stable ids with ``--frontend warp`` and ``presets=newton_mjwarp``.
"""
