# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp implementation of the stable direct Ant task.

The manager-based MDP twins live in the shared
:mod:`isaaclab_tasks_experimental.core.locomotion.mdp` package (mirroring the
stable layout). ``Isaac-Ant-Direct`` resolves
:class:`~isaaclab_tasks_experimental.core.locomotion.ant.ant_warp_env.AntWarpEnv`
by name from the stable env class (``AntEnv`` → ``AntWarpEnv`` on the mirrored
package); run the stable ids with ``--frontend warp`` and ``presets=newton_mjwarp``.
"""
