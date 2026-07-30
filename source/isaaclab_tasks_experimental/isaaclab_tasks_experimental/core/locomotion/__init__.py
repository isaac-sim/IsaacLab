# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp implementations for the stable locomotion tasks.

The ``mdp`` package holds the twins for the shared
:mod:`isaaclab_tasks.core.locomotion.mdp` terms used by the Humanoid and Ant
tasks. There is no separate task registration: run the stable ids with
``--frontend warp`` and ``presets=newton_mjwarp``.
"""
