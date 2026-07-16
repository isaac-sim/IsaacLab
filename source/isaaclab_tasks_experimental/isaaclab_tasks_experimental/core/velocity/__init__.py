# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp MDP twins for the stable velocity locomotion tasks.

There is no separate task registration: run the stable flat velocity ids
(e.g. ``Isaac-Velocity-Flat-AnymalD``) with ``--frontend warp`` and
``presets=newton_mjwarp``. Rough-terrain variants remain unsupported until
:class:`~isaaclab.terrains.TerrainImporter` gains Warp APIs (no
``height_scan`` twin).
"""
