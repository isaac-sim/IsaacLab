# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp entry points for the core task families.

Mirrors the ``isaaclab_tasks.core`` layout. Manager-based tasks register
``*-Warp-v0`` ids that reuse the stable env cfgs (adapted to the warp runtime at
construction); direct tasks register their own warp env classes here. Task
definitions (cfgs) live in ``isaaclab_tasks`` to avoid duplication.
"""
