# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity locomotion experimental task registrations (manager-based)."""

from isaaclab_experimental.envs.frontend import register_mdp_route

# Warp twins for the stable velocity MDP terms live in this package's ``mdp``.
register_mdp_route("isaaclab_tasks.core.velocity", f"{__name__}.mdp")
