# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka pour task: grasp a bowl full of granular MPM media and pour it into a second bowl.

The submodules are intentionally import-light at the package level so pure-geometry helpers
(:mod:`cube_bowl_mesh`, :mod:`media_fill`) can be imported and unit-tested without launching the
simulator. The environment classes live in :mod:`pour_env` / :mod:`pour_env_cfg` and the gym
registration in :mod:`config.franka`.
"""
