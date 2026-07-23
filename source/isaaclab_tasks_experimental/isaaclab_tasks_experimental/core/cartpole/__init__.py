# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp implementations for the stable Cartpole tasks.

Holds the MDP twins for the manager-based ``Isaac-Cartpole`` task and the
direct :class:`~isaaclab_tasks_experimental.core.cartpole.cartpole_warp_env.CartpoleWarpEnv`
declared by ``Isaac-Cartpole-Direct``. There is no separate task registration:
run the stable ids with ``--frontend warp`` and ``presets=newton_mjwarp``.
"""
