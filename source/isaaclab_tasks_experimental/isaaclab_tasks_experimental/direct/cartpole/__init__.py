# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp implementation of the stable direct Cartpole task.

There is no separate task registration: the stable ``Isaac-Cartpole-Direct``
registration declares :class:`~isaaclab_tasks_experimental.direct.cartpole.cartpole_warp_env.CartpoleWarpEnv`
as its ``warp_entry_point``; run it with ``--frontend warp`` and
``presets=newton_mjwarp``.
"""
