# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp implementation of the stable direct Ant task.

There is no separate task registration: the stable ``Isaac-Ant-Direct``
registration declares :class:`~isaaclab_tasks_experimental.direct.ant.ant_env_warp.AntWarpEnv`
as its ``warp_entry_point``; run it with ``--frontend warp`` and
``presets=newton_mjwarp``.
"""
