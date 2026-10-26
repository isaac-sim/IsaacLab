# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entrypoints for teleoperation, demonstration recording and session replay.

Each ``run_*_cli`` function takes the command-line arguments without the program name.
``teleop_se3_agent.py``, ``record_demos.py`` and ``teleop_replay_agent.py`` under ``scripts/`` call them.
They need ``isaaclab_tasks`` to resolve the task.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
