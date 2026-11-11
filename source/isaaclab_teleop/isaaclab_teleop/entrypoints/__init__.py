# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entrypoints for teleoperation, demonstration recording and replay, and session replay.

Each ``run_*_cli`` function takes the command-line arguments without the program name; ``run_cli`` selects
the ``isaaclab teleop`` workflow. ``teleop_se3_agent.py``, ``record_demos.py``, ``replay_demos.py`` and
``teleop_replay_agent.py`` under ``scripts/`` call them. They need ``isaaclab_tasks`` to resolve the task.
"""

import warp as wp

# Teleoperation only runs inference. Warp fixes this option when each kernel module is imported, so set it
# here, before any entrypoint module imports one, whether through a script, ``isaaclab teleop`` or a direct call.
wp.config.enable_backward = False

from isaaclab.utils.module import lazy_export  # noqa: E402

lazy_export()
