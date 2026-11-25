# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym task registrations for contributed environments.

Unlike :mod:`isaaclab_tasks`, this package is **not** walked and imported
automatically at startup -- nothing in the stock scripts imports
``isaaclab_contrib``. Each task subpackage therefore exposes a ``register()``
entry point suitable for the ``--external_callback`` argument that
``scripts/environments/teleoperation/teleop_se3_agent.py`` and friends accept,
which is called before ``gym.make`` precisely so out-of-tree environments can
register themselves without the script being modified.

Example:
    .. code-block:: bash

        ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
            --external_callback isaaclab_contrib.tasks.agibot_g2.register \
            --task Isaac-Teleop-G2-Ik7d-v0 --num_envs 1
"""
