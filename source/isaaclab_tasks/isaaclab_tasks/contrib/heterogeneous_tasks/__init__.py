# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heterogeneous tasks for vision-language-action policies: a Franka with a Robotiq gripper manipulating objects, on
Newton.

The tasks share the environment configuration, objects and success terms of this package. ``tasks`` holds the tasks
and the environment whose environments each run one of them, switched on reset by
:class:`~isaaclab.managers.HeterogeneousTaskManager`; ``config/<robot>`` registers them for a robot.
"""

import os

VLA_BENCHMARKS_ASSETS_DIR = os.environ.get("VLA_BENCHMARKS_ASSETS_DIR", "")
"""Root of the tasks' asset package exported from the RoboLab scenes and the Omniverse libraries, set by the
environment variable of the same name. The tasks register only when it names an existing directory.

TODO: Default to the asset package on the Isaac Lab Nucleus once it is uploaded.
"""
