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

# TODO: Point to the Isaac Lab Nucleus once the assets are uploaded there.
VLA_BENCHMARKS_ASSETS_DIR = os.environ.get(
    "VLA_BENCHMARKS_ASSETS_DIR", "https://huggingface.co/datasets/LiFanxing/HeteroTask/resolve/main"
)
"""Root of the tasks' assets: the objects from the RoboLab and Omniverse libraries as complete USD files, and the
robot table and dome light texture. Defaults to the HeteroTask dataset on Hugging Face; the environment variable of the
same name overrides it, e.g. with a local copy. The robot is
:obj:`~isaaclab_assets.robots.franka.FRANKA_ROBOTIQ_USD_PATH`.
"""
