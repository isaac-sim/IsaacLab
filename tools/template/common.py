# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

# paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS_DIR = os.path.join(ROOT_DIR, "source", "isaaclab_tasks", "isaaclab_tasks")
TEMPLATE_DIR = os.path.join(ROOT_DIR, "tools", "template", "templates")

# RL algorithms
SINGLE_AGENT_ALGORITHMS = ["AMP", "PPO"]
MULTI_AGENT_ALGORITHMS = ["IPPO", "MAPPO"]
