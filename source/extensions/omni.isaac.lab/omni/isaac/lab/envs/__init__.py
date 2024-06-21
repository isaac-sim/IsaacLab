# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for environment definitions.

Environments define the interface between the agent and the simulation.
In the simplest case, the environment provides the agent with the current
observations and executes the actions provided by the agent. However, the
environment can also provide additional information such as the current
reward, done flag, and information about the current episode.

Based on these, there are two types of task design patterns:

* **Manager-based workflow**: This workflow decomposes the environment into
  individual components (or managers) that handle different aspects of the
  environment (such as computing observations, applying actions, and applying
  randomization). The environment is responsible for mainly coordinating
  the managers and calling their functions.
* **Direct workflow**: This workflow provides a more direct interface to the
  task designing. The environment is implemented into a single class that directly
  handles all the necessary functionality without the need for additional
  managers.

For more information about the workflow design patterns, see the `Task Design Workflows`_ section.

.. _`Task Design Workflows`: https://isaac-sim.github.io/IsaacLab/source/features/workflows.html
"""

from . import mdp, ui
from .common import VecEnvObs, VecEnvStepReturn, ViewerCfg
from .direct_rl_env import DirectRLEnv
from .direct_rl_env_cfg import DirectRLEnvCfg
from .manager_based_env import ManagerBasedEnv
from .manager_based_env_cfg import ManagerBasedEnvCfg
from .manager_based_rl_env import ManagerBasedRLEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
