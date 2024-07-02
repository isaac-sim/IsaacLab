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

There are two types of environment designing workflows:

* **Manager-based**: The environment is decomposed into individual components (or managers)
  for different aspects (such as computing observations, applying actions, and applying
  randomization. The users mainly configure the managers and the environment coordinates the
  managers and calls their functions.
* **Direct**: The user implements all the necessary functionality directly into a single class
  directly without the need for additional managers.

Based on these workflows, there are the following environment classes:

* :class:`ManagerBasedEnv`: The manager-based workflow base environment which only provides the
  agent with the current observations and executes the actions provided by the agent.
* :class:`ManagerBasedRLEnv`: The manager-based workflow RL task environment which besides the
  functionality of the base environment also provides additional Markov Decision Process (MDP)
  related information such as the current reward, done flag, and information.
* :class:`DirectRLEnv`: The direct workflow RL task environment which provides implementations for
  implementing scene setup, computing dones, performing resets, and computing reward and observation.

For more information about the workflow design patterns, see the `Task Design Workflows`_ section.

.. _`Task Design Workflows`: https://isaac-sim.github.io/IsaacLab/source/features/task_workflows.html
"""

from . import mdp, ui
from .common import VecEnvObs, VecEnvStepReturn, ViewerCfg
from .direct_rl_env import DirectRLEnv
from .direct_rl_env_cfg import DirectRLEnvCfg
from .manager_based_env import ManagerBasedEnv
from .manager_based_env_cfg import ManagerBasedEnvCfg
from .manager_based_rl_env import ManagerBasedRLEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
