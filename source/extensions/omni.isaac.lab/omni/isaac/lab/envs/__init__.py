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

Based on these, there are two types of environments:

* :class:`ManagerBasedEnv`: The manager-based workflow base environment which
  only provides the agent with the
  current observations and executes the actions provided by the agent.
* :class:`ManagerBasedRLEnv`: The manager-based workflow RL task environment which
  besides the functionality of
  the base environment also provides additional Markov Decision Process (MDP)
  related information such as the current reward, done flag, and information.

In addition, RL task environments can use the direct workflow implementation:

* :class:`DirectRLEnv`: The direct workflow RL task environment which provides implementations
  for implementing scene setup, computing dones, performing resets, and computing
  reward and observation.

"""

from . import mdp, ui
from .base_env_cfg import ManagerBasedEnvCfg, ViewerCfg
from .direct_rl_env import DirectRLEnv
from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env import ManagerBasedRLEnv
from .rl_env_cfg import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from .types import VecEnvObs, VecEnvStepReturn
