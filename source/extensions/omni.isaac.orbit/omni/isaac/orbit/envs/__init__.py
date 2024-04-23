# Copyright (c) 2022-2024, The ORBIT Project Developers.
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

* :class:`BaseEnv`: The base environment which only provides the agent with the
  current observations and executes the actions provided by the agent.
* :class:`RLTaskEnv`: The RL task environment which besides the functionality of
  the base environment also provides additional Markov Decision Process (MDP)
  related information such as the current reward, done flag, and information.

"""

from . import mdp, ui
from .base_env import BaseEnv, VecEnvObs
from .base_env_cfg import BaseEnvCfg, ViewerCfg
from .rl_task_env import RLTaskEnv, VecEnvStepReturn
from .rl_task_env_cfg import RLTaskEnvCfg
