# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Dict, TypeVar  # noqa: UP035

import gymnasium as gym
import torch

##
# Types.
##

SpaceType = TypeVar("SpaceType", gym.spaces.Space, int, set, tuple, list, dict)
"""A sentinel object to indicate a valid space type to specify states, observations and actions."""

VecEnvObs = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
"""Observation returned by the environment.

The observations are stored in a dictionary. The keys are the group to which the observations belong.
This is useful for various setups such as reinforcement learning with asymmetric actor-critic or
multi-agent learning. For non-learning paradigms, this may include observations for different components
of a system.

Within each group, the observations can be stored either as a dictionary with keys as the names of each
observation term in the group, or a single tensor obtained from concatenating all the observation terms.
For example, for asymmetric actor-critic, the observation for the actor and the critic can be accessed
using the keys ``"policy"`` and ``"critic"`` respectively.

Note:
    By default, most learning frameworks deal with default and privileged observations in different ways.
    This handling must be taken care of by the wrapper around the :class:`ManagerBasedRLEnv` instance.

    For included frameworks (RSL-RL, RL-Games, skrl), the observations must have the key "policy". In case,
    the key "critic" is also present, then the critic observations are taken from the "critic" group.
    Otherwise, they are the same as the "policy" group.

"""

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]
"""The environment signals processed at the end of each step.

The tuple contains batched information for each sub-environment. The information is stored in the following order:

1. **Observations**: The observations from the environment.
2. **Rewards**: The rewards from the environment.
3. **Terminated Dones**: Whether the environment reached a terminal state, such as task success or robot falling etc.
4. **Timeout Dones**: Whether the environment reached a timeout state, such as end of max episode length.
5. **Extras**: A dictionary containing additional information from the environment.
"""

AgentID = TypeVar("AgentID")
"""Unique identifier for an agent within a multi-agent environment.

The identifier has to be an immutable object, typically a string (e.g.: ``"agent_0"``).
"""

ObsType = TypeVar("ObsType", torch.Tensor, Dict[str, torch.Tensor])
"""A sentinel object to indicate the data type of the observation.
"""

ActionType = TypeVar("ActionType", torch.Tensor, Dict[str, torch.Tensor])
"""A sentinel object to indicate the data type of the action.
"""

StateType = TypeVar("StateType", torch.Tensor, dict)
"""A sentinel object to indicate the data type of the state.
"""

EnvStepReturn = tuple[
    Dict[AgentID, ObsType],
    Dict[AgentID, torch.Tensor],
    Dict[AgentID, torch.Tensor],
    Dict[AgentID, torch.Tensor],
    Dict[AgentID, dict],
]
"""The environment signals processed at the end of each step.

The tuple contains batched information for each sub-environment (keyed by the agent ID).
The information is stored in the following order:

1. **Observations**: The observations from the environment.
2. **Rewards**: The rewards from the environment.
3. **Terminated Dones**: Whether the environment reached a terminal state, such as task success or robot falling etc.
4. **Timeout Dones**: Whether the environment reached a timeout state, such as end of max episode length.
5. **Extras**: A dictionary containing additional information from the environment.
"""
