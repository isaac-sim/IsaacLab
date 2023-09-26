# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module introduces the managers for handling various aspects of the environment.

The managers are used to handle various aspects of the environment such as randomization, curriculum, and
observations. Each manager implements a specific functionality for the environment. The managers are
designed to be modular and can be easily extended to support new functionality.

Each manager is implemented as a class that inherits from the :class:`ManagerBase` class. Each manager
class should also have a corresponding configuration class that defines the configuration terms for the
manager. Each term should the :class:`ManagerBaseTermCfg` class or its subclass.

Example pseudo-code for a manager:

    .. code-block:: python

        from omni.isaac.orbit.utils import configclass
        from omni.isaac.orbit.utils.mdp import ManagerBase, ManagerBaseTermCfg

        @configclass
        class MyManagerCfg:

            my_term_1: ManagerBaseTermCfg = ManagerBaseTermCfg(...)
            my_term_2: ManagerBaseTermCfg = ManagerBaseTermCfg(...)
            my_term_3: ManagerBaseTermCfg = ManagerBaseTermCfg(...)

        # define manager instance
        my_manager = ManagerBase(cfg=ManagerCfg(), env=env)

"""

from __future__ import annotations

from .action_manager import ActionManager, ActionTerm
from .curriculum_manager import CurriculumManager
from .manager_base import ManagerBase
from .manager_cfg import (
    ActionTermCfg,
    CurriculumTermCfg,
    ManagerBaseTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RandomizationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from .observation_manager import ObservationManager
from .randomization_manager import RandomizationManager
from .reward_manager import RewardManager
from .termination_manager import TerminationManager

__all__ = [
    # base
    "SceneEntityCfg",
    "ManagerBaseTermCfg",
    "ManagerBase",
    # action
    "ActionTermCfg",
    "ActionTerm",
    "ActionManager",
    # curriculum
    "CurriculumTermCfg",
    "CurriculumManager",
    # observation
    "ObservationGroupCfg",
    "ObservationTermCfg",
    "ObservationManager",
    # reward
    "RewardTermCfg",
    "RewardManager",
    # randomization
    "RandomizationTermCfg",
    "RandomizationManager",
    # termination
    "TerminationTermCfg",
    "TerminationManager",
]
