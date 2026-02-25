# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for environment managers.

The managers are used to handle various aspects of the environment such as randomization events, curriculum,
and observations. Each manager implements a specific functionality for the environment. The managers are
designed to be modular and can be easily extended to support new functionality.
"""

import lazy_loader as lazy

from .manager_term_cfg import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    DatasetExportMode,
    EventTermCfg,
    ManagerTermBaseCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RecorderManagerBaseCfg,
    RecorderTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from .scene_entity_cfg import SceneEntityCfg

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "action_manager": ["ActionManager", "ActionTerm"],
        "command_manager": ["CommandManager", "CommandTerm"],
        "curriculum_manager": ["CurriculumManager"],
        "event_manager": ["EventManager"],
        "manager_base": ["ManagerBase", "ManagerTermBase"],
        "observation_manager": ["ObservationManager"],
        "recorder_manager": ["RecorderManager", "RecorderTerm"],
        "reward_manager": ["RewardManager"],
        "termination_manager": ["TerminationManager"],
    },
)
__all__ += [
    "ActionTermCfg",
    "CommandTermCfg",
    "CurriculumTermCfg",
    "DatasetExportMode",
    "EventTermCfg",
    "ManagerTermBaseCfg",
    "ObservationGroupCfg",
    "ObservationTermCfg",
    "RecorderManagerBaseCfg",
    "RecorderTermCfg",
    "RewardTermCfg",
    "TerminationTermCfg",
    "SceneEntityCfg",
]
