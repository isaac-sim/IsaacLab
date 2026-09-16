# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import RecorderTermCfg
from isaaclab.utils import config_field


@dataclass
class LocomanipulationSDGOutputDataRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type | str = config_field("{DIR}.locomanipulation_sdg_env:LocomanipulationSDGOutputDataRecorder")


@dataclass
class LocomanipulationSDGRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_locomanipulation_sdg_output_data: Any = config_field(LocomanipulationSDGOutputDataRecorderCfg())


@dataclass
class LocomanipulationSDGTerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = config_field(DoneTerm(func=base_mdp.time_out, time_out=True))


@dataclass
class LocomanipulationSDGEventCfg:
    """Configuration for events."""

    reset_all: Any = config_field(EventTerm(func=base_mdp.reset_scene_to_default, mode="reset"))


@dataclass
class LocomanipulationSDGEnvCfg(ManagerBasedRLEnvCfg):
    recorders: LocomanipulationSDGRecorderManagerCfg = config_field(LocomanipulationSDGRecorderManagerCfg())
    terminations: LocomanipulationSDGTerminationsCfg = config_field(LocomanipulationSDGTerminationsCfg())
    events: LocomanipulationSDGEventCfg = config_field(LocomanipulationSDGEventCfg())
