# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import RecorderTermCfg
from isaaclab.utils import ConfigMixin


@dataclass
class LocomanipulationSDGOutputDataRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type | str = "{DIR}.locomanipulation_sdg_env:LocomanipulationSDGOutputDataRecorder"


@dataclass
class LocomanipulationSDGRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_locomanipulation_sdg_output_data = LocomanipulationSDGOutputDataRecorderCfg()


@dataclass
class LocomanipulationSDGTerminationsCfg(ConfigMixin):
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@dataclass
class LocomanipulationSDGEventCfg(ConfigMixin):
    """Configuration for events."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")


@dataclass
class LocomanipulationSDGEnvCfg(ManagerBasedRLEnvCfg):
    recorders: LocomanipulationSDGRecorderManagerCfg = LocomanipulationSDGRecorderManagerCfg()
    terminations: LocomanipulationSDGTerminationsCfg = LocomanipulationSDGTerminationsCfg()
    events: LocomanipulationSDGEventCfg = LocomanipulationSDGEventCfg()
