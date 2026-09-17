# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import RecorderTermCfg


@dataclass
class LocomanipulationSDGOutputDataRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type | str = (
        "isaaclab_mimic.locomanipulation_sdg.envs.locomanipulation_sdg_env:LocomanipulationSDGOutputDataRecorder"
    )


@dataclass
class LocomanipulationSDGRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_locomanipulation_sdg_output_data: Any = field(
        default_factory=LocomanipulationSDGOutputDataRecorderCfg
    )


@dataclass
class LocomanipulationSDGTerminationsCfg:
    """Termination terms for the MDP."""

    time_out: Any = field(default_factory=lambda: DoneTerm(func=base_mdp.time_out, time_out=True))


@dataclass
class LocomanipulationSDGEventCfg:
    """Configuration for events."""

    reset_all: Any = field(default_factory=lambda: EventTerm(func=base_mdp.reset_scene_to_default, mode="reset"))


@dataclass
class LocomanipulationSDGEnvCfg(ManagerBasedRLEnvCfg):
    recorders: LocomanipulationSDGRecorderManagerCfg = field(default_factory=LocomanipulationSDGRecorderManagerCfg)
    terminations: LocomanipulationSDGTerminationsCfg = field(default_factory=LocomanipulationSDGTerminationsCfg)
    events: LocomanipulationSDGEventCfg = field(default_factory=LocomanipulationSDGEventCfg)
