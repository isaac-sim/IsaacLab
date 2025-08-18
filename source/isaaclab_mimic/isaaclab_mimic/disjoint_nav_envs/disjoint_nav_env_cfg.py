# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from .disjoint_nav_env import DisjointNavOutputDataRecorder


@configclass
class DisjointNavOutputDataRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = DisjointNavOutputDataRecorder


@configclass
class DisjointNavRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_disjoint_nav_output_data = DisjointNavOutputDataRecorderCfg()


@configclass
class DisjointNavTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class DisjointNavEventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")


@configclass
class DisjointNavEnvCfg(ManagerBasedRLEnvCfg):
    recorders: DisjointNavRecorderManagerCfg = DisjointNavRecorderManagerCfg()
    terminations: DisjointNavTerminationsCfg = DisjointNavTerminationsCfg()
    events: DisjointNavEventCfg = DisjointNavEventCfg()
