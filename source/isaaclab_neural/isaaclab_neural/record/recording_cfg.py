# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for collecting dynamics data with the normal environment lifecycle."""

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.managers import RecorderManagerBaseCfg, RecorderTermCfg
from isaaclab.utils import configclass


@configclass
class NeRDExplorationActionCfg(JointPositionActionCfg):
    """Joint targets and restart cadence of an exploration controller."""

    asset_name: str = "robot"
    joint_names: list[str] = [".*"]
    scale: float = 1.0
    use_default_offset: bool = False
    episode_frames: tuple[int, int] = (64, 128)


@configclass
class NeRDDynamicsRecorderCfg(RecorderTermCfg):
    """Pre-step state and post-step control/label capture through RecorderManager."""

    class_type: str = "{DIR}.recording:NeRDDynamicsRecorder"


@configclass
class NeRDRecorderCfg(RecorderManagerBaseCfg):
    """Native NeRD episode output through the existing recorder manager."""

    dataset_file_handler_class_type: type | str = "{DIR}.recording:NeRDDatasetFileHandler"
    export_in_close: bool = True
    dynamics = NeRDDynamicsRecorderCfg()
