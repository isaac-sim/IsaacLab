# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for collecting dynamics data with the normal environment lifecycle."""

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTermCfg
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
    """Batched capture of pre-step state and post-step controls and labels into native shards."""

    class_type: str = "{DIR}.recording:NeRDDynamicsRecorder"

    shard_frames: int = 256
    """Frames buffered for all environments before a shard is written."""


@configclass
class NeRDRecorderCfg(RecorderManagerBaseCfg):
    """Recorder manager whose dynamics term writes native NeRD shards."""

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    """The dynamics term writes batched shards itself, so the manager exports no per-environment episodes."""

    dynamics = NeRDDynamicsRecorderCfg()
