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

    asset_name: str = "robot"
    """Name of the recorded articulation in the scene."""

    contact_bodies: str = ""
    """Body-name glob for the ground probes recorded as contact input; empty records none.

    The recording entrypoint sets it from the task's model definition.
    """

    prediction: str = "newton_neural.predictions:PredictionStateIncrement"
    """Native prediction class whose labels are recorded.

    The recording entrypoint sets it from the task's model definition.
    """

    shard_frames: int = 256
    """Frames buffered for all environments before a shard is written."""


@configclass
class NeRDRecorderCfg(RecorderManagerBaseCfg):
    """Recorder manager whose dynamics term writes native NeRD shards.

    The dataset is the folder :attr:`dataset_filename` in :attr:`dataset_export_dir_path`.
    """

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    """The dynamics term writes batched shards itself, so the manager exports no per-environment episodes."""

    dynamics = NeRDDynamicsRecorderCfg()
