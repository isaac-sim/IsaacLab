# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for collecting dynamics data with the normal environment lifecycle."""

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.utils import configclass


@configclass
class NeRDExplorationActionCfg(JointPositionActionCfg):
    """Joint targets and restart cadence of an exploration controller."""

    asset_name: str = "robot"
    joint_names: list[str] = [".*"]
    scale: float = 1.0
    use_default_offset: bool = False
    episode_frames: tuple[int, int] = (64, 128)
