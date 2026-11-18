# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL training with gaze supervision by a teacher.

The algorithm is resolved by name from :attr:`GazeSupervisedPPOCfg.class_name`, so importing this package does not
import RSL-RL.
"""

from .gaze_supervised_ppo_cfg import GazeSupervisedPPOCfg

__all__ = ["GazeSupervisedPPOCfg"]
