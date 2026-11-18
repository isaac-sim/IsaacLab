# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared agent configuration of the active-gaze camera tasks."""

from dataclasses import fields

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from isaaclab_fovi.rsl_rl import GazeSupervisedPPOCfg


def gaze_supervised(algorithm: RslRlPpoAlgorithmCfg, **kwargs) -> GazeSupervisedPPOCfg:
    """The PPO settings of ``algorithm`` with gaze supervision by the teacher."""
    values = {field.name: getattr(algorithm, field.name) for field in fields(algorithm) if field.name != "class_name"}
    return GazeSupervisedPPOCfg(**values, **kwargs)
