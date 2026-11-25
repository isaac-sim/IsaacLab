# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of :class:`~isaaclab_tasks.contrib.active_gaze.gaze_supervised_ppo.GazeSupervisedPPO`."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class GazeSupervisedPPOCfg(RslRlPpoAlgorithmCfg):
    """PPO whose loss also pulls the policy's gaze toward a teacher's.

    See :class:`~isaaclab_tasks.contrib.active_gaze.gaze_supervised_ppo.GazeSupervisedPPO`.
    """

    class_name: str = "isaaclab_tasks.contrib.active_gaze.gaze_supervised_ppo:GazeSupervisedPPO"
    oracle_loss_coef: float = 1.0
    """Weight of the gaze loss. Zero trains plain PPO."""
    oracle_group: str = "oracle"
    """Observation group of the teacher's labels."""
