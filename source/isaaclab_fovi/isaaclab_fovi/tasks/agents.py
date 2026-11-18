# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared agent configuration of the active-gaze camera tasks."""

from dataclasses import fields

from isaaclab.utils import replace

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from ..rsl_rl import GazeGaussianDistributionCfg, GazeSupervisedPPOCfg


def gaze_supervised(algorithm: RslRlPpoAlgorithmCfg, **kwargs) -> GazeSupervisedPPOCfg:
    """The PPO settings of ``algorithm`` with gaze supervision by the teacher."""
    values = {field.name: getattr(algorithm, field.name) for field in fields(algorithm) if field.name != "class_name"}
    return GazeSupervisedPPOCfg(**values, **kwargs)


def gaze_actor(actor: RslRlMLPModelCfg) -> RslRlMLPModelCfg:
    """The stock actor with a fixed, small noise on its gaze actions, the last two."""
    stock = actor.distribution_cfg
    head = GazeGaussianDistributionCfg(init_std=stock.init_std, std_type=stock.std_type, std_range=stock.std_range)
    return replace(actor, distribution_cfg=head)
