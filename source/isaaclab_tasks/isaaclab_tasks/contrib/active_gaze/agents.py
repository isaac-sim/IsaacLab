# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agents of the active-gaze tasks: the stock PPO with gaze supervision and a fixed gaze noise."""

from dataclasses import fields

from isaaclab.utils import configclass, replace

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpoleCameraPPORunnerCfg

from .gaze_distribution_cfg import GazeGaussianDistributionCfg
from .gaze_supervised_ppo_cfg import GazeSupervisedPPOCfg


def gaze_supervised(algorithm: RslRlPpoAlgorithmCfg, **kwargs) -> GazeSupervisedPPOCfg:
    """The PPO settings of ``algorithm`` with gaze supervision by the teacher."""
    values = {field.name: getattr(algorithm, field.name) for field in fields(algorithm) if field.name != "class_name"}
    return GazeSupervisedPPOCfg(**values, **kwargs)


def gaze_actor(actor: RslRlMLPModelCfg) -> RslRlMLPModelCfg:
    """The stock actor with a fixed, small noise on its gaze actions, the last two."""
    stock = actor.distribution_cfg
    head = GazeGaussianDistributionCfg(init_std=stock.init_std, std_type=stock.std_type, std_range=stock.std_range)
    return replace(actor, distribution_cfg=head)


@configclass
class CartpoleCameraFoviPPORunnerCfg(CartpoleCameraPPORunnerCfg):
    """The stock Cartpole camera PPO with gaze supervision; the actor and critic also see the gaze."""

    experiment_name = "cartpole_camera_fovi"
    obs_groups = {"actor": ["policy", "gaze"], "critic": ["critic", "gaze"]}
    actor = gaze_actor(CartpoleCameraPPORunnerCfg().actor)
    algorithm = gaze_supervised(CartpoleCameraPPORunnerCfg().algorithm)
