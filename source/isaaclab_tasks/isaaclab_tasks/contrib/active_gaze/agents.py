# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agents of the active-gaze tasks: the stock PPO with gaze supervision and a fixed gaze noise."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpoleCameraPPORunnerCfg

from .gaze_distribution_cfg import GazeGaussianDistributionCfg
from .gaze_supervised_ppo_cfg import GazeSupervisedPPOCfg

_STOCK_RUNNER = CartpoleCameraPPORunnerCfg()
_STOCK_DISTRIBUTION = _STOCK_RUNNER.actor.distribution_cfg


@configclass
class CartpoleCameraFoviPPORunnerCfg(CartpoleCameraPPORunnerCfg):
    """The stock Cartpole camera PPO with gaze supervision; the actor and critic also see the gaze."""

    experiment_name = "cartpole_camera_fovi"
    obs_groups = {"actor": ["policy", "gaze"], "critic": ["critic", "gaze"]}
    actor = _STOCK_RUNNER.actor.replace(
        distribution_cfg=GazeGaussianDistributionCfg(
            init_std=_STOCK_DISTRIBUTION.init_std,
            std_type=_STOCK_DISTRIBUTION.std_type,
            std_range=_STOCK_DISTRIBUTION.std_range,
        )
    )
    algorithm = GazeSupervisedPPOCfg(
        **{name: value for name, value in vars(_STOCK_RUNNER.algorithm).items() if name != "class_name"}
    )
