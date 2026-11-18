# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The stock camera PPO with gaze supervision; the actor and critic also see the gaze."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpoleCameraPPORunnerCfg

from ..agents import gaze_supervised


@configclass
class CartpoleFoviPPORunnerCfg(CartpoleCameraPPORunnerCfg):
    experiment_name = "cartpole_fovi"
    obs_groups = {"actor": ["policy", "gaze"], "critic": ["critic", "gaze"]}
    algorithm = gaze_supervised(CartpoleCameraPPORunnerCfg().algorithm)
