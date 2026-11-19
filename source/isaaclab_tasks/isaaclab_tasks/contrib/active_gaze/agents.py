# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stock Cartpole camera PPO with gaze state in the actor and critic observations."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpoleCameraPPORunnerCfg


@configclass
class CartpoleCameraFoviPPORunnerCfg(CartpoleCameraPPORunnerCfg):
    """Train the camera policy and gaze actions with the stock PPO runner."""

    experiment_name = "cartpole_camera_fovi"
    obs_groups = {"actor": ["policy", "gaze"], "critic": ["critic", "gaze"]}
