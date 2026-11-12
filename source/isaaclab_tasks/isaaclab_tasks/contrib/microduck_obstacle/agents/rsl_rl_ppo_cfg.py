# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO settings for ToF navigation with a frozen walking controller."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlSymmetryCfg

from isaaclab_tasks.contrib.microduck.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg
from isaaclab_tasks.utils import preset


@configclass
class MicroDuckBrakingBetaDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    """Navigation Beta support; opt in to forward residuals that only brake."""

    class_name = (
        "isaaclab_tasks.contrib.microduck_obstacle.agents.navigation_distribution:MicroDuckBrakingBetaDistribution"
    )
    forward_braking_only: bool = False


@configclass
class MicroDuckObstacleAvoidancePPORunnerCfg(MicroDuckPPORunnerCfg):
    """Learn three velocity commands from ToF history while the walking actor stays frozen."""

    experiment_name = preset(
        default="microduck_obstacle_avoidance_tof_flat", backlash="microduck_obstacle_avoidance_tof_flat_backlash"
    )

    obs_groups = {"actor": ["policy", "tof"], "critic": ["critic", "tof"]}
    clip_actions = 1.0
    max_iterations = 1500
    save_interval = 100
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="tanh",
        obs_normalization=True,
        distribution_cfg=MicroDuckBrakingBetaDistributionCfg(),
    )

    # Exploration stays inside the executed action range; do not reward raw Gaussian spread.
    algorithm = MicroDuckPPORunnerCfg().algorithm.replace(
        entropy_coef=0.0,
        learning_rate=3e-4,
        schedule="fixed",
        desired_kl=None,
        gamma=0.998,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
            data_augmentation_func="isaaclab_tasks.contrib.microduck_obstacle.mdp.symmetry:compute_symmetric_states",
        ),
    )
