# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.skrl import (
    SkrlDeterministicModelCfg,
    SkrlExperimentCfg,
    SkrlGaussianModelCfg,
    SkrlModelsCfg,
    SkrlNetworkCfg,
    SkrlPpoAgentCfg,
    SkrlRunnerCfg,
    SkrlTrainerCfg,
)


@configclass
class SkrlFlatPPORunnerCfg(SkrlRunnerCfg):
    models = SkrlModelsCfg(
        separate=False,
        policy=SkrlGaussianModelCfg(
            network=[
                SkrlNetworkCfg(layers=[128, 128, 128]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(layers=[128, 128, 128]),
            ],
        ),
    )
    agent = SkrlPpoAgentCfg(
        rollouts=24,
        learning_epochs=5,
        mini_batches=4,
        learning_rate=0.001,
        learning_rate_scheduler_kwargs={"kl_threshold": 0.01},
        entropy_loss_scale=0.01,
        value_loss_scale=1.0,
        experiment=SkrlExperimentCfg(
            directory="unitree_go2_flat",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=7200)


@configclass
class SkrlRoughPPORunnerCfg(SkrlRunnerCfg):
    models = SkrlModelsCfg(
        separate=False,
        policy=SkrlGaussianModelCfg(
            network=[
                SkrlNetworkCfg(layers=[512, 256, 128]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(layers=[512, 256, 128]),
            ],
        ),
    )
    agent = SkrlPpoAgentCfg(
        rollouts=24,
        learning_epochs=5,
        mini_batches=4,
        learning_rate=0.001,
        learning_rate_scheduler_kwargs={"kl_threshold": 0.01},
        entropy_loss_scale=0.01,
        value_loss_scale=1.0,
        experiment=SkrlExperimentCfg(
            directory="unitree_go2_rough",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=36000)
