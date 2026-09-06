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
class SkrlPPORunnerCfg(SkrlRunnerCfg):
    models = SkrlModelsCfg(
        separate=False,
        policy=SkrlGaussianModelCfg(
            network=[
                SkrlNetworkCfg(layers=[64, 64]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(layers=[64, 64]),
            ],
        ),
    )
    agent = SkrlPpoAgentCfg(
        rollouts=24,
        learning_epochs=5,
        mini_batches=4,
        learning_rate=0.001,
        learning_rate_scheduler_kwargs={"kl_threshold": 0.01},
        observation_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        entropy_loss_scale=0.01,
        value_loss_scale=1.0,
        experiment=SkrlExperimentCfg(
            directory="reach_ur10",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=24000)
