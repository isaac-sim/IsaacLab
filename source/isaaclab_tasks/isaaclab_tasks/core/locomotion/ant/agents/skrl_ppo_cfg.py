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
class SkrlDirectPPORunnerCfg(SkrlRunnerCfg):
    models = SkrlModelsCfg(
        separate=False,
        policy=SkrlGaussianModelCfg(
            network=[
                SkrlNetworkCfg(layers=[256, 128, 64]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(layers=[256, 128, 64]),
            ],
        ),
    )
    agent = SkrlPpoAgentCfg(
        rollouts=16,
        learning_epochs=4,
        mini_batches=2,
        learning_rate=0.0003,
        observation_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        value_loss_scale=1.0,
        rewards_shaper_scale=0.6,
        experiment=SkrlExperimentCfg(
            directory="ant_direct",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=8000)


@configclass
class SkrlManagerPPORunnerCfg(SkrlRunnerCfg):
    models = SkrlModelsCfg(
        separate=False,
        policy=SkrlGaussianModelCfg(
            network=[
                SkrlNetworkCfg(layers=[256, 128, 64]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(layers=[256, 128, 64]),
            ],
        ),
    )
    agent = SkrlPpoAgentCfg(
        rollouts=16,
        learning_epochs=4,
        mini_batches=2,
        learning_rate=0.0003,
        observation_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        value_loss_scale=1.0,
        rewards_shaper_scale=0.6,
        experiment=SkrlExperimentCfg(
            directory="ant",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=8000)
