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
                SkrlNetworkCfg(layers=[512, 512, 256, 128]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(layers=[512, 512, 256, 128]),
            ],
        ),
    )
    agent = SkrlPpoAgentCfg(
        rollouts=16,
        learning_epochs=5,
        mini_batches=4,
        learning_rate=0.0005,
        learning_rate_scheduler_kwargs={"kl_threshold": 0.016},
        observation_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        rewards_shaper_scale=0.01,
        experiment=SkrlExperimentCfg(
            directory="shadow_hand",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=80000)
