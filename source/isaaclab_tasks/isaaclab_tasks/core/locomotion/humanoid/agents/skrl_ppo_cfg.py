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
                SkrlNetworkCfg(layers=[400, 200, 100]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(layers=[400, 200, 100]),
            ],
        ),
    )
    agent = SkrlPpoAgentCfg(
        rollouts=32,
        learning_epochs=5,
        mini_batches=4,
        learning_rate=0.0005,
        learning_rate_scheduler_kwargs={"kl_threshold": 0.01},
        observation_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        experiment=SkrlExperimentCfg(
            directory="humanoid_direct",
            write_interval=32,
            checkpoint_interval=3200,
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=32000)


@configclass
class SkrlManagerPPORunnerCfg(SkrlRunnerCfg):
    """Configuration used for the reinforcement-learning documentation benchmark.

    Update the other RL-library configurations with this one to keep the Training Performance table synchronized.
    """

    models = SkrlModelsCfg(
        separate=False,
        policy=SkrlGaussianModelCfg(
            network=[
                SkrlNetworkCfg(layers=[400, 200, 100]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(layers=[400, 200, 100]),
            ],
        ),
    )
    agent = SkrlPpoAgentCfg(
        rollouts=32,
        learning_epochs=5,
        mini_batches=4,
        learning_rate=0.0005,
        learning_rate_scheduler_kwargs={"kl_threshold": 0.01},
        observation_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        experiment=SkrlExperimentCfg(
            directory="humanoid",
            write_interval=32,
            checkpoint_interval=3200,
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=32000)
