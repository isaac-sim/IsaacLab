# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.skrl import (
    SkrlDeterministicModelCfg,
    SkrlExperimentCfg,
    SkrlGaussianModelCfg,
    SkrlIppoAgentCfg,
    SkrlMappoAgentCfg,
    SkrlModelsCfg,
    SkrlNetworkCfg,
    SkrlPpoAgentCfg,
    SkrlRunnerCfg,
    SkrlTrainerCfg,
)


@configclass
class SkrlMarlIPPORunnerCfg(SkrlRunnerCfg):
    models = SkrlModelsCfg(
        separate=False,
        policy=SkrlGaussianModelCfg(
            network=[
                SkrlNetworkCfg(layers=[32, 32]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(layers=[32, 32]),
            ],
        ),
    )
    agent = SkrlIppoAgentCfg(
        rollouts=16,
        learning_epochs=8,
        mini_batches=1,
        learning_rate=0.0003,
        observation_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        experiment=SkrlExperimentCfg(
            directory="pendulum_marl_direct",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=4800)


@configclass
class SkrlMarlMAPPORunnerCfg(SkrlRunnerCfg):
    models = SkrlModelsCfg(
        separate=True,
        policy=SkrlGaussianModelCfg(
            network=[
                SkrlNetworkCfg(layers=[32, 32]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(input="STATES", layers=[32, 32]),
            ],
        ),
    )
    agent = SkrlMappoAgentCfg(
        rollouts=16,
        learning_epochs=8,
        mini_batches=1,
        learning_rate=0.0003,
        observation_preprocessor="RunningStandardScaler",
        state_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        experiment=SkrlExperimentCfg(
            directory="pendulum_marl_direct",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=4800)


@configclass
class SkrlMarlPPORunnerCfg(SkrlRunnerCfg):
    models = SkrlModelsCfg(
        separate=False,
        policy=SkrlGaussianModelCfg(
            network=[
                SkrlNetworkCfg(layers=[32, 32]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(layers=[32, 32]),
            ],
        ),
    )
    agent = SkrlPpoAgentCfg(
        rollouts=16,
        learning_epochs=8,
        mini_batches=1,
        learning_rate=0.0003,
        observation_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        experiment=SkrlExperimentCfg(
            directory="pendulum_marl_direct",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=4800)
