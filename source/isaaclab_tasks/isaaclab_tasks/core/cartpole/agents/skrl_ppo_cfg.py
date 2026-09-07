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
class SkrlDirectCameraPPORunnerCfg(SkrlRunnerCfg):
    models = SkrlModelsCfg(
        separate=False,
        policy=SkrlGaussianModelCfg(
            network=[
                SkrlNetworkCfg(
                    name="features_extractor",
                    layers=[
                        {"conv2d": {"out_channels": 32, "kernel_size": 8, "stride": 4, "padding": 0}},
                        {"conv2d": {"out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 0}},
                        {"conv2d": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0}},
                        "flatten",
                    ],
                    activations="relu",
                ),
                SkrlNetworkCfg(input="features_extractor", layers=[512]),
            ],
        ),
        value=SkrlDeterministicModelCfg(
            network=[
                SkrlNetworkCfg(
                    name="features_extractor",
                    layers=[
                        {"conv2d": {"out_channels": 32, "kernel_size": 8, "stride": 4, "padding": 0}},
                        {"conv2d": {"out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 0}},
                        {"conv2d": {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0}},
                        "flatten",
                    ],
                    activations="relu",
                ),
                SkrlNetworkCfg(input="features_extractor", layers=[512]),
            ],
        ),
    )
    agent = SkrlPpoAgentCfg(
        rollouts=64,
        learning_epochs=4,
        mini_batches=32,
        learning_rate=0.0001,
        value_preprocessor="RunningStandardScaler",
        value_loss_scale=1.0,
        experiment=SkrlExperimentCfg(
            directory="cartpole_camera_direct",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=32000)


@configclass
class SkrlDirectPPORunnerCfg(SkrlRunnerCfg):
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
        rollouts=32,
        learning_epochs=8,
        mini_batches=8,
        learning_rate=0.0005,
        observation_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        rewards_shaper_scale=0.1,
        experiment=SkrlExperimentCfg(
            directory="cartpole_direct",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=4800)


@configclass
class SkrlManagerPPORunnerCfg(SkrlRunnerCfg):
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
        mini_batches=8,
        learning_rate=0.0003,
        experiment=SkrlExperimentCfg(
            directory="cartpole",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=2400)
