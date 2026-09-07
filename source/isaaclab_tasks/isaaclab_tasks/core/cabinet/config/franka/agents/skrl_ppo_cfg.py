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
        rollouts=96,
        learning_epochs=5,
        mini_batches=96,
        learning_rate=0.0005,
        entropy_loss_scale=0.001,
        experiment=SkrlExperimentCfg(
            directory="franka_open_drawer_direct",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=38400)


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
        rollouts=96,
        learning_epochs=5,
        mini_batches=96,
        learning_rate=0.0005,
        entropy_loss_scale=0.001,
        experiment=SkrlExperimentCfg(
            directory="franka_open_drawer",
        ),
    )
    trainer = SkrlTrainerCfg(timesteps=38400)
