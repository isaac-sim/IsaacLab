# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from isaaclab_rl.skrl import (
    SkrlDeterministicModelCfg,
    SkrlExperimentCfg,
    SkrlGaussianModelCfg,
    SkrlModelsCfg,
    SkrlNetworkCfg,
    SkrlPpoAgentCfg,
    SkrlRunnerCfg,
    SkrlTrainerCfg,
    skrl_cfg_to_dict,
)


def _make_cfg() -> SkrlRunnerCfg:
    return SkrlRunnerCfg(
        models=SkrlModelsCfg(
            policy=SkrlGaussianModelCfg(network=[SkrlNetworkCfg(layers=[32, 32])]),
            value=SkrlDeterministicModelCfg(network=[SkrlNetworkCfg(layers=[32, 32])]),
        ),
        agent=SkrlPpoAgentCfg(
            rollouts=16,
            learning_epochs=8,
            mini_batches=4,
            learning_rate=3.0e-4,
            experiment=SkrlExperimentCfg(directory="cartpole"),
        ),
        trainer=SkrlTrainerCfg(timesteps=4800),
    )


def test_to_runner_dict_uses_skrl_schema() -> None:
    cfg = _make_cfg()

    runner_cfg = cfg.to_runner_dict()

    assert runner_cfg["models"]["policy"]["class"] == "GaussianMixin"
    assert runner_cfg["memory"]["class"] == "RandomMemory"
    assert runner_cfg["agent"]["class"] == "PPO"
    assert runner_cfg["agent"]["gae_lambda"] == 0.95
    assert "class_name" not in runner_cfg["agent"]


def test_config_instances_do_not_share_mutable_values() -> None:
    first = _make_cfg()
    second = _make_cfg()

    first.models.policy.network[0].layers.append(16)
    first.agent.learning_rate_scheduler_kwargs["kl_threshold"] = 0.02

    assert second.models.policy.network[0].layers == [32, 32]
    assert second.agent.learning_rate_scheduler_kwargs == {"kl_threshold": 0.008}


def test_skrl_cfg_to_dict_preserves_yaml_compatibility() -> None:
    cfg = {"seed": 42}

    assert skrl_cfg_to_dict(cfg) is cfg


def test_skrl_cfg_to_dict_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError, match="SkrlRunnerCfg or dict"):
        skrl_cfg_to_dict(object())
