# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source-compatible normalizer cadence through the real PPO rollout/update boundary."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict

from isaaclab.utils import to_dict

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckSwizzlePPORunnerCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_next_observations_update_once_per_step_and_never_again_after_ppo():
    torch.manual_seed(42)
    obs = TensorDict({"policy": torch.zeros(4, 61), "critic": torch.zeros(4, 78)}, batch_size=[4])
    cfg = to_dict(MicroDuckSwizzlePPORunnerCfg())
    cfg["multi_gpu"] = None
    cfg["num_steps_per_env"] = 2
    cfg["algorithm"]["class_name"] = (
        "isaaclab_tasks.contrib.microduck.velocity.agents.rollout_normalization:MicroDuckRolloutNormalizationPPO"
    )
    for role in ["actor", "critic"]:
        cfg[role]["class_name"] = "isaaclab_tasks.contrib.microduck.velstand.agents.model:MicroDuckVelStandMLPModel"
        cfg[role]["hidden_dims"] = [8]
    cfg["algorithm"]["num_mini_batches"] = 1
    cfg["algorithm"]["num_learning_epochs"] = 1
    alg = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=4, num_actions=14), cfg, "cpu")
    reference_actor, reference_critic = EmpiricalNormalization(61), EmpiricalNormalization(78)
    with torch.inference_mode():
        for step in range(2):
            alg.act(obs)
            next_obs = TensorDict(
                {"policy": torch.randn(4, 61) + step, "critic": torch.randn(4, 78) - step}, batch_size=[4]
            )
            alg.process_env_step(next_obs, torch.zeros(4), torch.zeros(4), {})
            reference_actor.update(next_obs["policy"])
            reference_critic.update(next_obs["critic"])
            for normalizer, expected in [
                (alg.actor.obs_normalizer, reference_actor),
                (alg.critic.obs_normalizer, reference_critic),
            ]:
                assert normalizer.count == 4 * (step + 1)
                assert torch.equal(normalizer.mean, expected.mean)
                assert torch.equal(normalizer.std, expected.std)
            obs = next_obs
        alg.compute_returns(obs)
    alg.update()
    assert alg.actor.obs_normalizer.count == alg.critic.obs_normalizer.count == 8
