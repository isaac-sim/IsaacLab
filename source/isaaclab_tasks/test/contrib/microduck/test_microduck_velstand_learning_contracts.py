# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral contracts for the VelStand two-teacher learning adapter."""

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tasks.contrib.microduck.velstand.agents.artifacts import verify_artifact
from isaaclab_tasks.contrib.microduck.velstand.agents.distill import (
    body_frame_masks,
    expert_input,
    fallen_mask_from_obs,
    remap_anchor_yaw,
)
from isaaclab_tasks.contrib.microduck.velstand.agents.model import MicroDuckVelStandMLPModel
from isaaclab_tasks.contrib.microduck.velstand.agents.ppo import MicroDuckVelStandPPO

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def observations(n=8):
    policy = torch.zeros(n, 61)
    policy[:, 5] = -1
    return TensorDict({"policy": policy, "critic": torch.zeros(n, 76)}, batch_size=[n])


def model(obs, group="actor"):
    return MicroDuckVelStandMLPModel(
        obs,
        {"actor": ["policy"], "critic": ["critic"]},
        group,
        14 if group == "actor" else 1,
        hidden_dims=[16],
        obs_normalization=True,
    )


def test_downloaded_checkpoint_hash_is_mandatory(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"incomplete download")
    with pytest.raises(ValueError, match="SHA256"):
        verify_artifact(path, "0" * 64)


def test_normalizer_counts_each_next_rollout_sample_once():
    obs = observations()
    actor = model(obs)
    actor.train()
    actor.update_rollout_normalization(obs)
    count = actor.obs_normalizer.count.clone()
    actor.update_normalization(obs)
    assert actor.obs_normalizer.count == count == 8
    actor.update_rollout_normalization(obs)
    assert actor.obs_normalizer.count == 16


def test_teacher_routing_and_inputs_preserve_student_commands():
    obs = observations(5)["policy"]
    # Upright walking, PPO-only transition band, deep fallen, body tilt, body z.
    obs[1, 5] = -math.cos(math.radians(30))
    obs[1, 3] = math.sin(math.radians(30))
    obs[2, 5] = 0.2
    obs[2, 3] = 0.98
    obs[2, 6:48] = 1
    obs[:, 48] = 0.2
    obs[3:, 48:51] = 0
    obs[3, 58] = 0.1
    obs[4, 57] = -0.02
    original = obs.clone()
    assert fallen_mask_from_obs(obs, (3, 6), 35).tolist() == [False, False, True, False, False]
    assert (~fallen_mask_from_obs(obs, (3, 6), 25)).tolist() == [True, False, False, True, True]
    body, teach = body_frame_masks(obs, (48, 51), (55, 61), (2, 3, 4), (2,))
    assert body.tolist() == [False, False, False, True, True]
    assert teach.tolist() == [False, False, False, True, False]
    masked = expert_input(obs, (48, 51), tuple(range(6, 48)), fallen_mask_from_obs(obs, (3, 6), 75))
    assert torch.count_nonzero(masked[:, 48:51]) == 0
    assert torch.count_nonzero(masked[2, 6:48]) == 0
    torch.testing.assert_close(obs, original)


def test_anchor_yaw_interpolation_is_signed_and_zero_preserving():
    table = {"lin": [0.0, 1.0], "targets": [0.5, 1.0], "pos": [[1.0, 2.0], [2.0, 3.0]], "neg": [[2.0, 3.0], [3.0, 4.0]]}
    obs = observations(4)["policy"]
    obs[:, 50] = torch.tensor([0.0, 0.5, -0.5, -2.0])
    obs[1:3, 48] = 0.5
    mapped = remap_anchor_yaw(obs, (48, 51), table)
    torch.testing.assert_close(mapped[:, 50], torch.tensor([0.0, 1.5, -2.5, -3.0]))
    assert obs[1, 50] == 0.5


def test_warm_start_and_resume_do_not_inherit_source_curriculum_counter():
    # Exercise native save/load semantics on a tiny real optimizer/model, without teachers/network.
    obs = observations()
    from rsl_rl.storage import RolloutStorage

    alg = MicroDuckVelStandPPO(
        model(obs), model(obs, "critic"), RolloutStorage("rl", 8, 2, obs, [14], "cpu"), bc_cfg=None
    )
    env = SimpleNamespace(common_step_counter=0)
    reset_counters = []

    def reset():
        reset_counters.append(env.common_step_counter)
        env.active_recovery_stage = env.common_step_counter // 24
        env.cached_observation_stage = env.active_recovery_stage

    env.reset = reset
    alg.env = env
    saved = alg.save()
    saved["infos"] = {"env_state": {"common_step_counter": 90024}}
    saved["iter"] = 3750
    saved.pop("microduck_velstand_state")
    assert alg.load(saved, None, True) is False
    assert env.common_step_counter == 0
    env.common_step_counter = 240
    resumed = alg.save()
    assert alg.load(resumed, None, True) is True
    assert env.common_step_counter == 240
    assert reset_counters == [0, 240]
    assert env.active_recovery_stage == 10
    assert env.cached_observation_stage == 10


def test_bc_changes_only_student_actor_and_keeps_teachers_frozen():
    from rsl_rl.storage import RolloutStorage

    obs = observations(256)
    obs["policy"][:64, 5] = 0.0
    obs["policy"][:64, 3] = 1.0
    obs["policy"][128:192, 58] = 0.1
    obs["policy"][192:, 57] = -0.02
    actor, critic = model(obs), model(obs, "critic")
    alg = MicroDuckVelStandPPO(actor, critic, RolloutStorage("rl", 256, 2, obs, [14], "cpu"), bc_cfg=None)
    import copy

    alg.expert, alg.anchor = copy.deepcopy(actor), copy.deepcopy(actor)
    for teacher in (alg.expert, alg.anchor):
        teacher.eval().requires_grad_(False)
    with torch.no_grad():
        alg.anchor.mlp[-1].bias.add_(1.0)
        alg.expert.mlp[-1].bias.sub_(0.5)
    alg.bc_cfg = {
        "gravity_slice": (3, 6),
        "gate_tilt_deg": 35.0,
        "anchor_tilt_deg": 25.0,
        "twist_slice": (48, 51),
        "body_slice": (55, 61),
        "body_active_axes": (2, 3, 4),
        "body_no_teacher_axes": (2,),
        "min_samples": 64,
        "coef": 1.0,
        "anchor_coef": 1.0,
        "epochs": 1,
        "mini_batches": 4,
        "min_mini_batch": 512,
    }
    alg.bc_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    alg.storage.observations["policy"][:] = obs["policy"]
    before_actor = copy.deepcopy(actor.state_dict())
    frozen = [copy.deepcopy(m.state_dict()) for m in (critic, alg.expert, alg.anchor)]
    stats = alg._bc_update()
    assert stats["expert_bc"] > 0
    assert stats["expert_bc_anchor_frac"] == 0.25
    assert stats["expert_bc_fallen_frac"] == 0.25
    assert stats["expert_bc_body_teach_frac"] == 0.25
    assert any(not torch.equal(v, before_actor[k]) for k, v in actor.state_dict().items())
    for module, state in zip((critic, alg.expert, alg.anchor), frozen, strict=True):
        for k, v in module.state_dict().items():
            torch.testing.assert_close(v, state[k], rtol=0, atol=0)
