# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gaze geometry and gaze-supervised PPO, without a simulator."""

import pytest
import torch
from isaaclab_fovi.mdp import angles_to_direction, clamp_gaze, direction_to_angles
from isaaclab_fovi.rsl_rl.gaze_supervised_ppo import GazeSupervisedPPO, gaze_loss
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

NUM_ENVS = 64
STEPS = 8


def test_gaze_angles_round_trip_and_follow_the_camera_convention():
    angles = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [-20.0, 35.0]])
    directions = angles_to_direction(angles)
    torch.testing.assert_close(direction_to_angles(directions), angles, atol=1e-4, rtol=0)
    # OpenCV camera frame: z forward, y down, x right; positive tilt looks up, positive pan looks left.
    torch.testing.assert_close(directions[0], torch.tensor([0.0, 0.0, 1.0]))
    assert directions[1, 1] < 0 and directions[2, 0] < 0


def test_clamp_gaze_is_radial():
    gaze = torch.tensor([[0.3, 0.4], [3.0, 4.0]])
    torch.testing.assert_close(clamp_gaze(gaze, 1.0), torch.tensor([[0.3, 0.4], [0.6, 0.8]]))


def test_gaze_loss_ignores_invalid_labels():
    mean = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    labels = torch.tensor([[0.3, 0.4, 1.0], [float("nan"), float("nan"), 0.0]])
    loss = gaze_loss(mean, labels)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.5)
    assert torch.isfinite(mean.grad).all() and not mean.grad[1].any()


def _algorithm() -> tuple[GazeSupervisedPPO, TensorDict]:
    torch.manual_seed(0)
    labels = torch.rand(NUM_ENVS, 3) * 2 - 1
    labels[:, 2] = (torch.arange(NUM_ENVS) % 4 != 0).float()
    labels[labels[:, 2] == 0, :2] = float("nan")
    obs = TensorDict({"policy": torch.randn(NUM_ENVS, 6), "oracle": labels}, batch_size=[NUM_ENVS])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    distribution = {"class_name": "GaussianDistribution", "init_std": 1.0}
    actor = MLPModel(obs, groups, "actor", 3, hidden_dims=[32], distribution_cfg=distribution)
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[32])
    storage = RolloutStorage("rl", NUM_ENVS, STEPS, obs, [3], "cpu")
    settings = dict(num_learning_epochs=4, num_mini_batches=4, learning_rate=3e-3, schedule="fixed", device="cpu")
    return GazeSupervisedPPO(actor, critic, storage, **settings), obs


def test_update_pulls_the_gaze_toward_the_teacher():
    algorithm, obs = _algorithm()
    valid = obs["oracle"][:, 2] > 0.5

    def distance() -> float:
        with torch.no_grad():
            mean = algorithm.actor(obs)[:, -2:]
        return float((mean - obs["oracle"][:, :2])[valid].norm(dim=-1).mean())

    before = distance()
    for _ in range(40):
        for _ in range(STEPS):
            with torch.inference_mode():
                algorithm.act(obs)
                algorithm.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})
        algorithm.compute_returns(obs)
        losses = algorithm.update()
    assert losses["oracle_valid"] == pytest.approx(0.75)
    assert all(torch.isfinite(p).all() for p in algorithm.actor.parameters())
    assert distance() < 0.7 * before
