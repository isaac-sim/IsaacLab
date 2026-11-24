# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation uses normal play configuration and the configured runner."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from isaaclab_neural.entrypoints import evaluate

from isaaclab_tasks.utils import resolve_task_config

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_cross_physics_preserves_agent_and_domain_overrides(monkeypatch, tmp_path):
    received = {}
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()

    class Env:
        unwrapped = SimpleNamespace(device="cpu")

        def close(self):
            received["closed"] = True

    class Runner:
        def load(self, path, **kwargs):
            received["checkpoint"] = path

        def get_inference_policy(self, **kwargs):
            return object()

    def construct(env, cfg, *, inference):
        received["agent"] = cfg
        assert inference
        return Runner()

    def make(task, *, cfg):
        received["env"] = cfg
        return Env()

    monkeypatch.setattr(evaluate, "launch_simulation", lambda *args: nullcontext())
    monkeypatch.setattr(evaluate.gym, "make", make)
    monkeypatch.setattr(evaluate, "RslRlVecEnvWrapper", lambda env, **kwargs: env)
    monkeypatch.setattr(evaluate, "create_rsl_rl_runner", construct)
    monkeypatch.setattr(evaluate, "score", lambda *args: (1, 1.0, 1.0, {}))
    evaluate.main(
        [
            "--task",
            "Isaac-Reach-Franka",
            "--checkpoint",
            str(checkpoint),
            "--trained_on",
            "newton_mjwarp",
            "physics=newton_steel",
            "env.sim.physics.solver_cfg.articulation_constraint_max=48",
            "presets=hold",
            "agent.actor.hidden_dims=[128,128]",
        ]
    )
    agent, env = received["agent"], received["env"]
    assert agent.actor.hidden_dims == [128, 128]
    assert "hold" in agent.experiment_name
    assert env.observations.policy.ee_target_error is not None
    reference, _ = resolve_task_config("Isaac-Reach-Franka", None, overrides=["physics=newton_steel"])
    assert type(env.sim.physics.solver_cfg) is type(reference.sim.physics.solver_cfg)
    assert env.sim.physics.solver_cfg.articulation_constraint_max == 48
    assert received["closed"]


def test_score_counts_each_first_episode_once():
    import torch

    step = 0
    resets = []
    termination = SimpleNamespace(
        active_terms=["failure", "timeout"],
        get_term=lambda name: (
            torch.tensor([step == 1, False, False]) if name == "failure" else torch.tensor([False, step == 2, False])
        ),
    )
    task = SimpleNamespace(num_envs=3, device="cpu", max_episode_length=3, step_dt=0.1, termination_manager=termination)

    class Env:
        unwrapped = task

        def reset(self):
            return {}, {}

        def step(self, actions):
            nonlocal step
            step += 1
            return {}, torch.tensor([1.0 if step == 1 else 100.0, 2.0, 3.0]), torch.tensor([True, step >= 2, False]), {}

    class Policy:
        def __call__(self, obs):
            return torch.zeros(3, 1)

        def reset(self, dones):
            resets.append(dones.clone())

    count, reward, seconds, endings = evaluate.score(Env(), Policy())
    assert count == 2
    assert reward == pytest.approx(2.5)
    assert seconds == pytest.approx(0.15)
    assert endings == {"failure": 0.5, "timeout": 0.5}
    assert len(resets) == 3
