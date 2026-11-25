# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode evaluation excludes autoresets and resets recurrent policy state."""

from types import SimpleNamespace

import pytest

from isaaclab_rl.entrypoints.backends import play_rsl_rl

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


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

    count, reward, seconds, endings = play_rsl_rl.score_policy(Env(), Policy())
    assert count == 2
    assert reward == pytest.approx(2.5)
    assert seconds == pytest.approx(0.15)
    assert endings == {"failure": 0.5, "timeout": 0.5}
    assert len(resets) == 3


def test_evaluation_uses_play_configuration_and_cleanup(monkeypatch, tmp_path, caplog):
    """Evaluation uses the selected physics, agent overrides and checkpoint without exporting or playing forever."""
    from contextlib import nullcontext

    from isaaclab_tasks.utils import resolve_task_config

    received = {}
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setattr("sys.argv", ["play.py"])

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

    def make(task, cfg, args, **kwargs):
        received["env"] = cfg
        return Env()

    def resolve(*args, **kwargs):
        received["resolutions"] = received.get("resolutions", 0) + 1
        return resolve_task_config(*args, **kwargs)

    monkeypatch.setattr(play_rsl_rl, "resolve_task_config", resolve)
    monkeypatch.setattr(play_rsl_rl, "launch_simulation", lambda *args: nullcontext())
    monkeypatch.setattr(
        play_rsl_rl,
        "startup_screen",
        lambda *args, **kwargs: nullcontext(SimpleNamespace(stage=lambda *args: None, close=lambda: None)),
    )
    monkeypatch.setattr(play_rsl_rl, "show_run_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(play_rsl_rl, "create_isaaclab_env", make)
    monkeypatch.setattr(play_rsl_rl, "RslRlVecEnvWrapper", lambda env, **kwargs: env)
    monkeypatch.setattr(play_rsl_rl, "create_rsl_rl_runner", construct)
    monkeypatch.setattr(play_rsl_rl, "score_policy", lambda *args: (1, 1.0, 1.0, {"timeout": 1.0}))
    with caplog.at_level("INFO", logger=play_rsl_rl.__name__):
        play_rsl_rl.run(
            [
                "--evaluate",
                "--task",
                "Isaac-Reach-Franka",
                "--checkpoint",
                str(checkpoint),
                "physics=newton_steel",
                "env.sim.physics.solver_cfg.articulation_constraint_max=48",
                "presets=hold",
                "agent.actor.hidden_dims=[128,128]",
                "agent.clip_actions=1.5",
            ]
        )
    assert received["resolutions"] == 1
    assert received["checkpoint"] == str(checkpoint)
    assert received["agent"].actor.hidden_dims == [128, 128]
    assert received["agent"].clip_actions == 1.5
    assert "hold" in received["agent"].experiment_name
    assert received["env"].observations.policy.ee_target_error is not None
    assert received["env"].sim.physics.solver_cfg.articulation_constraint_max == 48
    assert received["closed"]
    assert "1 episodes, mean reward 1.000, mean duration 1.000 s" in caplog.text
    assert not (tmp_path / "exported").exists()
