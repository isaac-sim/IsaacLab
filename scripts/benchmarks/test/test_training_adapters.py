# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for RL-library-specific training metric capture."""

from types import SimpleNamespace

import pytest
import torch

from scripts.benchmarks.sb3 import bench_sb3
from scripts.benchmarks.skrl import bench_skrl


def test_sb3_iteration_time_includes_policy_update(monkeypatch: pytest.MonkeyPatch):
    """Test that SB3 reports collection time separately from the full training iteration."""
    timestamps = iter([1_000_000_000, 1_000_000_000, 3_000_000_000, 6_000_000_000])
    monkeypatch.setattr(bench_sb3.time, "perf_counter_ns", lambda: next(timestamps))

    callback = bench_sb3._build_benchmark_callback_class()()
    callback.model = SimpleNamespace(ep_info_buffer=[])
    callback._on_training_start()
    callback._on_rollout_start()
    callback._on_rollout_end()
    callback._on_training_end()

    assert callback.collection_times_s == [2.0]
    assert callback.iter_times_s == [5.0]


def test_skrl_reward_uses_episode_return_tracking(monkeypatch: pytest.MonkeyPatch):
    """Test that SKRL records its canonical total-reward metric at rollout boundaries."""
    from skrl.trainers.torch import SequentialTrainer

    class FakeEnv:
        num_agents = 1

        def step(self, actions):
            return None, torch.tensor([1.0]), None, None, {}

    class FakeAgent:
        cfg = SimpleNamespace(rollouts=2)

        def __init__(self):
            self.tracking_data = {}

        def post_interaction(self, *, timestep: int, timesteps: int) -> None:
            self.tracking_data.clear()

    def run_two_steps(trainer) -> None:
        for timestep in range(2):
            trainer.env.step(None)
            trainer.agents.tracking_data = {
                "Reward / Total reward (mean)": [10.0, 20.0],
                "Episode / Total timesteps (mean)": [5.0, 7.0],
            }
            trainer.agents.post_interaction(timestep=timestep, timesteps=2)

    monkeypatch.setattr(SequentialTrainer, "train", run_two_steps)
    timestamps = iter([1_000_000_000, 3_000_000_000, 6_000_000_000, 6_000_000_000])
    monkeypatch.setattr(bench_skrl.time, "perf_counter_ns", lambda: next(timestamps))
    trainer_class = bench_skrl._build_benchmark_trainer_class()
    trainer = trainer_class.__new__(trainer_class)
    trainer.env = FakeEnv()
    trainer.agents = FakeAgent()
    trainer.cfg = SimpleNamespace(timesteps=2)
    trainer.num_simultaneous_agents = 1
    trainer.collection_times_s = []
    trainer.iter_times_s = []
    trainer.iter_rewards = []
    trainer.iter_ep_lengths = []

    trainer.train()

    assert trainer.collection_times_s == [2.0]
    assert trainer.iter_times_s == [5.0]
    assert trainer.iter_rewards == [15.0]


def test_skrl_parser_rejects_unimplemented_modes():
    """Test that SKRL rejects modes which cannot emit complete benchmark metrics."""
    unsupported = [("--ml_framework", "jax"), ("--algorithm", "IPPO")]

    for option, value in unsupported:
        with pytest.raises(SystemExit):
            bench_skrl._parse_args(["--task", "unused", option, value])
    with pytest.raises(SystemExit):
        bench_skrl._parse_args(["--task", "unused", "--distributed"])
