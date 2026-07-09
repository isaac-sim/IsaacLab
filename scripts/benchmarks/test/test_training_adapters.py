# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for RL-library benchmark adapter behavior."""

from types import SimpleNamespace

import pytest
import torch

from scripts.benchmarks import play, training
from scripts.benchmarks.rl_games import benchmark_rl_games_play as play_rl_games
from scripts.benchmarks.rsl_rl import benchmark_rsl_rl_play as play_rsl_rl
from scripts.benchmarks.rsl_rl import benchmark_rsl_rl_train as train_rsl_rl
from scripts.benchmarks.sb3 import benchmark_sb3_play as play_sb3
from scripts.benchmarks.sb3 import benchmark_sb3_train as train_sb3
from scripts.benchmarks.skrl import benchmark_skrl_play as play_skrl
from scripts.benchmarks.skrl import benchmark_skrl_train as train_skrl


@pytest.mark.parametrize("library", ["rl_games", "rsl_rl", "sb3", "skrl"])
def test_training_dispatches_libraries_to_library_named_adapters(library: str):
    """The training dispatcher uses the library-named benchmark adapter."""
    assert training.LIBRARY_ENTRYPOINTS[library].name == f"benchmark_{library}_train.py"


@pytest.mark.parametrize("library", ["rl_games", "rsl_rl", "sb3", "skrl"])
def test_play_dispatches_libraries_to_library_named_adapters(library: str):
    """The play dispatcher uses the library-named benchmark adapter."""
    assert play.LIBRARY_ENTRYPOINTS[library].name == f"benchmark_{library}_play.py"


@pytest.mark.parametrize("adapter", [play_rl_games, play_rsl_rl, play_sb3, play_skrl])
def test_play_adapter_help_does_not_require_task(adapter):
    """Play adapter help exits successfully without a task selection."""
    with pytest.raises(SystemExit) as exc_info:
        adapter._parse_args(["--help"])
    assert exc_info.value.code == 0


def test_rsl_rl_disables_code_state_capture():
    logger = SimpleNamespace(git_status_repos=["rsl_rl.py"])
    runner = SimpleNamespace(logger=logger)

    train_rsl_rl._disable_code_state_capture(runner)

    assert logger.git_status_repos == []


def test_sb3_iteration_time_includes_policy_update(monkeypatch: pytest.MonkeyPatch):
    """Test that SB3 reports collection time separately from the full training iteration."""
    timestamps = iter([1_000_000_000, 1_000_000_000, 3_000_000_000, 6_000_000_000])
    monkeypatch.setattr(train_sb3.time, "perf_counter_ns", lambda: next(timestamps))

    callback = train_sb3._build_benchmark_callback_class()()
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
    monkeypatch.setattr(train_skrl.time, "perf_counter_ns", lambda: next(timestamps))
    trainer_class = train_skrl._build_benchmark_trainer_class()
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
            train_skrl._parse_args(["--task", "unused", option, value])
    with pytest.raises(SystemExit):
        train_skrl._parse_args(["--task", "unused", "--distributed"])
