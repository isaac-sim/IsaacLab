# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Franka Pour reset-dataset curriculum."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumTermCfg

from isaaclab_tasks.contrib.franka_pour.mdp.reset_dataset import PourResetDatasetCurriculum
from isaaclab_tasks.contrib.franka_pour.reset_sampler import ResetDatasetSamplerCfg


def _states() -> dict[str, torch.Tensor]:
    """Return a small reset dataset with two grasp rows."""
    return {
        "category": torch.tensor((1, 1, 0, 0, 0, 0), dtype=torch.int8),
        "objective": torch.tensor((1.0, 0.5, -1.0, -1.0, -1.0, -1.0)),
    }


class _FakeResetDatasetEnv:
    """Minimal environment state used by the curriculum term."""

    def __init__(
        self,
        *,
        freeze: bool = False,
        sampling_mode: str = "adaptive",
        top_grasp_count: int | None = None,
    ):
        self.num_envs = 4
        self.device = "cpu"
        self._reset_dataset_states = _states()
        self.cfg = SimpleNamespace(
            curriculum_freeze=freeze,
            pour_target_frac=0.7,
            reset_dataset_sampler=ResetDatasetSamplerCfg(
                monitored_history_len=4,
                target_success_rate=0.5,
                uniform_fraction=0.25,
                uniform_fraction_initial=0.5,
            ),
            reset_dataset_sampling_mode=sampling_mode,
            reset_dataset_top_grasp_count=top_grasp_count,
        )
        self.learning_progress = SimpleNamespace(ever_success=torch.zeros(self.num_envs, dtype=torch.bool))
        self.termination_manager = SimpleNamespace(
            get_term_cfg=lambda _name: SimpleNamespace(func=self.learning_progress)
        )
        self.reset_dataset_row_id = torch.full((self.num_envs,), -1, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.pour_target_frac = torch.zeros(self.num_envs)


def _term(env: _FakeResetDatasetEnv) -> PourResetDatasetCurriculum:
    """Create the curriculum term for a fake environment."""
    cfg = CurriculumTermCfg(func=PourResetDatasetCurriculum)
    return PourResetDatasetCurriculum(cfg, env)


def _install_adaptive_draw(
    monkeypatch: pytest.MonkeyPatch,
    term: PourResetDatasetCurriculum,
    rows: tuple[int, ...] = (0, 1, 2, 3),
) -> None:
    """Replace adaptive sampling with a deterministic row sequence."""
    sampled_rows = torch.tensor(rows, dtype=torch.long)

    def sample_with_uniform_replay(
        count: int,
    ) -> torch.Tensor:
        assert count <= sampled_rows.numel()
        return sampled_rows[:count].clone()

    monkeypatch.setattr(term._sampler, "_sample_with_uniform_replay", sample_with_uniform_replay)


def test_adaptive_sampling_updates_rows_from_local_progress(monkeypatch: pytest.MonkeyPatch):
    env = _FakeResetDatasetEnv()
    term = _term(env)
    _install_adaptive_draw(monkeypatch, term)

    term(env, slice(None))
    env.episode_length_buf[:] = 1
    env.learning_progress.ever_success[:] = torch.tensor((True, False, True, False))
    term(env, slice(None))

    assert term._sampler._history_sizes.tolist() == [1, 1, 1, 1, 0, 0]
    assert term._sampler._history_success_counts.tolist() == [1, 0, 1, 0, 0, 0]
    torch.testing.assert_close(
        term._sampler._success_rates,
        torch.tensor((1.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    )
    assert env.pour_target_frac.tolist() == pytest.approx([0.7] * env.num_envs)


def test_uniform_mode_samples_every_row_without_adaptive_draw(monkeypatch: pytest.MonkeyPatch):
    env = _FakeResetDatasetEnv(sampling_mode="uniform")
    term = _term(env)

    def randint(_high: int, size: tuple[int, ...], **_kwargs) -> torch.Tensor:
        return torch.tensor((4, 3, 2, 1), dtype=torch.long)[: size[0]]

    monkeypatch.setattr(torch, "randint", randint)
    monkeypatch.setattr(
        term._sampler,
        "_sample_with_uniform_replay",
        lambda *_args, **_kwargs: pytest.fail("uniform mode invoked the adaptive sampler"),
    )

    term(env, slice(None))
    assert env.reset_dataset_row_id.tolist() == [4, 3, 2, 1]

    env.episode_length_buf[:] = 1
    env.learning_progress.ever_success[:] = torch.tensor((False, True, False, True))
    term(env, slice(None))

    assert term._sampler._history_sizes.tolist() == [0, 1, 1, 1, 1, 0]
    assert term._sampler._history_success_counts.tolist() == [0, 1, 0, 1, 0, 0]


def test_frozen_playback_uses_top_grasp_rows(monkeypatch: pytest.MonkeyPatch):
    env = _FakeResetDatasetEnv(freeze=True, top_grasp_count=1)
    term = _term(env)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda _high, size, **_kwargs: torch.zeros(size, dtype=torch.long),
    )
    monkeypatch.setattr(
        term._sampler,
        "_record_validated",
        lambda *_args: pytest.fail("frozen playback recorded an outcome"),
    )
    metrics = term(env, slice(None))

    assert env.reset_dataset_row_id.tolist() == [0, 0, 0, 0]
    assert metrics == {
        "frozen_pool_fraction": pytest.approx(1.0 / 6.0),
        "frozen_pool_size": 1.0,
    }

    env.episode_length_buf[:] = 1
    env.learning_progress.ever_success[:] = True
    term(env, slice(None))
