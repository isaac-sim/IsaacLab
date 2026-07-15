# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Franka Pour reset-dataset curriculum adapter."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumTermCfg

from isaaclab_tasks.contrib.franka_pour.mdp.reset_dataset import (
    PourResetDatasetCurriculum,
    reset_dataset_difficulty,
)
from isaaclab_tasks.utils.adaptive_reset_sampler import AdaptiveResetSamplerCfg


def _task_contract() -> dict[str, object]:
    return {
        "arm_home": (0.0, 0.0),
        "arm_joint_limits": torch.tensor(((-2.0, 2.0), (-2.0, 2.0))),
        "source_region_center": (0.0, 0.0, 0.1),
        "target_center_xy": (0.0, -0.2),
        "tabletop_support_lower_xy": (-1.0, -1.0),
        "tabletop_support_upper_xy": (1.0, 1.0),
        "gripper_position_range": (0.0, 0.04),
    }


def _states() -> dict[str, torch.Tensor]:
    # Rows zero and one are grasping; rows two through five are increasingly hard reaching states.
    return {
        "category": torch.tensor((1, 1, 0, 0, 0, 0), dtype=torch.int8),
        "objective": torch.tensor((1.0, 0.0, -1.0, -1.0, -1.0, -1.0)),
        "arm_joint_position": torch.tensor(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.4, 0.4), (1.0, 1.0), (1.8, 1.8))),
        "source_root_pose": torch.tensor(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 1.0),
            )
        ),
        "target_root_pose": torch.tensor(
            (
                (0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 1.0),
            )
        ),
        "finger_joint_position": torch.tensor(
            ((0.028, 0.028), (0.028, 0.028), (0.04, 0.04), (0.03, 0.03), (0.02, 0.02), (0.0, 0.0))
        ),
    }


class FakeResetDatasetEnv:
    """Small manager-compatible environment stub."""

    def __init__(self, *, freeze: bool = False, top_grasp_count: int | None = None):
        self.num_envs = 4
        self.device = "cpu"
        self._uses_reset_dataset = True
        self._reset_dataset_states = _states()
        self._reset_dataset_metadata = {"task_contract": _task_contract()}
        self.cfg = SimpleNamespace(
            reset_dataset_sampler=AdaptiveResetSamplerCfg(
                target_success_rate=0.5,
                temperature=0.1,
                history_capacity=8,
                prior_strength=2.0,
                initial_frontier_size=2,
                probe_size=2,
                probe_fraction=0.1,
                replay_fraction=0.1,
                frontier_evidence=1.0,
            ),
            reset_dataset_top_grasp_count=top_grasp_count,
            curriculum_freeze=freeze,
            pour_target_frac=0.3,
        )
        self.reset_dataset_row_id = torch.full((self.num_envs,), -1, dtype=torch.long)
        self._forced_reset_dataset_row = torch.full_like(self.reset_dataset_row_id, -1)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool)
        self.pour_target_frac = torch.zeros(self.num_envs)


def _term(env: FakeResetDatasetEnv) -> PourResetDatasetCurriculum:
    return PourResetDatasetCurriculum(CurriculumTermCfg(func=PourResetDatasetCurriculum), env)


def test_reset_dataset_difficulty_orders_objectives_then_grades_non_grasps():
    difficulty = reset_dataset_difficulty(_states(), _task_contract())

    assert difficulty.shape == (6,)
    assert difficulty.tolist() == sorted(difficulty.tolist())
    assert difficulty[0] == 0.0
    assert difficulty[1] == pytest.approx(0.5)
    assert bool(((difficulty >= 0.0) & (difficulty <= 1.0)).all())


def test_reset_dataset_curriculum_honors_exact_raw_row_overrides():
    env = FakeResetDatasetEnv()
    env._forced_reset_dataset_row[:] = torch.tensor((5, -1, 3, -1))
    term = _term(env)

    metrics = term(env, slice(None))

    assert env.reset_dataset_row_id[[0, 2]].tolist() == [5, 3]
    assert bool(torch.isin(env.reset_dataset_row_id, torch.arange(6)).all())
    assert env.pour_target_frac.tolist() == pytest.approx([0.3] * 4)
    assert set(metrics) == {
        "predicted_success_rate",
        "observed_success_rate",
        "dataset_success_rate",
        "dataset_ever_solved_fraction",
        "frontier_fraction",
        "effective_pool_size",
    }


def test_reset_dataset_curriculum_records_completed_rows_and_reports_compact_progress():
    env = FakeResetDatasetEnv()
    term = _term(env)
    env._forced_reset_dataset_row[:] = torch.arange(4)
    term(env, slice(None))
    completed_rows = env.reset_dataset_row_id.clone()
    env.episode_length_buf[:] = 1
    env.episode_succeeded[:] = torch.tensor((True, False, True, False))

    metrics = term(env, slice(None))

    assert 0.0 <= metrics["predicted_success_rate"] <= 1.0
    assert 0.0 <= metrics["observed_success_rate"] <= 1.0
    assert metrics["dataset_success_rate"] > 0.0
    assert metrics["dataset_ever_solved_fraction"] > 0.0
    assert metrics["dataset_success_rate"] <= metrics["dataset_ever_solved_fraction"]
    assert completed_rows.min() >= 0


def test_frozen_reset_dataset_samples_only_top_grasps_but_allows_exact_diagnostics(monkeypatch):
    env = FakeResetDatasetEnv(freeze=True, top_grasp_count=1)
    term = _term(env)
    monkeypatch.setattr(torch, "randint", lambda *_args, **_kwargs: torch.zeros(4, dtype=torch.long))

    metrics = term(env, slice(None))
    assert env.reset_dataset_row_id.tolist() == [0, 0, 0, 0]
    assert metrics == {"frozen_pool_fraction": pytest.approx(1.0 / 6.0), "frozen_pool_size": 1.0}

    env._forced_reset_dataset_row[:] = torch.tensor((5, -1, -1, -1))
    env.episode_length_buf[:] = 1
    env.episode_succeeded[:] = True
    state_before = term._sampler.state_dict()
    term(env, slice(None))
    assert env.reset_dataset_row_id.tolist() == [5, 0, 0, 0]
    assert all(torch.equal(value, term._sampler.state_dict()[name]) for name, value in state_before.items())


def test_reset_dataset_curriculum_rejects_unknown_forced_row():
    env = FakeResetDatasetEnv()
    env._forced_reset_dataset_row[0] = 99
    term = _term(env)

    with pytest.raises(ValueError, match="Unknown raw reset-row IDs"):
        term(env, slice(None))
