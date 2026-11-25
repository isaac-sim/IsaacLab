# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for Franka Pour reset-dataset curriculum wiring."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumTermCfg
from isaaclab.test.utils import DeviceScope, test_devices

from isaaclab_tasks.contrib.franka_pour.mdp.reset_dataset import PourResetDatasetCurriculum
from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourEnv
from isaaclab_tasks.contrib.franka_pour.reset_sampler import ResetDatasetSampler, ResetDatasetSamplerCfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def _make_curriculum(num_envs, row_count, *, device="cpu", env=None, **overrides):
    cfg = SimpleNamespace(
        seed=17,
        reset_dataset_sampling_mode="adaptive",
        reset_dataset_sampler=ResetDatasetSamplerCfg(uniform_fraction=0.5),
        reset_dataset_top_grasp_count=None,
        curriculum_freeze=False,
        pour_target_frac=0.7,
    )
    vars(cfg).update(overrides)
    progress = SimpleNamespace(ever_success=torch.zeros(num_envs, dtype=torch.bool, device=device))
    if env is None:
        env = SimpleNamespace()
    vars(env).update(
        cfg=cfg,
        num_envs=num_envs,
        _reset_dataset_states={
            "category": torch.tensor([1, *([0] * (row_count - 1))], device=device),
            "objective": torch.zeros(row_count, device=device),
        },
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long, device=device),
        reset_buf=torch.zeros(num_envs, dtype=torch.bool, device=device),
        reset_dataset_row_id=torch.full((num_envs,), -1, device=device),
        pour_target_frac=torch.zeros(num_envs, device=device),
        _reset_dataset_seed=None,
        termination_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=progress)),
    )
    return PourResetDatasetCurriculum(CurriculumTermCfg(func=PourResetDatasetCurriculum), env), env, progress


def test_franka_pour_assignments_follow_environment_seed():
    """The same reset sequence and env seed yield the same row IDs across different global RNG states."""
    ids = torch.arange(8)
    sequences = []
    with torch.random.fork_rng(devices=[]):
        for global_seed in (53, 665):
            torch.manual_seed(global_seed)
            curriculum, env, _ = _make_curriculum(len(ids), row_count=17)
            rows = []
            for selection in (ids, ids[::2], ids[1::2], ids):
                curriculum(env, selection)
                rows.append(env.reset_dataset_row_id.clone())
            sequences.append(torch.stack(rows))
    torch.testing.assert_close(sequences[0], sequences[1])


def test_franka_pour_credits_only_finished_episodes():
    """Aborted episodes cannot change sampling scores when the curriculum starts from playback rows."""
    ids = torch.arange(4)
    curriculum, env, progress = _make_curriculum(
        len(ids), row_count=2, curriculum_freeze=True, reset_dataset_top_grasp_count=1
    )
    curriculum(env, ids)
    assert env.reset_dataset_row_id.tolist() == [0, 0, 0, 0]

    env.cfg.curriculum_freeze = False
    env.episode_length_buf[:] = 4
    progress.ever_success[0] = True
    env.reset_buf[0] = True
    metrics = curriculum(env, ids)
    assert metrics["sampler/effective_pool_fraction"] == pytest.approx(1.0)

    # Revisit the same row with a genuine unsuccessful episode. Its rolling rate
    # is now 1/2, so the adaptive probability should concentrate near it.
    env.cfg.curriculum_freeze = True
    curriculum(env, ids)
    env.cfg.curriculum_freeze = False
    progress.ever_success.zero_()
    metrics = curriculum(env, ids)
    assert metrics["sampler/effective_pool_fraction"] < 0.9


def test_franka_pour_replay_visits_each_row_in_successive_cycles():
    """Nearly all single-environment resets use exact shuffled replay after the first draw."""
    row_count = 5
    curriculum, env, _ = _make_curriculum(
        1, row_count, reset_dataset_sampler=ResetDatasetSamplerCfg(uniform_fraction=0.95)
    )
    env_ids = torch.tensor([0])
    assert curriculum(env, env_ids)["sampler/uniform_replay_fraction"] == 0.95
    for _ in range(2):
        rows = []
        for _ in range(row_count):
            curriculum(env, env_ids)
            rows.append(int(env.reset_dataset_row_id[0]))
        assert sorted(rows) == list(range(row_count))


@pytest.mark.parametrize("method", ["reset", "reset_to"])
def test_franka_pour_explicit_reset_seed_restarts_sampling(method):
    """The real public reset methods reseed the task samplers on every explicit seed call."""
    count = 8
    env = object.__new__(FrankaPourEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=count, write_data_to_sim=lambda: None, reset_to=lambda *args, **kwargs: None)
    env.sim = SimpleNamespace(forward=lambda: None)
    env.recorder_manager = SimpleNamespace(record_pre_reset=lambda ids: None, record_post_reset=lambda ids: None)
    env.observation_manager = SimpleNamespace(compute=lambda **kwargs: {})
    env.has_rtx_sensors = False
    env.extras = {}
    curriculum, _, _ = _make_curriculum(count, row_count=17, env=env, wait_for_textures=False)
    env._reset_idx = lambda ids: curriculum(env, ids)

    reset = env.reset if method == "reset" else lambda **kwargs: env.reset_to({}, **kwargs)
    with torch.random.fork_rng(devices=[]):
        reset(seed=123)
        first = env.reset_dataset_row_id.clone()
        reset(seed=456)
        assert not torch.equal(env.reset_dataset_row_id, first)
        reset(seed=123)
    torch.testing.assert_close(env.reset_dataset_row_id, first)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_franka_pour_resets_on_the_callers_cuda_stream(device):
    """An explicit playback row follows the queued Torch request on a non-default stream."""
    curriculum, env, _ = _make_curriculum(
        1,
        2,
        device=device,
        curriculum_freeze=True,
        reset_dataset_top_grasp_count=1,
        reset_dataset_sampler=ResetDatasetSamplerCfg(uniform_fraction=0.95),
    )
    env_ids = torch.tensor([0], device=device)
    curriculum(env, env_ids)

    # Warm the adaptive and forced paths and leave the last replay request at
    # row one. The next frozen reset must force the unique grasp at row zero.
    env.cfg.curriculum_freeze = False
    curriculum(env, env_ids)
    for _ in range(2):
        curriculum(env, env_ids)
        if env.reset_dataset_row_id.item() == 1:
            break
    assert env.reset_dataset_row_id.item() == 1

    env.cfg.curriculum_freeze = True
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
        curriculum(env, env_ids)
    stream.synchronize()
    assert env.reset_dataset_row_id.item() == 0


def _ring_values(sampler: ResetDatasetSampler, row: int) -> list[bool]:
    """Read one sampler ring in oldest-to-newest order."""
    size = int(sampler._history_sizes[row])
    pointer = int(sampler._history_pointers[row])
    capacity = sampler._success_history.shape[1]
    start = pointer if size == capacity else 0
    return [bool(sampler._success_history[row, (start + index) % capacity]) for index in range(size)]


@pytest.mark.filterwarnings("ignore:ResetDatasetSampler is deprecated:DeprecationWarning")
def test_legacy_sampler_rolling_history_matches_per_row_deques():
    """Duplicate IDs, wraparound, and oversized batches match sequential deque updates."""
    capacity = 5
    row_count = 4
    sampler = ResetDatasetSampler(
        row_count,
        "cpu",
        ResetDatasetSamplerCfg(monitored_history_len=capacity),
    )
    reference = [deque(maxlen=capacity) for _ in range(row_count)]
    generator = torch.Generator().manual_seed(31)

    for batch_size in (4, 11, 2, 17, 7, 23):
        rows = torch.randint(row_count, (batch_size,), generator=generator)
        success = torch.rand(batch_size, generator=generator) > 0.45
        sampler._record_validated(rows, success)
        for row, outcome in zip(rows.tolist(), success.tolist(), strict=True):
            reference[row].append(outcome)

        expected_sizes = torch.tensor([len(history) for history in reference], dtype=torch.int32)
        expected_counts = torch.tensor([sum(history) for history in reference], dtype=torch.int32)
        expected_rates = expected_counts.float() / expected_sizes.clamp_min(1)
        torch.testing.assert_close(sampler._history_sizes, expected_sizes)
        torch.testing.assert_close(sampler._history_success_counts, expected_counts)
        torch.testing.assert_close(sampler._success_rates, expected_rates)
        for row, history in enumerate(reference):
            assert _ring_values(sampler, row) == list(history)


def test_legacy_sampler_uniform_replay_covers_every_row_before_repeating(monkeypatch: pytest.MonkeyPatch):
    """The deprecated sampler still draws a complete shuffled cycle without replacement."""
    with pytest.warns(DeprecationWarning, match="isaaclab.envs.reset.ResetStateSampler"):
        sampler = ResetDatasetSampler(5, "cpu", ResetDatasetSamplerCfg(uniform_fraction=0.5))
    cyclic_rows: list[int] = []
    take_uniform_rows = sampler._take_uniform_rows

    def record_uniform_rows(count: int, generator: torch.Generator | None) -> torch.Tensor:
        rows = take_uniform_rows(count, generator)
        cyclic_rows.extend(rows.tolist())
        return rows

    monkeypatch.setattr(sampler, "_take_uniform_rows", record_uniform_rows)
    sampler._sample_with_uniform_replay(10, generator=torch.Generator().manual_seed(17))

    metrics = sampler.metrics()
    assert metrics["sampler/uniform_replay_fraction"] == pytest.approx(0.5)
    assert len(cyclic_rows) == sampler.row_count
    assert set(cyclic_rows) == set(range(sampler.row_count))
