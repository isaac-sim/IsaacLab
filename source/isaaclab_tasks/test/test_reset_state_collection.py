# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collection validity, reproducibility, and batch consistency at the task callback boundary."""

import random

import numpy as np
import pytest
import torch

from isaaclab_tasks.utils import ResetStateCollectionCfg, collect_reset_states

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


def test_collection_rejects_candidates_covers_groups_and_preserves_rng() -> None:
    values = torch.empty(4, 2)
    env_ids = torch.arange(4)

    def generate(ids, sign):
        values[ids, 0] = sign * (1.0 + torch.rand(len(ids)))
        values[ids, 1] = torch.from_numpy(np.random.random(len(ids))).float() + random.random()
        return ids != 0

    def capture(ids):
        return {"rigid_object": {"cube": {"root_pose": values[ids]}}}

    cfg = ResetStateCollectionCfg(size=7, max_batches=2, seed=17)
    generators = {"table": lambda ids: generate(ids, 1), "grasped": lambda ids: generate(ids, -1)}
    checks = {"reachable": lambda ids: ids != 3}
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    states, group_ids, report = collect_reset_states(cfg, env_ids, generators, capture, checks)
    assert group_ids.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert report == {
        "table": {"attempted": 8, "accepted": 4, "rejected/generator": 2, "rejected/reachable": 2},
        "grasped": {"attempted": 8, "accepted": 3, "rejected/generator": 2, "rejected/reachable": 2},
    }
    poses = states["rigid_object"]["cube"]["root_pose"]
    assert poses.shape == (7, 2)
    assert (poses[:, 0] * torch.tensor([1, 1, 1, 1, -1, -1, -1]) > 1.0).all()
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state
    torch.rand(3)
    np.random.random(3)
    random.random()
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    repeated, _, _ = collect_reset_states(cfg, env_ids, generators, capture, checks)
    torch.testing.assert_close(repeated["rigid_object"]["cube"]["root_pose"], poses)
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state


def test_impossible_collection_is_bounded_reports_rejections_and_restores_rng() -> None:
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    batches = 0

    def blocked(ids):
        nonlocal batches
        batches += 1
        torch.rand(len(ids))
        np.random.random(len(ids))
        random.random()

    with pytest.raises(RuntimeError, match="collected 2/5 states after 2 batches.*collision_free.*4"):
        collect_reset_states(
            ResetStateCollectionCfg(size=5, max_batches=2),
            torch.arange(3),
            {"blocked": blocked},
            lambda ids: {"pose": torch.full((len(ids), 1), 999.0)},
            {"collision_free": lambda ids: ids == 1},
        )
    assert batches == 2
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state
    with pytest.raises(ValueError, match="'generator' is reserved"):
        collect_reset_states(
            ResetStateCollectionCfg(size=1),
            torch.arange(3),
            lambda ids: torch.zeros_like(ids, dtype=torch.bool),
            lambda ids: {"pose": torch.zeros(len(ids), 7)},
            {"generator": lambda ids: torch.ones_like(ids, dtype=torch.bool)},
        )


def test_capture_requires_selected_rows_and_a_consistent_state_tree() -> None:
    cfg = ResetStateCollectionCfg(size=3, max_batches=2)
    with pytest.raises(ValueError, match="leading dimension of 2"):
        collect_reset_states(cfg, torch.arange(3), lambda ids: ids > 0, lambda ids: {"pose": torch.zeros(3, 7)})
    calls = 0

    def capture(ids):
        nonlocal calls
        calls += 1
        return {"pose": torch.zeros(len(ids), 7 if calls == 1 else 6)}

    with pytest.raises(ValueError, match="shape, dtype, or device changed"):
        collect_reset_states(cfg, torch.arange(2), lambda ids: None, capture)
