# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collection validity, reproducibility, and tensor-state consistency at the public API."""

import random

import numpy as np
import pytest
import torch

from isaaclab.envs.reset import ResetStateBuffer, ResetStateBufferCfg

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

    generators = {"table": lambda ids: generate(ids, 1), "grasped": lambda ids: generate(ids, -1)}
    checks = {"reachable": lambda ids: ids != 3}
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    buffer = ResetStateBuffer(ResetStateBufferCfg(size=7, max_batches=2, seed=17))
    buffer.collect(env_ids, generators, capture, checks)
    assert buffer.num_states == 7
    assert buffer.group_names == ("table", "grasped")
    assert buffer.group_ids.tolist() == [0, 0, 0, 0, 1, 1, 1]
    exported_groups = buffer.group_ids
    exported_groups.fill_(-1)
    assert buffer.group_ids.tolist() == [0, 0, 0, 0, 1, 1, 1]
    exported_groups.resize_(0)
    assert buffer.num_states == 7
    with pytest.raises(AttributeError):
        buffer.group_ids = torch.empty(0, dtype=torch.long)
    with pytest.raises(AttributeError):
        buffer.group_names = ()
    assert buffer.collection_report == {
        "table": {"attempted": 8, "accepted": 4, "rejected/generator": 2, "rejected/reachable": 2},
        "grasped": {"attempted": 8, "accepted": 3, "rejected/generator": 2, "rejected/reachable": 2},
    }
    row_ids = torch.tensor([6, 0, 4])
    selected = buffer.get_state(row_ids)["rigid_object"]["cube"]["root_pose"]
    assert (selected[:, 0] * torch.tensor([-1, 1, -1]) > 1.0).all()
    expected = selected.clone()
    values.zero_()
    selected.zero_()
    exported = buffer.state
    exported["rigid_object"]["cube"]["root_pose"].zero_()
    torch.testing.assert_close(buffer.get_state(row_ids)["rigid_object"]["cube"]["root_pose"], expected)
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state
    torch.rand(3)
    np.random.random(3)
    random.random()
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    repeated = ResetStateBuffer(buffer.cfg)
    repeated.collect(env_ids, generators, capture, checks)
    torch.testing.assert_close(
        repeated.state["rigid_object"]["cube"]["root_pose"], buffer.state["rigid_object"]["cube"]["root_pose"]
    )
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert random.getstate() == python_state


def test_impossible_recollection_is_bounded_and_preserves_the_complete_catalog() -> None:
    buffer = ResetStateBuffer(ResetStateBufferCfg(size=5, max_batches=2))
    buffer.collect(torch.arange(3), lambda ids: None, lambda ids: {"pose": ids[:, None].float()})
    catalog = buffer.get_state(torch.arange(5))["pose"]
    with pytest.raises(RuntimeError, match="collected 2/5 states after 2 batches.*collision_free.*4"):
        buffer.collect(
            torch.arange(3),
            {"blocked": lambda ids: None},
            lambda ids: {"pose": torch.full((len(ids), 1), 999.0)},
            {"collision_free": lambda ids: ids == 1},
        )
    assert buffer.num_states == 5
    torch.testing.assert_close(buffer.get_state(torch.arange(5))["pose"], catalog)
    assert buffer.group_names == ("default",)
    assert buffer.group_ids.tolist() == [0] * 5
    assert buffer.collection_report == {"default": {"attempted": 6, "accepted": 5}}
    with pytest.raises(ValueError, match="'generator' is reserved"):
        buffer.collect(
            torch.arange(3),
            lambda ids: torch.zeros_like(ids, dtype=torch.bool),
            lambda ids: {"pose": torch.zeros(len(ids), 7)},
            {"generator": lambda ids: torch.ones_like(ids, dtype=torch.bool)},
        )


def test_capture_requires_selected_rows_and_a_consistent_state_tree() -> None:
    buffer = ResetStateBuffer(ResetStateBufferCfg(size=3, max_batches=2))
    with pytest.raises(ValueError, match="leading dimension of 2"):
        buffer.collect(torch.arange(3), lambda ids: ids > 0, lambda ids: {"pose": torch.zeros(3, 7)})
    calls = 0

    def capture(ids):
        nonlocal calls
        calls += 1
        return {"pose": torch.zeros(len(ids), 7 if calls == 1 else 6)}

    with pytest.raises(ValueError, match="shape, dtype, or device changed"):
        buffer.collect(torch.arange(2), lambda ids: None, capture)
