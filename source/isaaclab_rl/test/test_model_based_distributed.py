# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distributed updates must match learning from the combined batch."""

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from isaaclab_rl.model_based.models import _CudaGraphNormalization
from isaaclab_rl.model_based.optim import _optimize_step


def _compare_updates(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    x = torch.tensor([[1.0, 3.0], [-2.0, 4.0], [5.0, -1.0], [0.0, -3.0]])
    target = torch.tensor([[4.0], [1.0], [-2.0], [3.0]])
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1]]))
        model.bias.fill_(0.3)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    expected = []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.7)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        (model(x) - target).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()
        expected.append({key: value.clone() for key, value in model.state_dict().items()})

    model.load_state_dict(initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.7)
    normalizer = _CudaGraphNormalization(2)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        for state in expected:
            normalizer.update(x[rank::2])
            loss = (model(x[rank::2]) - target[rank::2]).square().mean()
            _optimize_step([optimizer], loss, list(model.parameters()), max_norm=0.4, clip_value=0.1)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key])
        torch.testing.assert_close(normalizer.mean, x.mean(0))
        torch.testing.assert_close(normalizer.std, x.std(0, unbiased=False))
        assert normalizer.count == 5 * len(x)
    finally:
        dist.destroy_process_group()


def test_distributed_updates_match_global_batch(tmp_path: Path):
    """Catch independent rank updates and clipping before gradient averaging."""
    mp.spawn(_compare_updates, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2)
