# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resumable resident training for FlashNeRD-scale budgets."""

import pytest

pytest.importorskip("physicsnemo_robotics")

import torch
from isaaclab_physicsnemo.resident_training import ResidentOptions, ResidentWindows, fit_resident
from physicsnemo.distributed import DistributedManager
from physicsnemo_robotics.dynamics import FieldNormalizer
from physicsnemo_robotics.models import ParallelNeRD


class _LastFrameModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def predict_last(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs[:, -1])


class _Preempted(Exception):
    pass


@pytest.fixture(scope="module", autouse=True)
def distributed():
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()


def _problem() -> tuple[ResidentWindows, ResidentWindows, FieldNormalizer]:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(256, 4, 3, generator=generator)
    windows = ResidentWindows(inputs, inputs[:, -1, :2] * 0.5)
    normalizer = FieldNormalizer({"target": torch.zeros(2)}, {"target": torch.ones(2)})
    return windows, windows, normalizer


def _options(**overrides) -> ResidentOptions:
    values = dict(
        max_updates=200,
        batch_size=32,
        learning_rate=5.0e-2,
        final_learning_rate=5.0e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        seed=3,
        log_every=1000,
        mixed_precision=False,
        checkpoint_every=100,
        validate_every=0,
    )
    return ResidentOptions(**(values | overrides))


def test_preempted_training_resumes_from_its_last_checkpoint(tmp_path):
    """A run stopped after a checkpoint continues from it instead of restarting, as preempted cluster jobs do."""
    train, validation, normalizer = _problem()
    checkpoint = tmp_path / "resident.pt"
    first = _LastFrameModel()

    def preempt(updates: int) -> None:
        if updates == 100:
            raise _Preempted

    with pytest.raises(_Preempted):
        fit_resident(
            first,
            train,
            validation,
            normalizer,
            _options(),
            device=torch.device("cpu"),
            checkpoint_path=checkpoint,
            identity={"run": 1},
            on_checkpoint=preempt,
        )
    resumed_checkpoints = []
    updates, metrics = fit_resident(
        _LastFrameModel(),
        train,
        validation,
        normalizer,
        _options(),
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
        identity={"run": 1},
        on_checkpoint=resumed_checkpoints.append,
    )

    # Only the remaining half ran: a restart from scratch would checkpoint at update 100 again.
    assert resumed_checkpoints == [200]
    assert updates == 200
    assert metrics["normalized_delta_rmse"] < 0.05


def test_checkpoint_of_another_run_is_not_resumed(tmp_path):
    """Changing the run configuration must not silently continue an unrelated optimization."""
    train, validation, normalizer = _problem()
    checkpoint = tmp_path / "resident.pt"
    fit_resident(
        _LastFrameModel(),
        train,
        validation,
        normalizer,
        _options(max_updates=100),
        device=torch.device("cpu"),
        checkpoint_path=checkpoint,
        identity={"run": 1},
    )

    with pytest.raises(ValueError, match="different run"):
        fit_resident(
            _LastFrameModel(),
            train,
            validation,
            normalizer,
            _options(),
            device=torch.device("cpu"),
            checkpoint_path=checkpoint,
            identity={"run": 2},
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP16 autocast requires CUDA")
def test_mixed_precision_survives_attention_logits_beyond_fp16_range(tmp_path):
    """Trained ParallelNeRD attention logits exceed 65,504, which FP16 autocast turns into NaN losses."""
    device = torch.device("cuda")
    network = ParallelNeRD(
        3, 2, temporal_attention_spans=[2], num_heads=1, head_dim=32, intermediate_size=32, head_hidden=32
    ).to(device)
    block = network.layers[0]
    with torch.no_grad():
        # Query and key rows large enough that their dot products overflow FP16.
        block.in_projection.weight[: block.hidden_size + block.head_dim] = 60.0
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(64, 2, 3, generator=generator).to(device)
    windows = ResidentWindows(inputs.half(), torch.zeros(64, 2, device=device))
    normalizer = FieldNormalizer({"target": torch.zeros(2)}, {"target": torch.ones(2)})

    updates, metrics = fit_resident(
        network,
        windows,
        windows,
        normalizer,
        _options(max_updates=3, batch_size=16, mixed_precision=True, checkpoint_every=0),
        device=device,
        checkpoint_path=tmp_path / "resident.pt",
        identity={"run": 1},
    )

    assert updates == 3
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
