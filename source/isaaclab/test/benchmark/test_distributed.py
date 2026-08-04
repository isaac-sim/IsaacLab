# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for distributed training benchmark metric aggregation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from isaaclab.benchmark._distributed import (
    AggregatedTrainingTiming,
    DistributedContext,
    LocalTrainingTiming,
    add_multigpu_benchmark_args,
    aggregate_training_timing,
    build_distributed_metadata,
    validate_multigpu_benchmark_args,
)
from isaaclab.benchmark.schema import StartupTime


def _aggregate_worker(rank: int, init_path: str, output_path: str, mismatch: bool) -> None:
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        GLOO_SOCKET_IFNAME="lo",
    )
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        context = DistributedContext.from_env(enabled=True)
        iteration_times = (2.0, 3.0) if rank == 0 else ((1.0,) if mismatch else (1.0, 4.0))
        local = LocalTrainingTiming(
            startup_time_s=StartupTime(
                app_launch=1.0 + rank,
                env_creation=4.0 - rank,
                first_step=0.5 + rank,
                python_imports=0.2 + rank,
                task_config=0.1 + rank,
            ),
            iteration_times_s=iteration_times,
            collection_times_s=(1.5, 2.0) if rank == 0 else (1.0, 3.0),
            environment_step_times_s=(0.3, 0.2) if rank == 0 else (0.2, 0.4),
            simulation_step_times_s=(0.1, 0.25) if rank == 0 else (0.2, 0.3),
            simulation_step_calls=8,
            num_envs=16 if rank == 0 else 32,
            steps_per_iteration=64 if rank == 0 else 128,
        )
        try:
            aggregate = aggregate_training_timing(local, context)
        except ValueError as exc:
            if rank == 0:
                Path(output_path).write_text(json.dumps({"error": str(exc)}))
        else:
            if rank == 0:
                Path(output_path).write_text(json.dumps(asdict(aggregate)))
    finally:
        dist.destroy_process_group()


def _run_aggregate_workers(tmp_path: Path, *, mismatch: bool = False) -> dict:
    init_path = tmp_path / "rendezvous"
    output_path = tmp_path / "result.json"
    mp.spawn(_aggregate_worker, args=(str(init_path), str(output_path), mismatch), nprocs=2, join=True)
    return json.loads(output_path.read_text())


def test_distributed_context_disabled_ignores_environment(monkeypatch: pytest.MonkeyPatch):
    """Regular benchmarks should remain rank zero even under unrelated rank variables."""
    monkeypatch.setenv("RANK", "7")
    context = DistributedContext.from_env(enabled=False)
    assert context == DistributedContext(False, 0, 0, 1, 1)
    assert context.is_main
    assert context.num_nodes == 1


def test_distributed_context_reads_torchrun_environment(monkeypatch: pytest.MonkeyPatch):
    """Distributed metadata should be derived from torchrun's rank environment."""
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    context = DistributedContext.from_env(enabled=True)
    assert context == DistributedContext(True, 5, 1, 8, 4)
    assert not context.is_main
    assert context.num_nodes == 2


def test_distributed_context_rejects_single_worker(monkeypatch: pytest.MonkeyPatch):
    """A distributed benchmark requires at least two global workers."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")

    with pytest.raises(ValueError, match="WORLD_SIZE.*at least 2"):
        DistributedContext.from_env(enabled=True)


def test_distributed_context_rejects_non_divisible_world(monkeypatch: pytest.MonkeyPatch):
    """A heterogeneous process layout should fail instead of reporting the wrong node count."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="WORLD_SIZE.*divisible"):
        DistributedContext.from_env(enabled=True)


def test_nccl_collective_uses_current_cuda_device(monkeypatch: pytest.MonkeyPatch):
    """Aggregation should respect per-process CUDA remapping instead of raw local rank."""
    import torch

    from isaaclab.benchmark import _distributed

    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    context = DistributedContext(True, rank=1, local_rank=1, world_size=2, local_world_size=2)

    assert _distributed._collective_device(context) == torch.device("cuda:0")


def test_aggregate_training_timing_uses_global_work_and_slowest_rank(tmp_path: Path):
    """Concurrent rank work should be summed while critical-path times use the maximum."""
    data = _run_aggregate_workers(tmp_path)
    aggregate = AggregatedTrainingTiming(**{**data, "startup_time_s": StartupTime(**data["startup_time_s"])})

    assert aggregate.num_envs == 48
    assert aggregate.steps_per_iteration == 192
    assert aggregate.iteration_times_s == pytest.approx((2.0, 4.0))
    assert aggregate.collection_times_s == pytest.approx((1.5, 3.0))
    assert aggregate.collection_fps == pytest.approx((128.0, 64.0))
    assert aggregate.total_fps == pytest.approx((96.0, 48.0))
    assert aggregate.environment_step_times_s == pytest.approx((0.3, 0.4))
    assert aggregate.simulation_step_times_s == pytest.approx((0.2, 0.3))
    assert aggregate.simulation_step_calls == 8
    assert aggregate.startup_time_s.app_launch == pytest.approx(2.0)
    assert aggregate.startup_time_s.env_creation == pytest.approx(4.0)


def test_aggregate_training_timing_rejects_mismatched_series_lengths(tmp_path: Path):
    """Rank series must stay aligned rather than being silently truncated."""
    data = _run_aggregate_workers(tmp_path, mismatch=True)
    assert "iteration_times_s" in data["error"]
    assert "length" in data["error"]
    assert "rank 0: 2" in data["error"]
    assert "rank 1: 1" in data["error"]


def test_aggregate_training_timing_rejects_disabled_context():
    """Distributed aggregation must not provide a second local timing path."""
    local = LocalTrainingTiming(
        startup_time_s=StartupTime(1.0, 1.0, 1.0),
        iteration_times_s=(1.0,),
        collection_times_s=(0.5,),
        environment_step_times_s=(0.25,),
        simulation_step_times_s=None,
        simulation_step_calls=None,
        num_envs=16,
        steps_per_iteration=64,
    )

    with pytest.raises(ValueError, match="enabled distributed context"):
        aggregate_training_timing(local, DistributedContext.from_env(enabled=False))


def test_build_distributed_metadata_reports_independent_resource_scopes():
    """Bundle metadata must distinguish node-wide GPU data from process data."""
    context = DistributedContext(True, rank=0, local_rank=0, world_size=4, local_world_size=2)

    assert build_distributed_metadata(context, num_envs_per_rank=32) == {
        "distributed": True,
        "world_size": 4,
        "local_world_size": 2,
        "num_nodes": 2,
        "num_envs_per_rank": 32,
        "learning_scope": "rank0",
        "resource_scope_gpu": "rank0_node",
        "resource_scope_cpu": "rank0_process",
        "resource_scope_ram": "rank0_process",
    }


def _benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--capture_env_sensors", type=int, default=0)
    parser.add_argument("--check_success", action="store_true")
    add_multigpu_benchmark_args(parser)
    return parser


@pytest.mark.parametrize(
    "argv,message",
    [
        (["--distributed"], "training_multigpu"),
        (["--benchmark_multigpu"], "requires --distributed"),
        (["--distributed", "--benchmark_multigpu", "--video"], "Video"),
        (["--distributed", "--benchmark_multigpu", "--capture_env_sensors", "1"], "sensor"),
        (["--distributed", "--benchmark_multigpu", "--check_success"], "early stopping"),
    ],
)
def test_multigpu_benchmark_argument_gating(argv: list[str], message: str, capsys: pytest.CaptureFixture[str]):
    """Only the dedicated launcher and rank-safe features should enter distributed mode."""
    parser = _benchmark_parser()
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit) as exc_info:
        validate_multigpu_benchmark_args(parser, args)
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_multigpu_benchmark_argument_gating_accepts_regular_and_distributed_modes():
    """Regular mode and the complete private distributed pair should both validate."""
    parser = _benchmark_parser()
    validate_multigpu_benchmark_args(parser, parser.parse_args([]))
    validate_multigpu_benchmark_args(parser, parser.parse_args(["--distributed", "--benchmark_multigpu"]))
