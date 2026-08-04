# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for distributed training benchmark configuration."""

from __future__ import annotations

import argparse

import pytest

from isaaclab.benchmark._distributed import (
    DistributedContext,
    add_multigpu_benchmark_args,
    build_distributed_metadata,
    global_training_work,
    validate_multigpu_benchmark_args,
)


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


def test_global_training_work_scales_rank_zero_work_by_world_size():
    """Global workload totals should combine identical work from every rank."""
    context = DistributedContext(True, rank=0, local_rank=0, world_size=4, local_world_size=2)

    assert global_training_work(context, num_envs_per_rank=32, steps_per_env=24) == (128, 3072)


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
        "timing_scope": "rank0",
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
