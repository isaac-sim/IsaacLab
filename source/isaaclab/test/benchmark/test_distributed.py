# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the rank bookkeeping shared by the multi-GPU benchmark workflows."""

from __future__ import annotations

import argparse

import pytest

from isaaclab._src.benchmark.distributed import DistributedContext, add_distributed_arg, validate_distributed_args


def test_context_is_single_rank_without_the_distributed_flag(monkeypatch: pytest.MonkeyPatch):
    """A single-process benchmark stays rank 0 even inside an unrelated distributed environment."""
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("WORLD_SIZE", "8")

    context = DistributedContext.from_env(enabled=False)

    assert context == DistributedContext()
    assert context.is_main
    assert context.num_nodes == 1


def test_context_reads_the_launcher_rank_environment(monkeypatch: pytest.MonkeyPatch):
    """Rank metadata comes from the variables torchrun exports."""
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    context = DistributedContext.from_env(enabled=True)

    assert context == DistributedContext(enabled=True, rank=5, local_rank=1, world_size=8, local_world_size=4)
    assert not context.is_main
    assert context.num_nodes == 2


def test_context_names_the_launcher_when_run_outside_one(monkeypatch: pytest.MonkeyPatch):
    """Passing --distributed by hand points the user at the command that sets up the ranks."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(ValueError, match="isaaclab benchmark runtime-multigpu"):
        DistributedContext.from_env(enabled=True, workflow="runtime")


def test_context_rejects_unevenly_sized_nodes(monkeypatch: pytest.MonkeyPatch):
    """An unsupported layout fails instead of reporting a wrong node count."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    with pytest.raises(ValueError, match="divisible"):
        DistributedContext.from_env(enabled=True)


def test_global_work_sums_identical_per_rank_workloads():
    """Synchronized ranks step the same workload, so the global totals are exact multiples."""
    context = DistributedContext(enabled=True, world_size=4, local_world_size=2)

    assert context.global_work(num_envs_per_rank=32, steps_per_env=24) == (128, 3072)


def test_bundle_metadata_records_the_scope_of_each_measurement():
    """The bundle states which measurements are global and which describe rank 0 alone."""
    context = DistributedContext(enabled=True, world_size=4, local_world_size=2)

    assert context.bundle_metadata(workload_scope="global", num_envs_per_rank=32) == {
        "distributed": True,
        "world_size": 4,
        "local_world_size": 2,
        "num_nodes": 2,
        "workload_scope": "global",
        "measurement_scope": "rank0_process",
        "gpu_measurement_scope": "rank0_node",
        "num_envs_per_rank": 32,
    }


def _parser() -> argparse.ArgumentParser:
    """Build a parser carrying the benchmark options that distributed runs restrict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--capture_env_sensors", type=int, default=0)
    parser.add_argument("--check_success", action="store_true")
    add_distributed_arg(parser)
    return parser


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--distributed", "--video"], "Video recording"),
        (["--distributed", "--capture_env_sensors", "1"], "sensor capture"),
        (["--distributed", "--check_success"], "early stopping"),
    ],
)
def test_unsupported_options_are_rejected_for_distributed_runs(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
):
    """Rank-unsafe benchmark features fail before any worker starts training."""
    parser = _parser()

    with pytest.raises(SystemExit) as exc_info:
        validate_distributed_args(parser, parser.parse_args(argv))

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["--video"], ["--distributed"], ["--distributed", "--capture_env_sensors", "0"]])
def test_supported_option_combinations_pass_validation(argv: list[str]):
    """Single-process runs keep every feature, and distributed runs keep the rank-safe ones."""
    parser = _parser()

    validate_distributed_args(parser, parser.parse_args(argv))
