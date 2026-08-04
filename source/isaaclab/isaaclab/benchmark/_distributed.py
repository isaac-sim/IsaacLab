# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private distributed helpers for multi-GPU training benchmarks."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    """Torch worker rank metadata for a benchmark process."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int

    @classmethod
    def from_env(cls, enabled: bool) -> DistributedContext:
        """Build rank metadata from the environment populated by torchrun."""
        if not enabled:
            return cls(False, 0, 0, 1, 1)
        context = cls(
            True,
            int(os.getenv("RANK", "0")),
            int(os.getenv("LOCAL_RANK", "0")),
            int(os.getenv("WORLD_SIZE", "1")),
            int(os.getenv("LOCAL_WORLD_SIZE", "1")),
        )
        if context.world_size < 2:
            raise ValueError(f"WORLD_SIZE={context.world_size} must be at least 2")
        if context.local_world_size < 1:
            raise ValueError("LOCAL_WORLD_SIZE must be positive")
        if not 0 <= context.rank < context.world_size:
            raise ValueError(f"RANK={context.rank} must be in [0, WORLD_SIZE={context.world_size})")
        if not 0 <= context.local_rank < context.local_world_size:
            raise ValueError(
                f"LOCAL_RANK={context.local_rank} must be in [0, LOCAL_WORLD_SIZE={context.local_world_size})"
            )
        if context.world_size % context.local_world_size != 0:
            raise ValueError(
                f"WORLD_SIZE={context.world_size} must be divisible by LOCAL_WORLD_SIZE={context.local_world_size}"
            )
        return context

    @property
    def is_main(self) -> bool:
        """Whether this is global rank 0."""
        return self.rank == 0

    @property
    def num_nodes(self) -> int:
        """Number of uniformly sized worker nodes."""
        return self.world_size // self.local_world_size


def build_distributed_metadata(
    context: DistributedContext, num_envs_per_rank: int
) -> dict[str, bool | int | str]:
    """Build bundle metadata for a distributed training benchmark.

    Args:
        context: Distributed worker metadata.
        num_envs_per_rank: Number of environments hosted by each worker.

    Returns:
        Distributed topology and output-scope metadata.
    """
    if not context.enabled:
        raise ValueError("Distributed metadata requires an enabled distributed context")
    return {
        "distributed": True,
        "world_size": context.world_size,
        "local_world_size": context.local_world_size,
        "num_nodes": context.num_nodes,
        "num_envs_per_rank": num_envs_per_rank,
        "learning_scope": "rank0",
        "timing_scope": "rank0",
        "resource_scope_gpu": "rank0_node",
        "resource_scope_cpu": "rank0_process",
        "resource_scope_ram": "rank0_process",
    }


def global_training_work(
    context: DistributedContext, num_envs_per_rank: int, steps_per_env: int
) -> tuple[int, int]:
    """Return global environment and per-iteration frame counts."""
    num_envs = num_envs_per_rank * context.world_size
    return num_envs, num_envs * steps_per_env


def add_multigpu_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add the private marker accepted only from the multi-GPU launcher."""
    parser.add_argument("--benchmark_multigpu", action="store_true", help=argparse.SUPPRESS)


def validate_multigpu_benchmark_args(parser: argparse.ArgumentParser, args_cli: argparse.Namespace) -> None:
    """Validate the dedicated distributed benchmark mode and rank-safe options."""
    distributed = bool(getattr(args_cli, "distributed", False))
    benchmark_multigpu = bool(getattr(args_cli, "benchmark_multigpu", False))
    if distributed and not benchmark_multigpu:
        parser.error("Distributed training benchmarks must use `isaaclab benchmark training_multigpu`.")
    if benchmark_multigpu and not distributed:
        parser.error("--benchmark_multigpu requires --distributed.")
    if not benchmark_multigpu:
        return
    if getattr(args_cli, "video", False):
        parser.error("Video recording is not supported by multi-GPU training benchmarks.")
    if getattr(args_cli, "capture_env_sensors", 0) > 0:
        parser.error("Environment sensor capture is not supported by multi-GPU training benchmarks.")
    if getattr(args_cli, "check_success", False):
        parser.error("Success-based early stopping is not supported by multi-GPU training benchmarks.")
