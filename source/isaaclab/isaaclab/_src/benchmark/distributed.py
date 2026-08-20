# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rank bookkeeping shared by the multi-GPU benchmark workflows.

Every rank of a multi-GPU benchmark runs the ordinary single-process workflow. This module
supplies the pieces that differ: which rank owns the emitted bundle, how per-rank work adds
up to the global workload, and what the bundle records about the distributed topology.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    """Rank metadata of one benchmark worker process.

    A non-distributed run is represented as a single main rank so callers can use the same
    context unconditionally.

    Args:
        enabled: Whether the run was launched with ``--distributed``.
        rank: Global rank of this process.
        local_rank: Rank of this process within its node.
        world_size: Total number of ranks across all nodes.
        local_world_size: Number of ranks on this node.
    """

    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    local_world_size: int = 1

    @classmethod
    def from_env(cls, enabled: bool, *, workflow: str = "training") -> DistributedContext:
        """Build rank metadata from the environment populated by ``torchrun``.

        Args:
            enabled: Whether ``--distributed`` was requested.
            workflow: Benchmark workflow name, used in the error message that names the
                launcher command.

        Returns:
            Rank metadata for this process.

        Raises:
            ValueError: If ``--distributed`` was requested outside a distributed launcher, or the
                launcher exported an inconsistent rank layout.
        """
        if not enabled:
            return cls()
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise ValueError(
                "--distributed was requested but no distributed launcher exported RANK and WORLD_SIZE."
                f" Launch the benchmark with `isaaclab benchmark {workflow}-multigpu`."
            )
        return cls(
            enabled=True,
            rank=int(os.environ["RANK"]),
            local_rank=int(os.getenv("LOCAL_RANK", "0")),
            world_size=int(os.environ["WORLD_SIZE"]),
            local_world_size=int(os.getenv("LOCAL_WORLD_SIZE", "1")),
        )

    def __post_init__(self) -> None:
        """Check that the rank layout is internally consistent.

        Raises:
            ValueError: If the ranks, world sizes, or node layout disagree.
        """
        if self.world_size < 1 or self.local_world_size < 1:
            raise ValueError(f"world_size ({self.world_size}) and local_world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank ({self.rank}) must be in [0, world_size={self.world_size})")
        if not 0 <= self.local_rank < self.local_world_size:
            raise ValueError(f"local_rank ({self.local_rank}) must be in [0, local_world_size={self.local_world_size})")
        if self.world_size % self.local_world_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by local_world_size ({self.local_world_size});"
                " benchmarking a job with unevenly sized nodes is not supported"
            )

    @property
    def is_main(self) -> bool:
        """Whether this process is global rank 0, which owns the emitted bundle."""
        return self.rank == 0

    @property
    def num_nodes(self) -> int:
        """Number of uniformly sized worker nodes."""
        return self.world_size // self.local_world_size

    def global_work(self, num_envs_per_rank: int, steps_per_env: int) -> tuple[int, int]:
        """Scale one rank's workload to the synchronized global workload.

        Ranks of a synchronized training job step the same number of environments for the same
        number of steps between policy updates, so the global totals are exact multiples of one
        rank's totals.

        Args:
            num_envs_per_rank: Number of parallel environments hosted by each rank.
            steps_per_env: Number of environment steps collected per environment per iteration.

        Returns:
            Global environment count and global environment steps per iteration.
        """
        num_envs = num_envs_per_rank * self.world_size
        return num_envs, num_envs * steps_per_env

    def bundle_metadata(
        self, *, workload_scope: str, num_envs_per_rank: int | None = None
    ) -> dict[str, bool | int | str]:
        """Build the ``extra`` bundle entries describing the distributed run.

        Only global rank 0 emits a bundle, so every measurement in it is rank-0 scoped except the
        workload counts and throughput, which are global when the ranks are synchronized. The
        scope entries record that split so a reader never has to infer it.

        Args:
            workload_scope: ``"global"`` when ranks step in lockstep and per-rank work sums exactly
                (synchronized training), or ``"rank0"`` when each rank runs an independent workload.
            num_envs_per_rank: Number of parallel environments hosted by each rank, or ``None`` for
                workflows that do not report an environment count.

        Returns:
            Topology and measurement-scope metadata.
        """
        if workload_scope not in ("global", "rank0"):
            raise ValueError(f"workload_scope must be 'global' or 'rank0', got {workload_scope!r}")
        metadata: dict[str, bool | int | str] = {
            "distributed": True,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "num_nodes": self.num_nodes,
            "workload_scope": workload_scope,
            # Timings, learning curves, CPU, and RAM come from the rank-0 process alone; the GPU
            # recorder samples every device visible to it, which is the whole rank-0 node.
            "measurement_scope": "rank0_process",
            "gpu_measurement_scope": "rank0_node",
        }
        if num_envs_per_rank is not None:
            metadata["num_envs_per_rank"] = num_envs_per_rank
        return metadata


def add_distributed_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--distributed`` flag to a workflow that has no reinforcement-learning parser.

    The training benchmark adapters already receive the flag from
    :func:`isaaclab_rl.entrypoints.common.add_common_train_args`.

    Args:
        parser: Parser to extend.
    """
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help=(
            "Run as one rank of a distributed launch. Set automatically by `isaaclab benchmark <workflow>-multigpu`."
        ),
    )


def validate_distributed_args(parser: argparse.ArgumentParser, args_cli: argparse.Namespace) -> None:
    """Reject training benchmark options that a distributed run cannot honor.

    Recording video, capturing sensor frames, and stopping early on success all act on one rank's
    environments, which neither describes the job nor stays in step with the other ranks.

    Args:
        parser: Parser used to report the error.
        args_cli: Parsed benchmark arguments.
    """
    if not getattr(args_cli, "distributed", False):
        return
    if getattr(args_cli, "video", False):
        parser.error("Video recording is not supported by multi-GPU training benchmarks.")
    if getattr(args_cli, "capture_env_sensors", 0) > 0:
        parser.error("Environment sensor capture is not supported by multi-GPU training benchmarks.")
    if getattr(args_cli, "check_success", False):
        parser.error("Success-based early stopping is not supported by multi-GPU training benchmarks.")
