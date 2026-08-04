# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private distributed helpers for multi-GPU training benchmarks."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from isaaclab.benchmark.schema import StartupTime


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
        if context.world_size < 1 or context.local_world_size < 1:
            raise ValueError("WORLD_SIZE and LOCAL_WORLD_SIZE must be positive")
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


@dataclass(frozen=True)
class LocalTrainingTiming:
    """Numeric training measurements contributed by one worker rank."""

    startup_time_s: StartupTime
    iteration_times_s: tuple[float, ...]
    collection_times_s: tuple[float, ...]
    environment_step_times_s: tuple[float, ...]
    simulation_step_times_s: tuple[float, ...] | None
    simulation_step_calls: int | None
    num_envs: int
    steps_per_iteration: int


@dataclass(frozen=True)
class AggregatedTrainingTiming(LocalTrainingTiming):
    """Critical-path timings and global work totals across all ranks."""

    collection_fps: tuple[float, ...]
    total_fps: tuple[float, ...]


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


def _collective_device(context: DistributedContext):
    import torch
    import torch.distributed as dist

    backend = str(dist.get_backend()).lower()
    return torch.device("cuda", torch.cuda.current_device()) if "nccl" in backend else torch.device("cpu")


def _validate_collective_context(context: DistributedContext) -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed benchmark aggregation requires an initialized torch process group")
    if dist.get_rank() != context.rank or dist.get_world_size() != context.world_size:
        raise ValueError(
            "Torch process-group rank metadata does not match the distributed benchmark environment: "
            f"group=({dist.get_rank()}, {dist.get_world_size()}), "
            f"environment=({context.rank}, {context.world_size})"
        )


def _equal_int(value: int, name: str, device) -> int:
    import torch
    import torch.distributed as dist

    local = torch.tensor(value, dtype=torch.int64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    observed = [int(item.item()) for item in gathered]
    if len(set(observed)) != 1:
        details = ", ".join(f"rank {rank}: {rank_value}" for rank, rank_value in enumerate(observed))
        raise ValueError(f"{name} must match across ranks; observed {details}")
    return observed[0]


def _sum_int(value: int, device) -> int:
    import torch
    import torch.distributed as dist

    total = torch.tensor(value, dtype=torch.int64, device=device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return int(total.item())


def _max_series(values: tuple[float, ...], name: str, device) -> tuple[float, ...]:
    import torch
    import torch.distributed as dist

    _equal_int(len(values), f"{name} length", device)
    if not values:
        return ()
    reduced = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return tuple(float(value) for value in reduced.cpu().tolist())


def _max_optional_float(value: float | None, name: str, device) -> float | None:
    import torch
    import torch.distributed as dist

    present = _equal_int(int(value is not None), f"{name} presence", device)
    if not present:
        return None
    reduced = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return float(reduced.item())


def _max_startup(startup: StartupTime, device) -> StartupTime:
    return StartupTime(
        app_launch=float(_max_optional_float(startup.app_launch, "startup app_launch", device)),
        env_creation=float(_max_optional_float(startup.env_creation, "startup env_creation", device)),
        first_step=float(_max_optional_float(startup.first_step, "startup first_step", device)),
        python_imports=_max_optional_float(startup.python_imports, "startup python_imports", device),
        task_config=_max_optional_float(startup.task_config, "startup task_config", device),
    )


def _local_aggregate(local: LocalTrainingTiming) -> AggregatedTrainingTiming:
    return AggregatedTrainingTiming(
        **local.__dict__,
        collection_fps=tuple(local.steps_per_iteration / value for value in local.collection_times_s),
        total_fps=tuple(local.steps_per_iteration / value for value in local.iteration_times_s),
    )


def aggregate_training_timing(
    local: LocalTrainingTiming, context: DistributedContext
) -> AggregatedTrainingTiming:
    """Reduce rank-local timing samples into global-work critical-path metrics."""
    if not context.enabled:
        return _local_aggregate(local)
    _validate_collective_context(context)
    device = _collective_device(context)

    iteration_times = _max_series(local.iteration_times_s, "iteration_times_s", device)
    collection_times = _max_series(local.collection_times_s, "collection_times_s", device)
    environment_step_times = _max_series(local.environment_step_times_s, "environment_step_times_s", device)

    simulation_present = _equal_int(
        int(local.simulation_step_times_s is not None), "simulation_step_times_s presence", device
    )
    simulation_step_times = (
        _max_series(local.simulation_step_times_s or (), "simulation_step_times_s", device)
        if simulation_present
        else None
    )
    simulation_calls_present = _equal_int(
        int(local.simulation_step_calls is not None), "simulation_step_calls presence", device
    )
    simulation_step_calls = (
        _equal_int(int(local.simulation_step_calls), "simulation_step_calls", device)
        if simulation_calls_present
        else None
    )
    if bool(simulation_present) != bool(simulation_calls_present):
        raise ValueError("simulation_step_times_s and simulation_step_calls must either both be present or both absent")

    num_envs = _sum_int(local.num_envs, device)
    steps_per_iteration = _sum_int(local.steps_per_iteration, device)
    if any(value <= 0 for value in (*iteration_times, *collection_times)):
        raise ValueError("Distributed iteration and collection timings must be positive")

    return AggregatedTrainingTiming(
        startup_time_s=_max_startup(local.startup_time_s, device),
        iteration_times_s=iteration_times,
        collection_times_s=collection_times,
        environment_step_times_s=environment_step_times,
        simulation_step_times_s=simulation_step_times,
        simulation_step_calls=simulation_step_calls,
        num_envs=num_envs,
        steps_per_iteration=steps_per_iteration,
        collection_fps=tuple(steps_per_iteration / value for value in collection_times),
        total_fps=tuple(steps_per_iteration / value for value in iteration_times),
    )
