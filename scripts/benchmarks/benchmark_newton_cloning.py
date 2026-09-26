# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure CPU cloning of a real task, stopping before model finalization and solver creation.

Run each case in a fresh process with the same Newton version and asset cache::

    uv run python scripts/benchmarks/benchmark_newton_cloning.py --num_envs 16384 --strategy round_robin
    uv run python scripts/benchmarks/benchmark_newton_cloning.py --num_envs 16384 --strategy sequential --serial
    uv run python scripts/benchmarks/benchmark_newton_cloning.py --num_envs 16384 --strategy sequential

The serial control disables only contiguous-run batching. Timings include index and label
construction, but exclude asset import, model finalization, GPU allocation, reset IK and stepping.
The JSON result is printed with a ``CLONING_RESULT`` prefix. This is not end-to-end startup.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import newton

from isaaclab.app import launch_simulation
from isaaclab.cloner import round_robin, sequential

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


class _CloningComplete(BaseException):
    """Stop construction before finalization without treating it as an environment failure."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="IsaacContrib-Keyboard-SO101")
    parser.add_argument("--num_envs", type=int, default=16384)
    parser.add_argument("--strategy", choices=("sequential", "round_robin"), default="sequential")
    parser.add_argument("--serial", action="store_true", help="Disable contiguous-run batching for attribution.")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num_envs must be positive")

    cfg = parse_env_cfg(args.task, device="cpu", num_envs=args.num_envs, overrides=["physics=newton_mjwarp"])
    cfg.seed = 42
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    cfg.scene.clone_cfg.clone_strategy = {"sequential": sequential, "round_robin": round_robin}[args.strategy]
    module = importlib.import_module("isaaclab_newton.cloner.replicate")
    original = module.replicate_builder_mapping
    result = {}

    def measure(*positional, **kwargs):
        if args.serial:
            kwargs.pop("create_builder", None)
        start = time.perf_counter()
        original(*positional, **kwargs)
        elapsed = time.perf_counter() - start
        builder = kwargs["builder"]
        result.update(
            task=args.task,
            num_envs=args.num_envs,
            strategy=args.strategy,
            serial=args.serial,
            newton_distribution_version=importlib.metadata.version("newton"),
            newton_module=newton.__file__,
            cloning_seconds=elapsed,
            source_count=len(kwargs["sources"]),
            world_count=builder.world_count,
            body_count=builder.body_count,
            shape_count=builder.shape_count,
            joint_count=builder.joint_count,
        )
        if builder.world_count != args.num_envs:
            raise RuntimeError(f"Expected {args.num_envs} worlds, got {builder.world_count}.")
        raise _CloningComplete()

    with patch.object(module, "replicate_builder_mapping", measure):
        with launch_simulation(cfg, {"device": "cpu", "headless": True, "visualizer": None}):
            with contextlib.suppress(_CloningComplete):
                gym.make(args.task, cfg=cfg)
    if not result:
        raise RuntimeError("The task did not invoke Newton cloning.")
    for name, root in (
        ("isaaclab", Path(importlib.import_module("isaaclab").__file__).resolve().parents[3]),
        ("newton", Path(newton.__file__).resolve().parents[1]),
    ):
        if (root / ".git").exists():
            result[f"{name}_commit"] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            result[f"{name}_dirty"] = bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
                ).strip()
            )
    print("CLONING_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
