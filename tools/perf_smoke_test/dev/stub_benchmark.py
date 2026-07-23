#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local-testing stub that fakes a ``perf_runtime.py`` run without a GPU/sim.

Emits a schema-v1 :class:`~isaaclab.test.benchmark.schema.RuntimeBundle`
(``benchmark_runtime_{task}_{stamp}.json``) identical in shape to what the real
driver writes, so ``build_bench_result.py`` -> ``benchmark_result_adapter`` ->
``oracle`` can be exercised end-to-end offline. Uses the real
``isaaclab.test.benchmark`` builders/serialize (pure-Python, no GPU), so it stays
in lockstep with the schema; run it with the Isaac Lab Python env.
"""

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent.parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from backend_identity import split_backend_key  # noqa: E402

from isaaclab.test.benchmark import builders, serialize  # noqa: E402
from isaaclab.test.benchmark.schema import (  # noqa: E402
    GpuDeviceInfo,
    Hardware,
    MeanStd,
    Resources,
    StartupTime,
    Versions,
)


def _stub_versions() -> Versions:
    """Versions carrying every field the runtime-compatibility contract references."""
    return Versions(
        isaaclab="0.0.0-stub",
        isaacsim="0.0.0-stub",
        kit=None,
        newton="0.0.0-stub",
        warp="0.0.0-stub",
        mjwarp=None,
        torch="0.0.0-stub",
        rsl_rl=None,
        rl_games=None,
        skrl=None,
        sb3=None,
        git_commit="stubcommit",
        git_branch="stub",
        git_dirty=False,
        isaaclab_physx="0.0.0-stub",
        isaaclab_newton="0.0.0-stub",
        isaaclab_ov="0.0.0-stub",
        cuda_bindings="0.0-stub",
    )


def _stub_hardware() -> Hardware:
    return Hardware(
        hostname="stub-host",
        gpu_devices=[GpuDeviceInfo(name="NVIDIA L40S", mem_gb=44.9, compute_cap="8.9")],
        cpu_name="Stub CPU",
        cpu_count=8,
        ram_gb=64.0,
    )


def _stub_resources() -> Resources:
    return Resources(
        gpu_util_pct=MeanStd(mean=90.0, std=1.0, peak=None),
        gpu_mem_gb=MeanStd(mean=8.0, std=0.2, peak=8.5),
        cpu_util_pct=MeanStd(mean=30.0, std=2.0, peak=None),
        ram_gb=MeanStd(mean=10.0, std=0.5, peak=11.0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--warmup_frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps_mean", type=float, default=200.0, help="Target per-env-step effective FPS.")
    parser.add_argument("--failure_phase", default="none", choices=["none", "import", "init", "runtime"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- simulated pre-output failures (no bundle written) ---
    if args.failure_phase == "import":
        print("Traceback (most recent call last):")
        print('  File "<string>", line 1, in <module>')
        print("ImportError: simulated import failure")
        sys.exit(1)
    if args.failure_phase == "init":
        print("AppLauncher initialization complete")
        sys.exit(2)

    identity = split_backend_key(args.backend)
    if identity is None:
        raise RuntimeError(f"Cannot parse backend identity from {args.backend!r}")

    # --- synthesize a steady-state fps series and aggregate it like the driver ---
    n = max(1, int(args.num_frames))
    rng = random.Random(args.seed)
    fps_series = [max(0.0, rng.gauss(args.fps_mean, 0.02 * args.fps_mean)) for _ in range(n)]
    step_times = [args.num_envs / f for f in fps_series if f > 0]

    startup = StartupTime(app_launch=2.0, env_creation=1.0, first_step=0.3, python_imports=1.5)
    runtime = builders.build_runtime(
        startup_time_s=startup,
        iteration_times_s=step_times,
        collection_fps=fps_series,
        total_fps=fps_series,
        steps_per_iteration=args.num_envs,
        aggregate_throughput=True,
    )
    cfg = builders.build_run_config(
        physics_backend=identity.physics_backend,
        rendering_backend=identity.render_backend or "none",
        presets=[],
    )
    start_utc = datetime.now(timezone.utc).isoformat()
    run = builders.build_run_identity(
        run_id=f"stub-{args.task_id}-{identity.backend_key}",
        framework=None,
        config=cfg,
        task=args.task_id,
        seed=args.seed,
        start_utc=start_utc,
        end_utc=start_utc,
        num_envs=args.num_envs,
    )
    bundle = builders.build_runtime_bundle(
        run=run,
        versions=_stub_versions(),
        hardware=_stub_hardware(),
        runtime=runtime,
        resources=_stub_resources(),
        extra={"num_frames": args.num_frames, "warmup_frames": args.warmup_frames},
    )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"benchmark_runtime_{args.task_id}_{stamp}.json"
    serialize.write_bundle_file(bundle, str(out_path))

    # Progress marker consumed by subprocess_runner.classify_failure_phase (matches
    # perf_runtime.py's stdout contract) so a runtime crash is not misread as init.
    print("Step Frametimes", flush=True)

    if args.failure_phase == "runtime":
        print("RuntimeError: simulated crash during runtime")
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
