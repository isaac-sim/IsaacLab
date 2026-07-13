# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Perf-smoke-test runtime benchmark driver (random actions, no policy).

Thin wrapper over the merged Isaac Lab benchmark core (benchmark refactor
Part 1/5, PR #6197) that produces a schema-v1
:class:`~isaaclab.test.benchmark.schema.RuntimeBundle`.  It intentionally does
**not** depend on the still-unmerged ``scripts/benchmarks/runtime.py`` (Part 2/5,
PR #6198); it only imports the stable, merged building blocks
(:mod:`~isaaclab.test.benchmark.stepping`, :mod:`~isaaclab.test.benchmark.builders`,
:mod:`~isaaclab.test.benchmark.capture`) so the perf gate can adopt the typed
bundle schema before the rest of the refactor lands.

Difference from the upstream runtime script: the perf gate discards a
configurable number of leading **warmup** steps *before* aggregation, so the
reported ``total_fps`` is steady-state.  This replaces the gate's previous
post-hoc ``excluded_frames`` mechanism (which required the raw per-frame series
the bundle schema deliberately drops); handling warmup at the source keeps the
aggregate directly comparable to the pre-migration steady-state mean without
serialising the raw series.

Usage example::

    ./isaaclab.sh -p tools/perf_smoke_test/perf_runtime.py \
        --task Isaac-Cartpole-Direct \
        --num_envs 4096 --num_frames 300 --warmup_frames 100 \
        --benchmark_formatter schema \
        --output_path /tmp/bench_out \
        presets=newton_mjwarp --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import time

from benchmark_result_adapter import steady_state_slice

from isaaclab.app import AppLauncher

from isaaclab_tasks.utils import setup_preset_cli

# --- argument parsing -------------------------------------------------------
parser = argparse.ArgumentParser(description="Benchmark environment runtime (random actions, no policy).")
parser.add_argument("--task", type=str, required=True, help="Gym task id to benchmark.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument(
    "--num_frames",
    type=int,
    default=300,
    help="Total number of environment steps to run (including warmup).",
)
parser.add_argument(
    "--warmup_frames",
    type=int,
    default=0,
    help="Number of leading steps to discard before aggregation (steady-state warmup exclusion).",
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="schema",
    help=(
        "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
        " 'omniperf', 'osmo', 'json', 'summary'. Example: 'schema,omniperf'."
    ),
)

# append AppLauncher cli args and resolve Hydra preset tokens
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

# --- heavy imports (after CLI parse, before app launch is measured) ---------
imports_time_begin = time.perf_counter_ns()

import contextlib

import gymnasium as gym

from isaaclab.app import launch_simulation
from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, stepping
from isaaclab.test.benchmark.schema import StartupTime

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

imports_time_end = time.perf_counter_ns()


def main(env_cfg, app_start_time_begin: int, app_start_time_end: int) -> None:
    """Run the runtime benchmark and write the selected formatter outputs.

    Args:
        env_cfg: Resolved environment configuration for :attr:`args_cli.task`.
        app_start_time_begin: ``time.perf_counter_ns()`` sampled just before the
            simulation app launch.
        app_start_time_end: ``time.perf_counter_ns()`` sampled just after the
            simulation app launch.
    """
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    formatter_types = [value.strip() for value in args_cli.benchmark_formatter.split(",") if value.strip()]
    formatter_types = formatter_types or ["schema"]

    # RunConfig (physics/rendering/presets) is derived from the Hydra preset
    # tokens and the resolved env cfg, matching the upstream runtime script.
    cfg = capture.run_config_from_presets(hydra_args, env_cfg=env_cfg)

    start_utc = capture.now_utc_iso()

    benchmark = BaseIsaacLabBenchmark(
        benchmark_name="benchmark_runtime",
        formatter_type=args_cli.benchmark_formatter,
        output_path=args_cli.output_path,
        use_recorders=True,
        frametime_recorders=any(t in ("summary", "omniperf") for t in formatter_types),
        output_prefix=f"benchmark_runtime_{args_cli.task}",
        workflow_metadata={
            "metadata": [
                {"name": "task", "data": args_cli.task},
                {"name": "num_envs", "data": args_cli.num_envs},
                {"name": "num_frames", "data": args_cli.num_frames},
                {"name": "warmup_frames", "data": args_cli.warmup_frames},
                {"name": "presets", "data": ",".join(cfg.presets)},
            ]
        },
    )

    # --- create env -------------------------------------------------------
    env_t0 = time.perf_counter_ns()
    with contextlib.closing(gym.make(args_cli.task, cfg=env_cfg)) as env:
        env_t1 = time.perf_counter_ns()

        num_envs = env.unwrapped.num_envs

        # --- step (warmup + measured) with resource monitoring ------------
        with BenchmarkMonitor(benchmark, interval=1.0):
            step_times_s = stepping.run_runtime_loop(env, args_cli.num_frames)

        # Progress marker consumed by subprocess_runner.classify_failure_phase to
        # tell an init-phase failure from a runtime-phase failure (a later crash
        # with this marker present is attributed to runtime). Matches the legacy
        # driver's stdout contract.
        print("Step Frametimes", flush=True)

        benchmark.update_manual_recorders()

        # --- warmup exclusion at source: aggregate only steady-state frames.
        measured_step_times, warmup = steady_state_slice(step_times_s, args_cli.warmup_frames)
        if warmup != args_cli.warmup_frames:
            print(
                f"[perf_runtime] WARNING: warmup_frames={args_cli.warmup_frames} leaves no measured"
                f" frames out of {len(step_times_s)}; clamped to {warmup} to keep >=1 steady-state frame."
            )
        fps = [num_envs / t for t in measured_step_times if t > 0]

        # ``first_step`` is the (cold) first observed step, kept as a startup
        # signal even though it is excluded from the steady-state fps.
        startup = StartupTime(
            app_launch=(app_start_time_end - app_start_time_begin) / 1e9,
            env_creation=(env_t1 - env_t0) / 1e9,
            first_step=(step_times_s[0] if step_times_s else 0.0),
            python_imports=(imports_time_end - imports_time_begin) / 1e9,
        )

        runtime = builders.build_runtime(
            startup_time_s=startup,
            iteration_times_s=measured_step_times,
            collection_fps=fps,
            total_fps=fps,
            steps_per_iteration=num_envs,
        )

        versions = capture.capture_versions(benchmark)
        hardware = capture.capture_hardware(benchmark)
        resources = capture.capture_resources(benchmark)

        end_utc = capture.now_utc_iso()
        stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
        seed = args_cli.seed if args_cli.seed is not None else 0
        run_id = capture.synth_run_id(None, cfg.physics_backend, args_cli.task, seed, stamp)

        run = builders.build_run_identity(
            run_id=run_id,
            framework=None,
            config=cfg,
            task=args_cli.task,
            seed=seed,
            start_utc=start_utc,
            end_utc=end_utc,
            num_envs=num_envs,
        )

        # ``extra`` carries producer-specific scalars the perf gate needs but
        # that are not part of the stable schema contract (num_frames is not a
        # RunIdentity field; warmup_frames is gate-specific). The gate's
        # benchmark_result_adapter reads these; other consumers may ignore them.
        bundle = builders.build_runtime_bundle(
            run=run,
            versions=versions,
            hardware=hardware,
            runtime=runtime,
            resources=resources,
            extra={"num_frames": args_cli.num_frames, "warmup_frames": warmup},
        )

        benchmark.attach_bundle(bundle)
        benchmark._finalize_impl()


if __name__ == "__main__":
    env_cfg, _agent_cfg = resolve_task_config(args_cli.task, None)

    app_start_time_begin = time.perf_counter_ns()
    with launch_simulation(env_cfg, args_cli):
        app_start_time_end = time.perf_counter_ns()
        main(env_cfg, app_start_time_begin, app_start_time_end)
