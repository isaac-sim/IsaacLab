# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Benchmark environment runtime (random actions, no policy).

Standalone script that steps an Isaac Lab environment with random actions. The
schema formatter emits a :class:`~isaaclab.test.benchmark.RuntimeBundle`;
other selected formatters receive equivalent measurement phases.
Supports all physics backends (PhysX, Newton/MJWarp, Newton/Kamino, OVPhysX)
via Hydra preset tokens — no ``--rl_library`` dispatch needed.

Usage example::

    uv run isaaclab benchmark runtime \\
        --task Isaac-Cartpole-Direct \\
        --num_envs 16 --num_frames 1000 --warmup_frames 50 \\
        presets=newton_mjwarp --headless
"""

from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse benchmark arguments and retain Hydra overrides in ``sys.argv``.

    Args:
        argv: Command-line arguments excluding the script path.

    Returns:
        Parsed arguments and the remaining Hydra overrides.
    """
    from isaaclab.app import add_launcher_args
    from isaaclab.test.benchmark._cli import parse_non_negative_int, parse_positive_int

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Benchmark environment runtime (random actions, no policy).")
    parser.add_argument("--task", type=str, required=True, help="Gym task id to benchmark.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument(
        "--num_frames", type=parse_positive_int, default=1000, help="Number of environment steps to benchmark."
    )
    parser.add_argument(
        "--warmup_frames",
        type=parse_non_negative_int,
        default=50,
        help="Exact number of environment steps to exclude from timing; zero measures the first step.",
    )
    parser.add_argument(
        "--measure_sync_step",
        action="store_true",
        help="Measure a serialized synchronized simulation and outside-simulation step breakdown.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--benchmark_formatter",
        type=str,
        default="schema",
        help=(
            "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
            " 'omniperf', 'osmo', 'json', 'summary'."
            " Example: 'schema,omniperf'."
        ),
    )
    add_launcher_args(parser)

    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + remaining
    return args, remaining


def run(argv: list[str]) -> None:
    """Run the runtime benchmark and write the selected formatter outputs.

    Args:
        argv: Command-line arguments excluding the script path.
    """
    import time

    imports_t0 = time.perf_counter_ns()
    import contextlib

    import gymnasium as gym

    from isaaclab.app import launch_simulation
    from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, stepping
    from isaaclab.test.benchmark.schema import StartupTime

    # Importing the task packages registers their gym environments so the
    # requested ``--task`` can be resolved.
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    # PLACEHOLDER: Extension template (do not remove this comment)
    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    args, remaining = _parse_args(argv)
    imports_t1 = time.perf_counter_ns()

    task_config_t0 = time.perf_counter_ns()
    env_cfg, _ = resolve_task_config(args.task, None)
    task_config_t1 = time.perf_counter_ns()

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args):
        app_t1 = time.perf_counter_ns()

        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if args.device is not None:
            env_cfg.sim.device = args.device
        if args.seed is not None:
            env_cfg.seed = args.seed

        formatter_types = [value.strip() for value in args.benchmark_formatter.split(",") if value.strip()]
        formatter_types = formatter_types or ["omniperf"]
        cfg = capture.run_config_from_presets(remaining, env_cfg=env_cfg)

        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_runtime",
            formatter_type=args.benchmark_formatter,
            output_path=args.output_path,
            use_recorders=True,
            frametime_recorders=any(t in ("summary", "omniperf") for t in formatter_types),
            output_prefix=f"benchmark_runtime_{args.task}",
            workflow_metadata={
                "metadata": [
                    {"name": "task", "data": args.task},
                    {"name": "num_envs", "data": args.num_envs},
                    {"name": "num_frames", "data": args.num_frames},
                    {"name": "warmup_frames", "data": args.warmup_frames},
                    {
                        "name": "environment_step_measurement_mode",
                        "data": ("serialized_synchronized" if args.measure_sync_step else "host_return"),
                    },
                    {"name": "presets", "data": ",".join(cfg.presets)},
                ]
            },
        )

        env_t0 = time.perf_counter_ns()
        with contextlib.closing(gym.make(args.task, cfg=env_cfg)) as env:
            env_t1 = time.perf_counter_ns()

            num_envs = env.unwrapped.num_envs

            warmup_step_times_s = stepping.run_runtime_warmup(env, args.warmup_frames)
            environment_step_timer = stepping.EnvironmentStepTimingRecorder(
                env, measure_synchronized_step_breakdown=args.measure_sync_step
            )
            with environment_step_timer, BenchmarkMonitor(benchmark, interval=1.0):
                step_times_s = stepping.run_runtime_loop(env, args.num_frames, reset=False)

            first_step_s = warmup_step_times_s[0] if warmup_step_times_s else step_times_s[0]

            benchmark.update_manual_recorders()

            startup = StartupTime(
                app_launch=(app_t1 - app_t0) / 1e9,
                env_creation=(env_t1 - env_t0) / 1e9,
                first_step=first_step_s,
                python_imports=(imports_t1 - imports_t0) / 1e9,
                task_config=(task_config_t1 - task_config_t0) / 1e9,
            )

            fps = [num_envs / t for t in step_times_s if t > 0]
            runtime = builders.build_runtime(
                startup_time_s=startup,
                iteration_times_s=step_times_s,
                collection_fps=fps,
                total_fps=fps,
                steps_per_iteration=num_envs,
                frames_per_environment_step=env.unwrapped.num_envs,
                aggregate_throughput=True,
                environment_step_times_s=environment_step_timer.step_times_s,
                simulation_step_times_s=environment_step_timer.simulation_step_times_s,
                simulation_step_calls=environment_step_timer.simulation_step_calls,
            )

            versions = capture.capture_versions(benchmark)
            hardware = capture.capture_hardware(benchmark)
            resources = capture.capture_resources(benchmark)

            end_utc = capture.now_utc_iso()
            stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]

            seed = args.seed if args.seed is not None else 0
            run_id = capture.synth_run_id(None, cfg.physics_backend, args.task, seed, stamp)

            run = builders.build_run_identity(
                run_id=run_id,
                framework=None,
                config=cfg,
                task=args.task,
                seed=seed,
                start_utc=start_utc,
                end_utc=end_utc,
                num_envs=num_envs,
            )

            bundle = builders.build_runtime_bundle(
                run=run,
                versions=versions,
                hardware=hardware,
                runtime=runtime,
                resources=resources,
                extra={"warmup_frames": args.warmup_frames},
            )

            benchmark.attach_bundle(bundle)

            benchmark._finalize_impl()


if __name__ == "__main__":
    run(sys.argv[1:])
