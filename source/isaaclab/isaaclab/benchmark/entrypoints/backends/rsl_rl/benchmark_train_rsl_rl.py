# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL adapter for the unified training benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.benchmark import BenchmarkResult

import sys
import time
from typing import Any

from isaaclab_rl.entrypoints import common as _common


def _disable_code_state_capture(runner: Any) -> None:
    """Disable RSL-RL Git state capture while retaining TensorBoard logging."""
    runner.logger.git_status_repos = []


def _parse_args(argv: list[str]):
    """Parse CLI arguments and forward the remaining Hydra preset tokens via ``sys.argv``.

    Args:
        argv: Raw command-line arguments (``sys.argv[1:]`` after dispatcher strips
            ``--rl_library``).

    Returns:
        Tuple of ``(parsed_args, remaining)`` where *remaining* are the verbatim Hydra
        preset tokens written back to ``sys.argv`` for ``launch_simulation`` to pick up.
    """
    import argparse

    from isaaclab.app import add_launcher_args
    from isaaclab.benchmark._cli import parse_non_negative_int, parse_positive_int

    from isaaclab_tasks.utils import setup_preset_cli

    add_common_train_args = _common.add_common_train_args
    enable_cameras_for_video = _common.enable_cameras_for_video
    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args

    parser = argparse.ArgumentParser(description="Benchmark RL training with RSL-RL.")
    add_common_train_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
        include_distributed=False,
        max_iterations_type=parse_positive_int,
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)

    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--measure_sync_step",
        action="store_true",
        help="Measure a serialized synchronized simulation and outside-simulation step breakdown.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=parse_non_negative_int,
        default=1,
        help="Exclude the first N env.step() calls from environment-step timing. Default 1 removes cold start.",
    )
    parser.add_argument(
        "--benchmark_formatter",
        type=str,
        default="schema",
        help=(
            "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
            " 'omniperf', 'osmo', 'json', 'summary'"
            " Example: 'schema,omniperf'."
        ),
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.1,
        help="EMA smoothing factor for learning curves (higher = more recent weight).",
    )
    parser.add_argument(
        "--no_series",
        action="store_true",
        default=False,
        help="Omit per-iteration series data from the bundle to reduce file size.",
    )

    from isaaclab.benchmark.entrypoints.early_stop import add_success_cli_args

    add_success_cli_args(parser)

    if "--distributed" in argv:
        parser.error("Distributed training benchmarks are not supported.")

    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    sys.argv = [sys.argv[0]] + remaining_args

    return args_cli, remaining_args, cli_args


def run(argv: list[str]) -> BenchmarkResult:
    """Run the RSL-RL training benchmark and write a :class:`~isaaclab.benchmark.TrainingBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    imports_t0 = time.perf_counter_ns()

    import contextlib
    import importlib.metadata as metadata
    import os
    from datetime import datetime

    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, BenchmarkResult, builders, capture, stepping
    from isaaclab.benchmark.metrics import RL_LIBRARY_DESCRIPTORS, parse_tf_logs
    from isaaclab.benchmark.schema import StartupTime

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config

    apply_env_overrides = _common.apply_env_overrides
    from isaaclab.benchmark.entrypoints.early_stop import (
        RslRlEarlyStopWrapper,
        build_success_kwargs,
        get_success_tracker,
        success_measurements,
    )

    imports_t1 = time.perf_counter_ns()

    args_cli, remaining_args, cli_args = _parse_args(argv)

    config_t0 = time.perf_counter_ns()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    config_t1 = time.perf_counter_ns()

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args_cli):
        with contextlib.ExitStack() as cleanup:
            cleanup.enter_context(
                _common.scoped_torch_backend_flags(
                    cuda_matmul_allow_tf32=True,
                    cudnn_allow_tf32=True,
                    cudnn_deterministic=False,
                    cudnn_benchmark=False,
                )
            )
            app_t1 = time.perf_counter_ns()

            apply_env_overrides(args_cli, env_cfg)
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            agent_cfg.max_iterations = (
                args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
            )
            installed_rsl_rl = metadata.version("rsl-rl-lib")
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl)
            env_cfg.seed = agent_cfg.seed

            cfg = capture.run_config_from_presets(remaining_args, env_cfg=env_cfg)
            formatter_types = [value.strip() for value in args_cli.benchmark_formatter.split(",") if value.strip()]
            formatter_types = formatter_types or ["omniperf"]

            benchmark = BaseIsaacLabBenchmark(
                benchmark_name="benchmark_training",
                formatter_type=formatter_types,
                output_path=args_cli.output_path,
                use_recorders=True,
                frametime_recorders=any(t in ("summary", "omniperf") for t in formatter_types),
                output_prefix=f"benchmark_training_{args_cli.task}",
                workflow_metadata={
                    "metadata": [
                        {"name": "task", "data": args_cli.task},
                        {"name": "seed", "data": agent_cfg.seed},
                        {"name": "num_envs", "data": env_cfg.scene.num_envs},
                        {"name": "max_iterations", "data": agent_cfg.max_iterations},
                        {
                            "name": "environment_step_measurement_mode",
                            "data": ("serialized_synchronized" if args_cli.measure_sync_step else "host_return"),
                        },
                        {"name": "environment_step_warmup_steps", "data": args_cli.warmup_steps},
                        {"name": "presets", "data": ",".join(cfg.presets)},
                    ]
                },
            )

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            resume_path = (
                get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
                if agent_cfg.resume or agent_cfg.class_name == "DistillationRunner"
                else None
            )
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if agent_cfg.run_name:
                log_dir += f"_{agent_cfg.run_name}"
            log_dir = os.path.join(log_root_path, log_dir)
            _common.write_run_manifest(
                log_dir, library="rsl_rl", task=args_cli.task, metadata={"agent": args_cli.agent}
            )
            env_cfg.log_dir = log_dir

            env_t0 = time.perf_counter_ns()
            env = _common.create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=True)
            cleanup.callback(lambda: env.close())
            env = _common.wrap_record_video(env, log_dir, args_cli)
            env_t1 = time.perf_counter_ns()

            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            runner_types = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}
            if agent_cfg.class_name not in runner_types:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            runner = runner_types[agent_cfg.class_name](
                env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
            )
            _disable_code_state_capture(runner)
            if resume_path is not None:
                runner.load(resume_path)

            early = RslRlEarlyStopWrapper(
                env, runner, num_steps_per_env=agent_cfg.num_steps_per_env, **build_success_kwargs(args_cli)
            )

            environment_step_timer = stepping.EnvironmentStepTimingRecorder(
                env,
                measure_synchronized_step_breakdown=args_cli.measure_sync_step,
                warmup_steps=args_cli.warmup_steps,
            )
            with early, environment_step_timer, BenchmarkMonitor(benchmark, interval=1.0):
                runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

            benchmark.update_manual_recorders()

            desc = RL_LIBRARY_DESCRIPTORS["rsl_rl"]
            log_data = parse_tf_logs(log_dir, desc.tfevents_pattern)
            if not log_data or (not log_data.get(desc.reward_tag) and agent_cfg.max_iterations >= 1):
                print(
                    f"[WARNING] No TensorBoard data parsed from {log_dir!r};"
                    " the emitted bundle will report zero metrics. Check the log directory.",
                    file=sys.stderr,
                )

            # RSL-RL reports collection and learning durations separately in seconds.
            coll = log_data.get("Perf/collection_time", [])
            learn_ = log_data.get("Perf/learning_time", [])
            iteration_times_s = [c + lrn for c, lrn in zip(coll, learn_)]
            collection_fps_series = [env.unwrapped.num_envs * agent_cfg.num_steps_per_env / c for c in coll if c > 0]
            total_fps_series = list(log_data.get("Perf/total_fps", []))

            startup = StartupTime(
                app_launch=(app_t1 - app_t0) / 1e9,
                env_creation=(env_t1 - env_t0) / 1e9,
                first_step=(iteration_times_s[0] if iteration_times_s else 0.0),
                python_imports=(imports_t1 - imports_t0) / 1e9,
                task_config=(config_t1 - config_t0) / 1e9,
            )

            runtime = builders.build_runtime(
                startup_time_s=startup,
                iteration_times_s=iteration_times_s,
                collection_fps=collection_fps_series,
                total_fps=total_fps_series,
                steps_per_iteration=env.unwrapped.num_envs * agent_cfg.num_steps_per_env,
                frames_per_environment_step=env.unwrapped.num_envs,
                environment_step_times_s=environment_step_timer.step_times_s,
                simulation_step_times_s=environment_step_timer.simulation_step_times_s,
                simulation_step_calls=environment_step_timer.simulation_step_calls,
            )

            learning = builders.build_learning(
                reward_series=log_data.get(desc.reward_tag, []),
                ep_length_series=log_data.get(desc.ep_length_tag, []),
                ema_alpha=args_cli.ema_alpha,
                keep_series=not args_cli.no_series,
            )

            tracker = get_success_tracker(args_cli, early.tracker, log_data)
            success_rate = round(tracker.tail_mean, 4) if (tracker and tracker.history) else None

            versions = capture.capture_versions(benchmark)
            hardware = capture.capture_hardware(benchmark)
            resources = capture.capture_resources(benchmark)

            end_utc = capture.now_utc_iso()
            stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
            seed = agent_cfg.seed if agent_cfg.seed is not None else 0

            run_identity = builders.build_run_identity(
                run_id=capture.synth_run_id("rsl_rl", cfg.physics_backend, args_cli.task, seed, stamp),
                framework="rsl_rl",
                config=cfg,
                task=args_cli.task,
                seed=seed,
                start_utc=start_utc,
                end_utc=end_utc,
                num_envs=env.unwrapped.num_envs,
                max_iterations=agent_cfg.max_iterations,
            )

            checkpoint_path = None
            video_path = os.path.join(log_dir, "videos") if getattr(args_cli, "video", False) else None

            bundle = builders.build_training_bundle(
                run=run_identity,
                versions=versions,
                hardware=hardware,
                runtime=runtime,
                resources=resources,
                learning=learning,
                success_rate=success_rate,
                checkpoint_path=checkpoint_path,
                video_path=video_path,
            )

            benchmark.attach_bundle(bundle)
            benchmark.add_measurement("train", success_measurements(tracker))

            output_paths = benchmark.finalize()
            result = BenchmarkResult(bundle=bundle, output_paths=output_paths)

    return result


if __name__ == "__main__":
    run(sys.argv[1:])
