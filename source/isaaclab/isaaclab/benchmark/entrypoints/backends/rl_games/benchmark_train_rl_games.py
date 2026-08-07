# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL-Games adapter for the unified training benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.benchmark import BenchmarkResult

import sys
import time

from isaaclab.benchmark.entrypoints.backends.rl_games.registry import register_scoped_rl_games_environment

from isaaclab_rl.entrypoints import common as _common


def _close_rl_games_writer(observer: Any) -> None:
    """Flush and close an RL-Games observer writer when one was created."""
    writer = getattr(getattr(observer, "algo", None), "writer", None)
    if writer is None:
        return
    try:
        writer.flush()
        writer.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARNING] rl_games TensorBoard writer cleanup failed: {exc!r}", file=sys.stderr)


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

    parser = argparse.ArgumentParser(description="Benchmark RL training with RL-Games.")
    add_common_train_args(
        parser,
        agent_default="rl_games_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
        include_distributed=False,
        max_iterations_type=parse_positive_int,
    )
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

    return args_cli, remaining_args


def run(argv: list[str]) -> BenchmarkResult:
    """Run the RL-Games training benchmark and write a :class:`~isaaclab.benchmark.TrainingBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    imports_t0 = time.perf_counter_ns()

    import contextlib
    import math
    import os
    import random
    from datetime import datetime

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    from isaaclab.app import launch_simulation
    from isaaclab.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, BenchmarkResult, builders, capture, stepping
    from isaaclab.benchmark.metrics import RL_LIBRARY_DESCRIPTORS, parse_tf_logs
    from isaaclab.benchmark.schema import StartupTime

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    import isaaclab_tasks  # noqa: F401

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    from isaaclab_tasks.utils import resolve_task_config

    apply_env_overrides = _common.apply_env_overrides
    from isaaclab.benchmark.entrypoints.early_stop import (
        RlGamesEarlyStopObserver,
        build_success_kwargs,
        get_success_tracker,
        success_measurements,
    )

    imports_t1 = time.perf_counter_ns()

    args_cli, remaining_args = _parse_args(argv)

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

            if args_cli.seed == -1:
                args_cli.seed = random.randint(0, 10000)
            agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]

            if args_cli.max_iterations is not None:
                agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations

            env_cfg.seed = agent_cfg["params"]["seed"]

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
                        {"name": "seed", "data": agent_cfg["params"]["seed"]},
                        {"name": "num_envs", "data": env_cfg.scene.num_envs},
                        {"name": "max_iterations", "data": agent_cfg["params"]["config"].get("max_epochs")},
                        {
                            "name": "environment_step_measurement_mode",
                            "data": ("serialized_synchronized" if args_cli.measure_sync_step else "host_return"),
                        },
                        {"name": "environment_step_warmup_steps", "data": args_cli.warmup_steps},
                        {"name": "presets", "data": ",".join(cfg.presets)},
                    ]
                },
            )

            config_name = agent_cfg["params"]["config"]["name"]
            log_root_path = os.path.abspath(os.path.join("logs", "rl_games", config_name))
            log_dir = agent_cfg["params"]["config"].get(
                "full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            agent_cfg["params"]["config"]["train_dir"] = log_root_path
            agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
            run_log_dir = os.path.join(log_root_path, log_dir)
            _common.write_run_manifest(
                run_log_dir, library="rl_games", task=args_cli.task, metadata={"agent": args_cli.agent}
            )
            env_cfg.log_dir = run_log_dir
            _common.apply_video_recording(env_cfg, run_log_dir, args_cli, subdir="benchmark")

            rl_device = agent_cfg["params"]["config"]["device"]
            clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
            clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

            env_t0 = time.perf_counter_ns()
            env = _common.create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=True)
            cleanup.callback(lambda: env.close())
            env_t1 = time.perf_counter_ns()

            env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

            # Register with rl_games vecenv registry.
            cleanup.callback(
                register_scoped_rl_games_environment(
                    vecenv,
                    env_configurations,
                    env,
                    lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
                )
            )

            agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

            observer = RlGamesEarlyStopObserver(IsaacAlgoObserver(), **build_success_kwargs(args_cli))
            cleanup.callback(_close_rl_games_writer, observer)
            runner = Runner(observer)
            runner.load(agent_cfg)
            env.seed(agent_cfg["params"]["seed"])
            runner.reset()

            environment_step_timer = stepping.EnvironmentStepTimingRecorder(
                env,
                measure_synchronized_step_breakdown=args_cli.measure_sync_step,
                warmup_steps=args_cli.warmup_steps,
            )
            with environment_step_timer, BenchmarkMonitor(benchmark, interval=1.0):
                runner.run({"train": True, "play": False, "sigma": None})

            benchmark.update_manual_recorders()

            # Flush TensorBoard events before parsing them; ExitStack closes the writer.
            writer = getattr(getattr(observer, "algo", None), "writer", None)
            if writer is not None:
                writer.flush()

            desc = RL_LIBRARY_DESCRIPTORS["rl_games"]
            tb_dir = run_log_dir
            log_data = parse_tf_logs(tb_dir, desc.tfevents_pattern)
            max_epochs = agent_cfg["params"]["config"].get("max_epochs", 1)
            if not log_data or (not log_data.get(desc.reward_tag) and max_epochs >= 1):
                print(
                    f"[WARNING] No TensorBoard data parsed from {tb_dir!r};"
                    " the emitted bundle will report zero metrics. Check the log directory.",
                    file=sys.stderr,
                )

            # RL-Games logs FPS directly; iteration time is steps divided by total FPS.
            horizon_length = agent_cfg["params"]["config"].get("horizon_length", 16)
            steps_per_iteration = env.unwrapped.num_envs * horizon_length
            total_fps_series = list(log_data.get("performance/step_inference_rl_update_fps", []))
            collection_fps_series = list(log_data.get("performance/step_inference_fps", []))
            iteration_times_s = [steps_per_iteration / f for f in total_fps_series if f > 0]

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
                steps_per_iteration=steps_per_iteration,
                frames_per_environment_step=env.unwrapped.num_envs,
                environment_step_warmup_steps=args_cli.warmup_steps,
                environment_step_times_s=environment_step_timer.step_times_s,
                simulation_step_times_s=environment_step_timer.simulation_step_times_s,
                simulation_step_calls=environment_step_timer.simulation_step_calls,
            )

            tracker = get_success_tracker(args_cli, observer.tracker, log_data)
            learning = builders.build_learning(
                reward_series=log_data.get(desc.reward_tag, []),
                ep_length_series=log_data.get(desc.ep_length_tag, []),
                success_rate_series=tracker.history if tracker is not None else None,
                ema_alpha=args_cli.ema_alpha,
                keep_series=not args_cli.no_series,
            )
            success_rate = round(tracker.tail_mean, 4) if (tracker and tracker.history) else None

            versions = capture.capture_versions(benchmark)
            hardware = capture.capture_hardware(benchmark)
            resources = capture.capture_resources(benchmark)

            end_utc = capture.now_utc_iso()
            stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
            seed = agent_cfg["params"]["seed"] if agent_cfg["params"]["seed"] is not None else 0

            run_identity = builders.build_run_identity(
                run_id=capture.synth_run_id("rl_games", cfg.physics_backend, args_cli.task, seed, stamp),
                framework="rl_games",
                config=cfg,
                task=args_cli.task,
                seed=seed,
                start_utc=start_utc,
                end_utc=end_utc,
                num_envs=env.unwrapped.num_envs,
                max_iterations=agent_cfg["params"]["config"].get("max_epochs"),
            )

            checkpoint_path = None
            video_path = os.path.join(run_log_dir, "videos") if getattr(args_cli, "video", False) else None

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
