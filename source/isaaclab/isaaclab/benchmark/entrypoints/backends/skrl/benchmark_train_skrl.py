# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SKRL adapter for the unified training benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.benchmark import BenchmarkResult

import sys
import time

from isaaclab_rl.entrypoints import common as _common


def _build_benchmark_trainer_class():
    """Create the trainer class after simulator startup to defer SKRL imports.

    Returns:
        The ``BenchmarkTrainer`` class.
    """
    from skrl.trainers.torch import SequentialTrainer

    from isaaclab.benchmark.metrics import RL_LIBRARY_DESCRIPTORS

    descriptor = RL_LIBRARY_DESCRIPTORS["skrl"]

    class BenchmarkTrainer(SequentialTrainer):
        """SequentialTrainer that records per-iteration timing, reward, and episode length.

        Wraps ``agent.post_interaction`` to capture metrics at rollout boundaries.

        Attributes:
            collection_times_s: Wall-clock seconds spent collecting each rollout.
            iter_times_s: Wall-clock seconds per iteration (one rollout-buffer fill).
            iter_rewards: Last observed mean completed-episode return per iteration.
            iter_ep_lengths: Last observed mean episode length per iteration.
        """

        def __init__(self, env, agents, cfg=None) -> None:
            super().__init__(env=env, agents=agents, cfg=cfg)
            self.collection_times_s: list[float] = []
            self.iter_times_s: list[float] = []
            self.iter_rewards: list[float] = []
            self.iter_ep_lengths: list[float] = []

        def train(self) -> None:
            """Run training and record per-iteration metrics.

            Resolves the rollout boundary from ``agent.cfg.rollouts`` (skrl >= 2.x)
            or ``agent._rollouts`` (skrl < 2.x).
            """
            if self.num_simultaneous_agents > 1 or self.env.num_agents > 1:
                raise ValueError("The SKRL training benchmark supports single-agent trainers only.")

            agent_obj = self.agents
            agent_cfg = getattr(agent_obj, "cfg", None)
            rollouts_val = (
                getattr(agent_cfg, "rollouts", None) if agent_cfg is not None else getattr(agent_obj, "_rollouts", None)
            )
            if not rollouts_val:
                raise ValueError(
                    "The SKRL agent does not expose a rollout length required for benchmark iteration metrics."
                )

            rollouts = int(rollouts_val)
            timesteps = self.cfg.timesteps
            max_iters = timesteps // rollouts

            _orig_post = agent_obj.post_interaction
            _iter_start_ns: list[int] = [time.perf_counter_ns()]

            _last_reward: list[float] = [float("nan")]
            _last_ep_len: list[float] = [float("nan")]

            def _patched_post(*, timestep: int, timesteps: int) -> None:
                # post_interaction may flush and clear these values.
                td = getattr(agent_obj, "tracking_data", {})
                reward_values = td.get(descriptor.reward_tag, ())
                ep_len_values = td.get(descriptor.ep_length_tag, ())
                if reward_values:
                    _last_reward[0] = float(sum(reward_values) / len(reward_values))
                if ep_len_values:
                    _last_ep_len[0] = float(sum(ep_len_values) / len(ep_len_values))

                at_boundary = (timestep + 1) % rollouts == 0
                collection_end_ns = time.perf_counter_ns() if at_boundary else 0
                _orig_post(timestep=timestep, timesteps=timesteps)

                if at_boundary:
                    iter_end_ns = time.perf_counter_ns()
                    self.collection_times_s.append((collection_end_ns - _iter_start_ns[0]) / 1e9)
                    self.iter_times_s.append((iter_end_ns - _iter_start_ns[0]) / 1e9)
                    self.iter_rewards.append(_last_reward[0])
                    self.iter_ep_lengths.append(_last_ep_len[0])
                    _iter_start_ns[0] = time.perf_counter_ns()

            agent_obj.post_interaction = _patched_post
            try:
                super().train()
            finally:
                agent_obj.post_interaction = _orig_post

            self.collection_times_s = self.collection_times_s[:max_iters]
            self.iter_times_s = self.iter_times_s[:max_iters]
            self.iter_rewards = self.iter_rewards[:max_iters]
            self.iter_ep_lengths = self.iter_ep_lengths[:max_iters]

    return BenchmarkTrainer


def _parse_args(argv: list[str]):
    """Parse CLI arguments and forward the remaining Hydra preset tokens via ``sys.argv``.

    Args:
        argv: Raw command-line arguments (``sys.argv[1:]`` after dispatcher
            strips ``--rl_library``).

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

    parser = argparse.ArgumentParser(description="Benchmark RL training with SKRL.")
    add_common_train_args(
        parser,
        agent_default=None,
        agent_help=(
            "Name of the RL agent configuration entry point. Defaults to None, in which"
            " case --algorithm is used to determine the default agent entry point."
        ),
        include_distributed=False,
        max_iterations_type=parse_positive_int,
    )
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch"],
        help="ML framework used for benchmark training.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["AMP", "PPO"],
        help="RL algorithm used for benchmark training.",
    )
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

    add_success_cli_args(parser, include_check_success=False)
    add_launcher_args(parser)

    if "--distributed" in argv:
        parser.error("Distributed training benchmarks are not supported.")

    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    sys.argv = [sys.argv[0]] + remaining_args

    return args_cli, remaining_args


def run(argv: list[str]) -> BenchmarkResult:
    """Run the SKRL training benchmark and write a :class:`~isaaclab.benchmark.TrainingBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    imports_t0 = time.perf_counter_ns()

    import contextlib
    import os

    from isaaclab.app import launch_simulation
    from isaaclab.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, BenchmarkResult, builders, capture, stepping
    from isaaclab.benchmark.metrics import RL_LIBRARY_DESCRIPTORS, parse_tf_logs
    from isaaclab.benchmark.schema import StartupTime

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    from isaaclab_tasks.utils import resolve_task_config

    apply_env_overrides = _common.apply_env_overrides
    from isaaclab.benchmark.entrypoints.early_stop import (
        SuccessRateTrackerWrapper,
        build_success_kwargs,
        get_success_tracker,
        success_measurements,
    )

    imports_t1 = time.perf_counter_ns()

    args_cli, remaining_args = _parse_args(argv)

    # Derive the default agent entry point from the selected algorithm.
    if args_cli.agent is None:
        algorithm = args_cli.algorithm.lower()
        agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"
    else:
        agent_cfg_entry_point = args_cli.agent
        algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

    config_t0 = time.perf_counter_ns()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point)
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

            rollouts = int(agent_cfg["agent"]["rollouts"])
            if args_cli.max_iterations is not None:
                agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * rollouts
            resolved_max_iterations = agent_cfg["trainer"]["timesteps"] // rollouts
            agent_cfg["trainer"]["close_environment_at_exit"] = False

            agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 0)
            env_cfg.seed = agent_cfg["seed"]

            log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
            from datetime import datetime

            log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
            if agent_cfg["agent"]["experiment"]["experiment_name"]:
                log_dir_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
            agent_cfg["agent"]["experiment"]["directory"] = log_root_path
            agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir_name
            log_dir = os.path.join(log_root_path, log_dir_name)
            _common.write_run_manifest(
                log_dir,
                library="skrl",
                task=args_cli.task,
                metadata={
                    "agent": agent_cfg_entry_point,
                    "algorithm": algorithm,
                    "ml_framework": args_cli.ml_framework,
                },
            )

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
                        {"name": "seed", "data": agent_cfg["seed"]},
                        {"name": "num_envs", "data": env_cfg.scene.num_envs},
                        {"name": "max_iterations", "data": resolved_max_iterations},
                        {"name": "algorithm", "data": algorithm.upper()},
                        {
                            "name": "environment_step_measurement_mode",
                            "data": ("serialized_synchronized" if args_cli.measure_sync_step else "host_return"),
                        },
                        {"name": "environment_step_warmup_steps", "data": args_cli.warmup_steps},
                        {"name": "presets", "data": ",".join(cfg.presets)},
                    ]
                },
            )

            env_cfg.log_dir = log_dir

            env_t0 = time.perf_counter_ns()
            env = _common.create_isaaclab_env(
                args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=algorithm == "ppo"
            )
            cleanup.callback(lambda: env.close())
            env = _common.wrap_record_video(env, log_dir, args_cli)
            env_t1 = time.perf_counter_ns()
            success_kwargs = build_success_kwargs(args_cli)
            success_context = SuccessRateTrackerWrapper(
                env,
                success_kwargs["threshold"],
                success_kwargs["window"],
                num_steps_per_env=rollouts,
            )

            env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

            from skrl.utils.runner.torch import Runner

            BenchmarkTrainer = _build_benchmark_trainer_class()

            class _BenchmarkRunner(Runner):
                """Runner that installs ``BenchmarkTrainer`` during initialization."""

                def _generate_trainer(self, env, cfg, agent):
                    from skrl.trainers.torch import SequentialTrainerCfg

                    trainer_cfg = SequentialTrainerCfg(**self._process_cfg(cfg["trainer"]))
                    return BenchmarkTrainer(env=env, agents=agent, cfg=trainer_cfg)

            runner = _BenchmarkRunner(env, agent_cfg)
            bt = runner._trainer

            environment_step_timer = stepping.EnvironmentStepTimingRecorder(
                env,
                measure_synchronized_step_breakdown=args_cli.measure_sync_step,
                warmup_steps=args_cli.warmup_steps,
            )
            with success_context, environment_step_timer, BenchmarkMonitor(benchmark, interval=1.0):
                runner.run()

            benchmark.update_manual_recorders()

            collection_times_s = list(bt.collection_times_s)
            iter_times_s = list(bt.iter_times_s)
            reward_series = [value for value in bt.iter_rewards if value == value]
            ep_len_series = [value for value in bt.iter_ep_lengths if value == value]

            num_envs = env.unwrapped.num_envs
            steps_per_iteration = num_envs * rollouts
            collection_fps = [steps_per_iteration / value for value in collection_times_s if value > 0]
            total_fps = [steps_per_iteration / value for value in iter_times_s if value > 0]

            startup = StartupTime(
                app_launch=(app_t1 - app_t0) / 1e9,
                env_creation=(env_t1 - env_t0) / 1e9,
                first_step=(iter_times_s[0] if iter_times_s else 0.0),
                python_imports=(imports_t1 - imports_t0) / 1e9,
                task_config=(config_t1 - config_t0) / 1e9,
            )

            runtime = builders.build_runtime(
                startup_time_s=startup,
                iteration_times_s=iter_times_s,
                collection_fps=collection_fps,
                total_fps=total_fps,
                steps_per_iteration=steps_per_iteration,
                frames_per_environment_step=env.unwrapped.num_envs,
                environment_step_times_s=environment_step_timer.step_times_s,
                simulation_step_times_s=environment_step_timer.simulation_step_times_s,
                simulation_step_calls=environment_step_timer.simulation_step_calls,
            )

            learning = builders.build_learning(
                reward_series=reward_series,
                ep_length_series=ep_len_series,
                ema_alpha=args_cli.ema_alpha,
                keep_series=not args_cli.no_series,
            )

            desc = RL_LIBRARY_DESCRIPTORS["skrl"]
            log_data = parse_tf_logs(log_dir, desc.tfevents_pattern)
            success_tracker = get_success_tracker(args_cli, success_context.tracker, log_data)
            success_rate = (
                round(success_tracker.tail_mean, 4) if (success_tracker and success_tracker.history) else None
            )

            versions = capture.capture_versions(benchmark)
            hardware = capture.capture_hardware(benchmark)
            resources = capture.capture_resources(benchmark)

            end_utc = capture.now_utc_iso()
            stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
            seed = agent_cfg["seed"] if agent_cfg.get("seed") is not None else 0

            run_identity = builders.build_run_identity(
                run_id=capture.synth_run_id("skrl", cfg.physics_backend, args_cli.task, seed, stamp),
                framework="skrl",
                config=cfg,
                task=args_cli.task,
                seed=seed,
                start_utc=start_utc,
                end_utc=end_utc,
                num_envs=num_envs,
                max_iterations=resolved_max_iterations,
            )

            bundle = builders.build_training_bundle(
                run=run_identity,
                versions=versions,
                hardware=hardware,
                runtime=runtime,
                resources=resources,
                learning=learning,
                success_rate=success_rate,
                checkpoint_path=None,
                video_path=os.path.join(log_dir, "videos") if args_cli.video else None,
            )

            benchmark.attach_bundle(bundle)
            benchmark.add_measurement("train", success_measurements(success_tracker))

            output_paths = benchmark.finalize()
            result = BenchmarkResult(bundle=bundle, output_paths=output_paths)

    return result


if __name__ == "__main__":
    run(sys.argv[1:])
