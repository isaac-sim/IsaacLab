# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-Baselines3 adapter for the unified training benchmark."""

from __future__ import annotations

import sys
import time

from isaaclab_rl.entrypoints import common as _common


def _build_benchmark_callback_class():
    """Create the callback class after simulator startup to defer SB3 imports.

    Returns:
        The ``_BenchmarkCallback`` class.
    """
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import safe_mean

    class _BenchmarkCallback(BaseCallback):
        """``BaseCallback`` that records per-iteration timing, reward, and episode length.

        The callback reads directly from ``self.model.ep_info_buffer``, the same
        buffer ``dump_logs`` uses, so metrics are available regardless
        of the ``log_interval`` setting passed to ``agent.learn``.

        Attributes:
            collection_times_s: Wall-clock seconds spent collecting each rollout.
            iter_times_s: Wall-clock seconds for each rollout and policy update.
            ep_rew_mean: Mean episode reward observed at the end of each rollout.
            ep_len_mean: Mean episode length observed at the end of each rollout.
        """

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.collection_times_s: list[float] = []
            self.iter_times_s: list[float] = []
            self.ep_rew_mean: list[float] = []
            self.ep_len_mean: list[float] = []
            self._rollout_start_ns: int = 0
            self._iteration_start_ns: int = 0
            self._rollout_started = False

        def _on_training_start(self) -> None:
            self._iteration_start_ns = time.perf_counter_ns()

        def _on_rollout_start(self) -> None:
            now = time.perf_counter_ns()
            if self._rollout_started:
                self.iter_times_s.append((now - self._iteration_start_ns) / 1e9)
            self._iteration_start_ns = now
            self._rollout_start_ns = now
            self._rollout_started = True

        def _on_rollout_end(self) -> None:
            elapsed_s = (time.perf_counter_ns() - self._rollout_start_ns) / 1e9
            self.collection_times_s.append(elapsed_s)

            buf = self.model.ep_info_buffer
            if buf and len(buf) > 0 and len(buf[0]) > 0:
                self.ep_rew_mean.append(float(safe_mean([ep["r"] for ep in buf])))
                self.ep_len_mean.append(float(safe_mean([ep["l"] for ep in buf])))
            else:
                self.ep_rew_mean.append(float("nan"))
                self.ep_len_mean.append(float("nan"))

        def _on_training_end(self) -> None:
            if self._rollout_started:
                self.iter_times_s.append((time.perf_counter_ns() - self._iteration_start_ns) / 1e9)

        def _on_step(self) -> bool:
            return True

    return _BenchmarkCallback


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

    from isaaclab_tasks.utils import setup_preset_cli

    add_common_train_args = _common.add_common_train_args
    enable_cameras_for_video = _common.enable_cameras_for_video

    parser = argparse.ArgumentParser(description="Benchmark RL training with Stable-Baselines3.")
    add_common_train_args(
        parser,
        agent_default="sb3_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
        include_distributed=False,
    )

    parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Continue training from checkpoint.")
    parser.add_argument(
        "--keep_all_info",
        action="store_true",
        default=False,
        help="Use a slower SB3 wrapper but keep all the extra training info.",
    )

    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
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

    from scripts.benchmarks.early_stop import add_success_cli_args

    add_success_cli_args(parser, include_check_success=False)
    add_launcher_args(parser)

    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    sys.argv = [sys.argv[0]] + remaining_args

    return args_cli, remaining_args


def run(argv: list[str]) -> None:
    """Run the sb3 training benchmark and write a :class:`~isaaclab.test.benchmark.TrainingBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    imports_t0 = time.perf_counter_ns()

    import contextlib
    import os
    from datetime import datetime

    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecNormalize

    from isaaclab.app import launch_simulation
    from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture
    from isaaclab.test.benchmark.metrics import RL_LIBRARY_DESCRIPTORS, parse_tf_logs
    from isaaclab.test.benchmark.schema import StartupTime

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

    import isaaclab_tasks  # noqa: F401

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    from isaaclab_tasks.utils import resolve_task_config

    apply_env_overrides = _common.apply_env_overrides

    from scripts.benchmarks.early_stop import (
        SuccessRateTrackerWrapper,
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
        app_t1 = time.perf_counter_ns()

        apply_env_overrides(args_cli, env_cfg)

        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 0)
        env_cfg.seed = agent_cfg["seed"]

        # Convert the iteration override to SB3 total timesteps.
        n_steps_cfg = agent_cfg.get("n_steps", 2048)
        if args_cli.max_iterations is not None:
            agent_cfg["n_timesteps"] = args_cli.max_iterations * n_steps_cfg * env_cfg.scene.num_envs
        steps_per_iteration = env_cfg.scene.num_envs * n_steps_cfg
        resolved_max_iterations = (int(agent_cfg["n_timesteps"]) + steps_per_iteration - 1) // steps_per_iteration

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
                    {"name": "presets", "data": ",".join(cfg.presets)},
                ]
            },
        )

        run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
        log_dir = os.path.join(log_root_path, run_info)
        _common.write_run_manifest(log_dir, library="sb3", task=args_cli.task, metadata={"agent": args_cli.agent})
        env_cfg.log_dir = log_dir

        agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
        policy_arch = agent_cfg.pop("policy")
        n_timesteps = agent_cfg.pop("n_timesteps")

        env_t0 = time.perf_counter_ns()
        env = _common.create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=True)
        env = _common.wrap_record_video(env, log_dir, args_cli)
        env_t1 = time.perf_counter_ns()
        success_kwargs = build_success_kwargs(args_cli)
        success_context = SuccessRateTrackerWrapper(
            env,
            success_kwargs["threshold"],
            success_kwargs["window"],
            num_steps_per_env=n_steps_cfg,
        )

        env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

        norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
        norm_args = {}
        for key in norm_keys:
            if key in agent_cfg:
                norm_args[key] = agent_cfg.pop(key)

        if norm_args and norm_args.get("normalize_input"):
            env = VecNormalize(
                env,
                training=True,
                norm_obs=norm_args["normalize_input"],
                norm_reward=norm_args.get("normalize_value", False),
                clip_obs=norm_args.get("clip_obs", 100.0),
                gamma=agent_cfg["gamma"],
                clip_reward=np.inf,
            )

        agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)

        if args_cli.checkpoint is not None:
            agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

        BenchmarkCallback = _build_benchmark_callback_class()
        cb = BenchmarkCallback()
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)

        with contextlib.suppress(KeyboardInterrupt), success_context, BenchmarkMonitor(benchmark, interval=1.0):
            agent.learn(
                total_timesteps=n_timesteps,
                callback=[checkpoint_callback, cb],
                progress_bar=False,
                log_interval=None,
            )
        agent.save(os.path.join(log_dir, "model"))

        benchmark.update_manual_recorders()

        collection_times_s = cb.collection_times_s
        iteration_times_s = cb.iter_times_s
        collection_fps = [steps_per_iteration / t for t in collection_times_s if t > 0]
        total_fps = [steps_per_iteration / t for t in iteration_times_s if t > 0]

        # Filter out NaN entries from rollouts where no episode finished.
        reward_series = [v for v in cb.ep_rew_mean if v == v]  # NaN != NaN
        ep_len_series = [v for v in cb.ep_len_mean if v == v]
        if not reward_series and iteration_times_s:
            print(
                "[WARNING] sb3: no episodes completed during the benchmarked rollouts;"
                " reward/episode-length curves are empty.",
                file=sys.stderr,
            )

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
            collection_fps=collection_fps,
            total_fps=total_fps,
            steps_per_iteration=steps_per_iteration,
        )

        learning = builders.build_learning(
            reward_series=reward_series,
            ep_length_series=ep_len_series,
            ema_alpha=args_cli.ema_alpha,
            keep_series=not args_cli.no_series,
        )

        desc = RL_LIBRARY_DESCRIPTORS["sb3"]
        log_data = parse_tf_logs(log_dir, desc.tfevents_pattern)
        success_tracker = get_success_tracker(args_cli, success_context.tracker, log_data)
        success_rate = round(success_tracker.tail_mean, 4) if (success_tracker and success_tracker.history) else None

        versions = capture.capture_versions(benchmark)
        hardware = capture.capture_hardware(benchmark)
        resources = capture.capture_resources(benchmark)

        end_utc = capture.now_utc_iso()
        stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
        seed = env_cfg.seed if env_cfg.seed is not None else 0

        run_identity = builders.build_run_identity(
            run_id=capture.synth_run_id("sb3", cfg.physics_backend, args_cli.task, seed, stamp),
            framework="sb3",
            config=cfg,
            task=args_cli.task,
            seed=seed,
            start_utc=start_utc,
            end_utc=end_utc,
            num_envs=env.unwrapped.num_envs,
            max_iterations=resolved_max_iterations,
        )

        checkpoint_path = os.path.join(log_dir, "model.zip")
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
        benchmark.add_measurement("train", success_measurements(success_tracker))

        benchmark._finalize_impl()

        env.close()


if __name__ == "__main__":
    run(sys.argv[1:])
