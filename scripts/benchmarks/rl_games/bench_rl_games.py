# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL-Games training-benchmark adapter.

Runs real training under a :class:`~isaaclab.test.benchmark.BenchmarkMonitor` and emits a
:class:`~isaaclab.test.benchmark.schema.TrainingBundle` JSON file. Dispatched from
``scripts/benchmarks/training.py`` via ``--rl_library rl_games``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.benchmarks._common import get_backend_types, import_module_from_path, preset_tokens

# ---------------------------------------------------------------------------
# Path setup — locate scripts/reinforcement_learning and load common helpers
# via explicit file path so that scripts/reinforcement_learning is never added
# to sys.path (it has no __init__.py and is not an importable package).
# ---------------------------------------------------------------------------

_BENCH_DIR = Path(__file__).resolve().parents[1]
_RL_SCRIPTS = _BENCH_DIR.parent / "reinforcement_learning"

_common = import_module_from_path("isaaclab_rl_common", _RL_SCRIPTS / "common.py")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


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

    from isaaclab_tasks.utils import setup_preset_cli

    add_common_train_args = _common.add_common_train_args
    add_isaaclab_launcher_args = _common.add_isaaclab_launcher_args
    enable_cameras_for_video = _common.enable_cameras_for_video

    parser = argparse.ArgumentParser(description="Benchmark RL training with RL-Games.")
    add_common_train_args(
        parser,
        agent_default="rl_games_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    add_isaaclab_launcher_args(parser)

    # benchmark-specific args
    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--benchmark_backend",
        type=str,
        default="schema",
        help=(
            "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
            " 'omniperf', 'osmo', 'json', 'summary'. Legacy long-form aliases accepted."
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

    add_success_cli_args(parser)

    args_cli, remaining_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    sys.argv = [sys.argv[0]] + remaining_args

    return args_cli, remaining_args


# ---------------------------------------------------------------------------
# Main run function (dispatch contract)
# ---------------------------------------------------------------------------


def run(argv: list[str]) -> None:
    """Run the RL-Games training benchmark and write a :class:`~isaaclab.test.benchmark.schema.TrainingBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    import contextlib
    import math
    import os
    import random
    import time
    from datetime import datetime

    import gymnasium as gym
    import torch
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    from isaaclab.app import launch_simulation
    from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture
    from isaaclab.test.benchmark.backend_descriptor import BACKEND_DESCRIPTORS
    from isaaclab.test.benchmark.metrics import parse_tf_logs
    from isaaclab.test.benchmark.schema import StartupTime

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    import isaaclab_tasks  # noqa: F401

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    from isaaclab_tasks.utils import resolve_task_config

    apply_env_overrides = _common.apply_env_overrides
    from scripts.benchmarks.early_stop import RlGamesEarlyStopObserver, build_success_kwargs, get_success_tracker

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    args_cli, remaining_args = _parse_args(argv)

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args_cli):
        app_t1 = time.perf_counter_ns()

        # Apply CLI overrides to env config.
        apply_env_overrides(args_cli, env_cfg)

        # Apply seed override.
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]

        # Apply max_iterations override.
        if args_cli.max_iterations is not None:
            agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations

        env_cfg.seed = agent_cfg["params"]["seed"]

        backend_types = get_backend_types(args_cli.benchmark_backend)
        tokens = preset_tokens(remaining_args)

        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_training",
            backend_type=backend_types,
            output_path=args_cli.output_path,
            use_recorders=True,
            frametime_recorders=any(t in ("summary", "omniperf") for t in backend_types),
            output_prefix=f"benchmark_training_{args_cli.task}",
            workflow_metadata={
                "metadata": [
                    {"name": "task", "data": args_cli.task},
                    {"name": "seed", "data": agent_cfg["params"]["seed"]},
                    {"name": "num_envs", "data": args_cli.num_envs},
                    {"name": "max_iterations", "data": agent_cfg["params"]["config"].get("max_epochs")},
                    {"name": "presets", "data": ",".join(tokens)},
                ]
            },
        )

        # Build log_dir the same way train_rl_games.py does.
        config_name = agent_cfg["params"]["config"]["name"]
        log_root_path = os.path.abspath(os.path.join("logs", "rl_games", config_name))
        log_dir = agent_cfg["params"]["config"].get(
            "full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        agent_cfg["params"]["config"]["train_dir"] = log_root_path
        agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

        # Read rl_games device / clipping config.
        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

        env_t0 = time.perf_counter_ns()
        env = gym.make(
            args_cli.task, cfg=env_cfg, render_mode="rgb_array" if getattr(args_cli, "video", False) else None
        )
        env_t1 = time.perf_counter_ns()

        # Wrap for rl_games.
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

        # Register with rl_games vecenv registry.
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

        observer = RlGamesEarlyStopObserver(IsaacAlgoObserver(), **build_success_kwargs(args_cli))
        runner = Runner(observer)
        runner.load(agent_cfg)
        env.seed(agent_cfg["params"]["seed"])
        runner.reset()

        with BenchmarkMonitor(benchmark, interval=1.0):
            runner.run({"train": True, "play": False, "sigma": None})

        benchmark.update_manual_recorders()

        # Flush and close the rl_games TensorBoard writer so all events are on
        # disk before parse_tf_logs reads them.  The writer lives on the algo
        # object attached to the observer after after_init().
        if observer.algo is not None and getattr(observer.algo, "writer", None) is not None:
            try:
                observer.algo.writer.flush()
                observer.algo.writer.close()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[WARNING] rl_games TensorBoard writer flush/close failed: {exc!r};"
                    " TB logs may be incomplete and metrics may be zero.",
                    file=sys.stderr,
                )

        # Parse TensorBoard logs.
        desc = BACKEND_DESCRIPTORS["rl_games"]
        tb_dir = os.path.join(log_root_path, log_dir)
        log_data = parse_tf_logs(tb_dir, desc.tfevents_pattern)
        max_epochs = agent_cfg["params"]["config"].get("max_epochs", 1)
        if not log_data or (not log_data.get(desc.reward_tag) and max_epochs >= 1):
            print(
                f"[WARNING] No TensorBoard data parsed from {tb_dir!r};"
                " the emitted bundle will report zero metrics. Check the log directory.",
                file=sys.stderr,
            )

        # Derive per-iteration timing.
        # rl_games logs FPS directly; iteration_time = steps / total_fps
        horizon_length = agent_cfg["params"]["config"].get("horizon_length", 16)
        steps_per_iteration = env.unwrapped.num_envs * horizon_length
        total_fps_series = list(log_data.get("performance/step_inference_rl_update_fps", []))
        collection_fps_series = list(log_data.get("performance/step_inference_fps", []))
        iteration_times_s = [steps_per_iteration / f for f in total_fps_series if f > 0]

        startup = StartupTime(
            app_launch=(app_t1 - app_t0) / 1e9,
            env_creation=(env_t1 - env_t0) / 1e9,
            first_step=(iteration_times_s[0] if iteration_times_s else 0.0),
        )

        runtime = builders.build_runtime(
            startup_time_s=startup,
            iteration_times_s=iteration_times_s,
            collection_fps=collection_fps_series,
            total_fps=total_fps_series,
            steps_per_iteration=steps_per_iteration,
        )

        learning = builders.build_learning(
            reward_series=log_data.get(desc.reward_tag, []),
            ep_length_series=log_data.get(desc.ep_length_tag, []),
            ema_alpha=args_cli.ema_alpha,
            keep_series=not args_cli.no_series,
        )

        tracker = get_success_tracker(args_cli, observer.tracker, log_data)
        success_rate = round(tracker.tail_mean, 4) if (tracker and tracker.history) else None

        versions = capture.capture_versions(benchmark)
        hardware = capture.capture_hardware(benchmark)
        resources = capture.capture_resources(benchmark)
        cfg = capture.run_config_from_presets(tokens)

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
        video_path = os.path.join(log_root_path, log_dir, "videos") if getattr(args_cli, "video", False) else None

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

        benchmark._finalize_impl()

        env.close()


if __name__ == "__main__":
    run(sys.argv[1:])
