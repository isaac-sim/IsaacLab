# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-Baselines3 play-benchmark adapter.

Rolls out a checkpointed Stable-Baselines3 policy under a
:class:`~isaaclab.test.benchmark.BenchmarkMonitor` and emits a
:class:`~isaaclab.test.benchmark.schema.PlayBundle` JSON file. Dispatched from
``scripts/benchmarks/play.py`` via ``--rl_library sb3``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from isaaclab_rl.entrypoints import common as _common


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

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Benchmark RL inference (play) with Stable-Baselines3.")
    help_requested = "-h" in argv or "--help" in argv
    parser.add_argument("--task", type=str, required=not help_requested, help="Gym task id to benchmark.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of inference steps to benchmark.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Local or Nucleus checkpoint path to roll out; falls back to the published checkpoint when omitted.",
    )
    parser.add_argument(
        "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )
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
    add_launcher_args(parser)

    args_cli, remaining_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + remaining_args

    return args_cli, remaining_args


def run(argv: list[str]) -> None:
    """Run the sb3 play benchmark and write a :class:`~isaaclab.test.benchmark.schema.PlayBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    import contextlib
    import os

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from isaaclab.app import launch_simulation
    from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, stepping
    from isaaclab.test.benchmark.schema import StartupTime

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

    # Importing the task packages registers their gym environments so the
    # requested ``--task`` can be resolved.
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    # PLACEHOLDER: Extension template (do not remove this comment)
    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    args_cli, remaining_args = _parse_args(argv)

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args_cli):
        app_t1 = time.perf_counter_ns()

        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 0)
        env_cfg.seed = agent_cfg["seed"]

        log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
        if args_cli.checkpoint in _common.CHECKPOINT_SELECTORS:
            resume_path = _common.resolve_checkpoint_selector(
                log_root_path,
                args_cli.checkpoint,
                library="sb3",
                task=args_cli.task,
                checkpoint_pattern=r"model_.*\.zip",
                preferred_checkpoint_pattern=r"model\.zip",
                metadata={"agent": args_cli.agent},
            )
        else:
            resume_path = _common.resolve_play_checkpoint(args_cli.checkpoint, "sb3", args_cli.task)

        cfg = capture.run_config_from_presets(remaining_args)
        formatter_types = [value.strip() for value in args_cli.benchmark_formatter.split(",") if value.strip()]
        formatter_types = formatter_types or ["omniperf"]

        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_play",
            formatter_type=formatter_types,
            output_path=args_cli.output_path,
            use_recorders=True,
            frametime_recorders=any(t in ("summary", "omniperf") for t in formatter_types),
            output_prefix=f"benchmark_play_{args_cli.task}",
            workflow_metadata={
                "metadata": [
                    {"name": "task", "data": args_cli.task},
                    {"name": "num_envs", "data": args_cli.num_envs},
                    {"name": "num_frames", "data": args_cli.num_frames},
                    {"name": "presets", "data": ",".join(cfg.presets)},
                ]
            },
        )

        env_t0 = time.perf_counter_ns()
        env = gym.make(args_cli.task, cfg=env_cfg)
        env_t1 = time.perf_counter_ns()

        # Post-process agent configuration the same way isaaclab_rl.entrypoints.backends.play_sb3 does.
        agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

        num_envs = env.unwrapped.num_envs

        # Wrap for stable-baselines3.
        env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

        # Load VecNormalize statistics when they were saved next to the checkpoint.
        vec_norm_path = Path(resume_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl"))
        if vec_norm_path.exists():
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        elif "normalize_input" in agent_cfg:
            env = VecNormalize(
                env,
                training=True,
                norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
                clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            )

        # Load the trained policy.
        agent = PPO.load(resume_path, env, print_system_info=True)

        def policy(obs):
            """Map an observation batch to a deterministic action batch via the sb3 agent.

            Mirrors the inference path in ``isaaclab_rl.entrypoints.backends.play_sb3``:
            the sb3-wrapped env returns NumPy observations, which ``agent.predict`` consumes
            directly, returning NumPy actions for ``env.step``.

            Args:
                obs: NumPy observation returned by the sb3-wrapped env.

            Returns:
                The NumPy action array to feed ``env.step``.
            """
            actions, _ = agent.predict(obs, deterministic=True)
            return actions

        with BenchmarkMonitor(benchmark, interval=1.0):
            step_times, reward, ep_length, success_rate = stepping.run_play_loop(env, policy, args_cli.num_frames)

        benchmark.update_manual_recorders()

        startup = StartupTime(
            app_launch=(app_t1 - app_t0) / 1e9,
            env_creation=(env_t1 - env_t0) / 1e9,
            first_step=(step_times[0] if step_times else 0.0),
        )

        fps = [num_envs / t for t in step_times if t > 0]
        runtime = builders.build_runtime(
            startup_time_s=startup,
            iteration_times_s=step_times,
            collection_fps=fps,
            total_fps=fps,
            steps_per_iteration=num_envs,
        )

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
            num_envs=num_envs,
        )

        bundle = builders.build_play_bundle(
            run=run_identity,
            versions=versions,
            hardware=hardware,
            runtime=runtime,
            resources=resources,
            success_rate=success_rate,
            reward=reward,
            ep_length=ep_length,
            checkpoint_path=resume_path,
        )

        benchmark.attach_bundle(bundle)

        benchmark._finalize_impl()

        env.close()


if __name__ == "__main__":
    run(sys.argv[1:])
