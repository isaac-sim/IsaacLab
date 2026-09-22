# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-Baselines3 play-benchmark adapter.

Rolls out a checkpointed Stable-Baselines3 policy under a
:class:`~isaaclab.benchmark.BenchmarkMonitor` and emits a
:class:`~isaaclab.benchmark.schema.PlayBundle` JSON file. Dispatched from
``isaaclab benchmark play`` via ``--rl_library sb3``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.benchmark.entrypoints._shared import (
    build_play_runtime,
    capture_snapshots,
    create_benchmark,
    finish_run_identity,
    step_timing_metadata,
)

from isaaclab_rl.entrypoints import common

if TYPE_CHECKING:
    from isaaclab.benchmark import BenchmarkResult


def _parse_args(argv: list[str]):
    """Parse CLI arguments and forward the remaining Hydra preset tokens via ``sys.argv``.

    Args:
        argv: Raw command-line arguments (``sys.argv[1:]`` after dispatcher strips
            ``--rl_library``).

    Returns:
        Tuple of ``(parsed_args, remaining)`` where *remaining* are the verbatim Hydra
        preset tokens written back to ``sys.argv`` for ``launch_simulation`` to pick up.
    """
    from isaaclab.app import add_launcher_args
    from isaaclab.benchmark._cli import add_benchmark_output_args, add_play_args

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Benchmark RL inference (play) with Stable-Baselines3.")
    add_play_args(
        parser, argv, agent_default="sb3_cfg_entry_point", agent_help="Name of the RL agent configuration entry point."
    )
    parser.add_argument(
        "--keep_all_info",
        action="store_true",
        default=False,
        help="Use a slower SB3 wrapper but keep all the extra training info.",
    )
    add_benchmark_output_args(parser)
    add_launcher_args(parser)
    common.add_frontend_args(parser)

    args_cli, remaining_args = setup_preset_cli(parser, argv)
    common.enable_cameras_for_video(args_cli)
    sys.argv = [sys.argv[0]] + remaining_args

    return args_cli, remaining_args


def run(argv: list[str]) -> BenchmarkResult:
    """Run the sb3 play benchmark and write a :class:`~isaaclab.benchmark.schema.PlayBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from isaaclab.app import launch_simulation
    from isaaclab.benchmark import BenchmarkMonitor, BenchmarkResult, builders, capture, stepping
    from isaaclab.benchmark.schema import StartupTime

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
    common.pre_launch_video_config(env_cfg, args_cli=args_cli)

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args_cli):
        with contextlib.ExitStack() as cleanup:
            app_t1 = time.perf_counter_ns()
            common.apply_video_recording(env_cfg, args_cli.output_path, args_cli, subdir="play")

            if args_cli.num_envs is not None:
                env_cfg.scene.num_envs = args_cli.num_envs
            agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 0)
            env_cfg.seed = agent_cfg["seed"]

            log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
            if args_cli.checkpoint in common.CHECKPOINT_SELECTORS:
                resume_path = common.resolve_checkpoint_selector(
                    log_root_path,
                    args_cli.checkpoint,
                    library="sb3",
                    task=args_cli.task,
                    checkpoint_pattern=r"model_.*\.zip",
                    preferred_checkpoint_pattern=r"model\.zip",
                    metadata={"agent": args_cli.agent},
                )
            else:
                resume_path = common.resolve_play_checkpoint(args_cli.checkpoint, "sb3", args_cli.task, env_cfg)

            cfg = capture.run_config_from_env_cfg(env_cfg)
            benchmark = create_benchmark(
                "benchmark_play",
                args_cli,
                output_prefix=f"benchmark_play_{args_cli.task}",
                metadata=[
                    {"name": "task", "data": args_cli.task},
                    {"name": "num_envs", "data": args_cli.num_envs},
                    {"name": "num_steps", "data": args_cli.num_steps},
                    *step_timing_metadata(args_cli),
                ],
            )

            env_t0 = time.perf_counter_ns()
            env = common.create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=True)
            cleanup.callback(lambda: env.close())
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

            environment_step_timer = stepping.EnvironmentStepTimingRecorder(
                env,
                measure_synchronized_step_breakdown=args_cli.measure_sync_step,
                warmup_steps=args_cli.warmup_steps,
            )
            total_steps = args_cli.warmup_steps + args_cli.num_steps
            with environment_step_timer, BenchmarkMonitor(benchmark, interval=1.0):
                all_step_times, reward, ep_length, success_rate = stepping.run_play_loop(env, policy, total_steps)

            benchmark.update_manual_recorders()

            startup = StartupTime(
                app_launch=(app_t1 - app_t0) / 1e9,
                env_creation=(env_t1 - env_t0) / 1e9,
                first_step=all_step_times[0],
            )
            runtime = build_play_runtime(
                startup=startup,
                step_times_s=all_step_times,
                num_envs=num_envs,
                warmup_steps=args_cli.warmup_steps,
                timer=environment_step_timer,
            )

            versions, hardware, resources = capture_snapshots(benchmark)

            seed = env_cfg.seed if env_cfg.seed is not None else 0
            run_identity = finish_run_identity(
                framework="sb3",
                config=cfg,
                task=args_cli.task,
                seed=seed,
                start_utc=start_utc,
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
                video_path=env_cfg.video_recorders[0].output_dir if args_cli.video else None,
            )

            benchmark.attach_bundle(bundle)

            output_paths = benchmark.finalize()
            result = BenchmarkResult(bundle=bundle, output_paths=output_paths)

    return result


if __name__ == "__main__":
    run(sys.argv[1:])
