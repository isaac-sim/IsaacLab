# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL play-benchmark adapter.

Rolls out a checkpointed RSL-RL policy under a :class:`~isaaclab.benchmark.BenchmarkMonitor`
and emits a :class:`~isaaclab.benchmark.schema.PlayBundle` JSON file. Dispatched from
``isaaclab benchmark play`` via ``--rl_library rsl_rl``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time
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


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments and forward the remaining Hydra preset tokens via ``sys.argv``.

    Builds the parser, appends launcher args via :func:`~isaaclab.app.add_launcher_args`, then
    calls :func:`~isaaclab_tasks.utils.setup_preset_cli` to split known args from the verbatim
    remainder (``physics=`` / ``renderer=`` / ``presets=`` tokens). The remainder is written back
    to ``sys.argv`` so that Hydra and ``launch_simulation`` pick up the preset selection.

    Args:
        argv: Raw command-line arguments (``sys.argv[1:]`` after the dispatcher strips
            ``--rl_library``).

    Returns:
        Tuple of ``(parsed_args, remaining)`` where *remaining* are the Hydra preset tokens.
    """
    from isaaclab.app import add_launcher_args
    from isaaclab.benchmark._cli import add_benchmark_output_args, add_play_args

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Benchmark RL inference (play) with RSL-RL.")
    add_play_args(
        parser,
        argv,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    add_benchmark_output_args(parser)
    add_launcher_args(parser)
    common.add_frontend_args(parser)

    args, remaining = setup_preset_cli(parser, argv)
    common.enable_cameras_for_video(args)
    sys.argv = [sys.argv[0]] + remaining
    return args, remaining


def run(argv: list[str]) -> BenchmarkResult:
    """Run the RSL-RL play benchmark and write a :class:`~isaaclab.benchmark.schema.PlayBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.benchmark import BenchmarkMonitor, BenchmarkResult, builders, capture, stepping
    from isaaclab.benchmark.schema import StartupTime

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    # Importing the task packages registers their gym environments so the
    # requested ``--task`` can be resolved.
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    # PLACEHOLDER: Extension template (do not remove this comment)
    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    args, remaining = _parse_args(argv)

    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
    common.pre_launch_video_config(env_cfg, args_cli=args)

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args):
        with contextlib.ExitStack() as cleanup:
            app_t1 = time.perf_counter_ns()
            common.apply_video_recording(env_cfg, args.output_path, args, subdir="play")

            if args.num_envs is not None:
                env_cfg.scene.num_envs = args.num_envs
            if args.seed is not None:
                agent_cfg.seed = args.seed
            env_cfg.seed = agent_cfg.seed

            installed_rsl_rl = metadata.version("rsl-rl-lib")
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl)

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            if args.checkpoint in common.CHECKPOINT_SELECTORS:
                resume_path = common.resolve_checkpoint_selector(
                    log_root_path,
                    args.checkpoint,
                    library="rsl_rl",
                    task=args.task,
                    checkpoint_pattern=r"model_.*\.pt",
                    metadata={"agent": args.agent},
                )
            else:
                resume_path = common.resolve_play_checkpoint(args.checkpoint, "rsl_rl", args.task, env_cfg)

            cfg = capture.run_config_from_env_cfg(env_cfg)
            benchmark = create_benchmark(
                "benchmark_play",
                args,
                output_prefix=f"benchmark_play_{args.task}",
                metadata=[
                    {"name": "task", "data": args.task},
                    {"name": "num_envs", "data": args.num_envs},
                    {"name": "num_steps", "data": args.num_steps},
                    *step_timing_metadata(args),
                ],
            )

            env_t0 = time.perf_counter_ns()
            env = common.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=True)
            cleanup.callback(lambda: env.close())
            env_t1 = time.perf_counter_ns()

            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            num_envs = env.unwrapped.num_envs

            # Load the trained policy the same way isaaclab_rl.entrypoints.backends.play_rsl_rl does.
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            environment_step_timer = stepping.EnvironmentStepTimingRecorder(
                env,
                measure_synchronized_step_breakdown=args.measure_sync_step,
                warmup_steps=args.warmup_steps,
            )
            total_steps = args.warmup_steps + args.num_steps
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
                warmup_steps=args.warmup_steps,
                timer=environment_step_timer,
            )

            versions, hardware, resources = capture_snapshots(benchmark)

            seed = agent_cfg.seed if agent_cfg.seed is not None else 0
            run_identity = finish_run_identity(
                framework="rsl_rl",
                config=cfg,
                task=args.task,
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
                video_path=env_cfg.video_recorders[0].output_dir if args.video else None,
            )

            benchmark.attach_bundle(bundle)

            output_paths = benchmark.finalize()
            result = BenchmarkResult(bundle=bundle, output_paths=output_paths)

    return result


if __name__ == "__main__":
    run(sys.argv[1:])
