# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SKRL play-benchmark adapter.

Rolls out a checkpointed SKRL policy under a :class:`~isaaclab.benchmark.BenchmarkMonitor`
and emits a :class:`~isaaclab.benchmark.schema.PlayBundle` JSON file. Dispatched from
``isaaclab benchmark play`` via ``--rl_library skrl``.
"""

from __future__ import annotations

import argparse
import contextlib
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

    parser = argparse.ArgumentParser(description="Benchmark RL inference (play) with SKRL.")
    add_play_args(
        parser,
        argv,
        agent_default=None,
        agent_help="Agent configuration entry point (default: the task's canonical SKRL configuration).",
    )
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax"],
        help="ML framework used for the skrl agent.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["AMP", "PPO", "IPPO", "MAPPO"],
        help="Optional algorithm selector; with --agent, the resolved agent.class must match.",
    )
    add_benchmark_output_args(parser)
    add_launcher_args(parser)
    common.add_frontend_args(parser)

    args_cli, remaining_args = setup_preset_cli(parser, argv)
    common.enable_cameras_for_video(args_cli)
    sys.argv = [sys.argv[0]] + remaining_args

    return args_cli, remaining_args


def run(argv: list[str]) -> BenchmarkResult:
    """Run the SKRL play benchmark and write a :class:`~isaaclab.benchmark.schema.PlayBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    from isaaclab.app import launch_simulation
    from isaaclab.benchmark import BenchmarkMonitor, BenchmarkResult, builders, capture, stepping
    from isaaclab.benchmark.schema import StartupTime

    from isaaclab_rl.skrl import SkrlVecEnvWrapper, resolve_skrl_agent_cfg_entry_point, resolve_skrl_algorithm

    # Importing the task packages registers their gym environments so the
    # requested ``--task`` can be resolved.
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    # PLACEHOLDER: Extension template (do not remove this comment)
    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    args_cli, remaining_args = _parse_args(argv)

    agent_cfg_entry_point = resolve_skrl_agent_cfg_entry_point(args_cli.agent, args_cli.algorithm)

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point)
    algorithm = resolve_skrl_algorithm(agent_cfg, args_cli.algorithm)
    common.pre_launch_video_config(env_cfg, args_cli=args_cli)

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args_cli):
        with contextlib.ExitStack() as cleanup:
            app_t1 = time.perf_counter_ns()
            common.apply_video_recording(env_cfg, args_cli.output_path, args_cli, subdir="play")

            if args_cli.ml_framework.startswith("jax"):
                import skrl

                cleanup.enter_context(common.preserve_attribute(skrl.config.jax, "backend"))
                skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

            if args_cli.num_envs is not None:
                env_cfg.scene.num_envs = args_cli.num_envs
            agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 0)
            env_cfg.seed = agent_cfg["seed"]

            log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
            if args_cli.checkpoint in common.CHECKPOINT_SELECTORS:
                resume_path = common.resolve_checkpoint_selector(
                    log_root_path,
                    args_cli.checkpoint,
                    library="skrl",
                    task=args_cli.task,
                    checkpoint_pattern=r".*",
                    other_dirs=["checkpoints"],
                    metadata={
                        "agent": agent_cfg_entry_point,
                        "algorithm": algorithm,
                        "ml_framework": args_cli.ml_framework,
                    },
                )
            else:
                resume_path = common.resolve_play_checkpoint(args_cli.checkpoint, "skrl", args_cli.task, env_cfg)

            cfg = capture.run_config_from_env_cfg(env_cfg)
            benchmark = create_benchmark(
                "benchmark_play",
                args_cli,
                output_prefix=f"benchmark_play_{args_cli.task}",
                metadata=[
                    {"name": "task", "data": args_cli.task},
                    {"name": "num_envs", "data": args_cli.num_envs},
                    {"name": "num_steps", "data": args_cli.num_steps},
                    {"name": "algorithm", "data": algorithm.upper()},
                    *step_timing_metadata(args_cli),
                ],
            )

            env_t0 = time.perf_counter_ns()
            env = common.create_isaaclab_env(
                args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=algorithm == "ppo"
            )
            cleanup.callback(lambda: env.close())
            env_t1 = time.perf_counter_ns()

            env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

            num_envs = env.unwrapped.num_envs

            if args_cli.ml_framework.startswith("torch"):
                from skrl.utils.runner.torch import Runner
            elif args_cli.ml_framework.startswith("jax"):
                from skrl.utils.runner.jax import Runner

            # Load the trained policy the same way isaaclab_rl.entrypoints.backends.play_skrl does.
            agent_cfg["trainer"]["close_environment_at_exit"] = False
            agent_cfg["agent"]["experiment"]["write_interval"] = 0
            agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
            runner = Runner(env, agent_cfg)
            runner.agent.load(resume_path)
            runner.agent.enable_training_mode(False, apply_to_models=True)

            def policy(obs):
                """Map an observation batch to a deterministic action batch via the skrl agent.

                Mirrors the inference path in ``isaaclab_rl.entrypoints.backends.play_skrl``:
                runs the agent's deterministic action, preferring the policy ``mean_actions``
                over the sampled action returned as the first element.

                Args:
                    obs: Observation returned by the skrl-wrapped env.

                Returns:
                    The action tensor to feed ``env.step``.
                """
                states = env.state()
                outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                return outputs[-1].get("mean_actions", outputs[0])

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

            seed = agent_cfg["seed"] if agent_cfg.get("seed") is not None else 0
            run_identity = finish_run_identity(
                framework="skrl",
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
