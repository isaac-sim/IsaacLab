# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SKRL play-benchmark adapter.

Rolls out a checkpointed SKRL policy under a :class:`~isaaclab.test.benchmark.BenchmarkMonitor`
and emits a :class:`~isaaclab.test.benchmark.schema.PlayBundle` JSON file. Dispatched from
``scripts/benchmarks/play.py`` via ``--rl_library skrl``.
"""

from __future__ import annotations

import sys
import time

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
    from isaaclab.test.benchmark._cli import parse_non_negative_int, parse_positive_int

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Benchmark RL inference (play) with SKRL.")
    help_requested = "-h" in argv or "--help" in argv
    parser.add_argument("--task", type=str, required=not help_requested, help="Gym task id to benchmark.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument(
        "--num_frames", type=parse_positive_int, default=100, help="Number of measured inference steps."
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Local or Nucleus checkpoint path to roll out; falls back to the published checkpoint when omitted.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help=(
            "Name of the RL agent configuration entry point. Defaults to None, in which"
            " case --algorithm is used to determine the default agent entry point."
        ),
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
        default="PPO",
        choices=["AMP", "PPO", "IPPO", "MAPPO"],
        help="The RL algorithm used for the skrl agent.",
    )
    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--measure_sync_step",
        action="store_true",
        help="Measure a serialized synchronized simulation and outside-simulation step breakdown.",
    )
    parser.add_argument(
        "--warmup_frames",
        type=parse_non_negative_int,
        default=1,
        help="Number of preceding env.step() calls to exclude from timing and throughput.",
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
    add_launcher_args(parser)

    args_cli, remaining_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + remaining_args

    return args_cli, remaining_args


def run(argv: list[str]) -> None:
    """Run the SKRL play benchmark and write a :class:`~isaaclab.test.benchmark.schema.PlayBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    import contextlib
    import os

    import gymnasium as gym

    from isaaclab.app import launch_simulation
    from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, stepping
    from isaaclab.test.benchmark.schema import StartupTime

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    # Importing the task packages registers their gym environments so the
    # requested ``--task`` can be resolved.
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    # PLACEHOLDER: Extension template (do not remove this comment)
    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    args_cli, remaining_args = _parse_args(argv)

    # Resolve agent entry point (mirrors isaaclab_rl.entrypoints.backends.play_skrl).
    if args_cli.agent is None:
        algorithm = args_cli.algorithm.lower()
        agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"
    else:
        agent_cfg_entry_point = args_cli.agent
        algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point)

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args_cli):
        app_t1 = time.perf_counter_ns()

        if args_cli.ml_framework.startswith("jax"):
            import skrl

            skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 0)
        env_cfg.seed = agent_cfg["seed"]

        log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
        if args_cli.checkpoint in _common.CHECKPOINT_SELECTORS:
            resume_path = _common.resolve_checkpoint_selector(
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
            resume_path = _common.resolve_play_checkpoint(args_cli.checkpoint, "skrl", args_cli.task)

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
                    {"name": "algorithm", "data": args_cli.algorithm},
                    {
                        "name": "environment_step_measurement_mode",
                        "data": ("serialized_synchronized" if args_cli.measure_sync_step else "host_return"),
                    },
                    {"name": "environment_step_warmup_frames", "data": args_cli.warmup_frames},
                    {"name": "presets", "data": ",".join(cfg.presets)},
                ]
            },
        )

        env_t0 = time.perf_counter_ns()
        env = gym.make(args_cli.task, cfg=env_cfg)
        env_t1 = time.perf_counter_ns()

        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        with contextlib.closing(env):
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
                warmup_steps=args_cli.warmup_frames,
            )
            total_frames = args_cli.warmup_frames + args_cli.num_frames
            with environment_step_timer, BenchmarkMonitor(benchmark, interval=1.0):
                all_step_times, reward, ep_length, success_rate = stepping.run_play_loop(env, policy, total_frames)

            first_step_s = all_step_times[0]
            step_times = all_step_times[args_cli.warmup_frames :]

            benchmark.update_manual_recorders()

            startup = StartupTime(
                app_launch=(app_t1 - app_t0) / 1e9,
                env_creation=(env_t1 - env_t0) / 1e9,
                first_step=first_step_s,
            )

            fps = [num_envs / t for t in step_times if t > 0]
            runtime = builders.build_runtime(
                startup_time_s=startup,
                iteration_times_s=step_times,
                collection_fps=fps,
                total_fps=fps,
                steps_per_iteration=num_envs,
                frames_per_environment_step=env.unwrapped.num_envs,
                environment_step_times_s=environment_step_timer.step_times_s,
                simulation_step_times_s=environment_step_timer.simulation_step_times_s,
                simulation_step_calls=environment_step_timer.simulation_step_calls,
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


if __name__ == "__main__":
    run(sys.argv[1:])
