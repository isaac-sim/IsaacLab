# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL play-benchmark adapter.

Rolls out a checkpointed RSL-RL policy under a :class:`~isaaclab.test.benchmark.BenchmarkMonitor`
and emits a :class:`~isaaclab.test.benchmark.schema.PlayBundle` JSON file. Dispatched from
``scripts/benchmarks/play.py`` via ``--rl_library rsl_rl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parents[1]
_RL_SCRIPTS = _BENCH_DIR.parent / "reinforcement_learning"
if str(_RL_SCRIPTS) not in sys.path:
    # Shared training utilities remain script-local, so their directory must be on sys.path.
    sys.path.insert(0, str(_RL_SCRIPTS))

import common as _common  # noqa: E402


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

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Benchmark RL inference (play) with RSL-RL.")
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
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
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

    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + remaining
    return args, remaining


def run(argv: list[str]) -> None:
    """Run the RSL-RL play benchmark and write a :class:`~isaaclab.test.benchmark.schema.PlayBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    import contextlib
    import importlib.metadata as metadata
    import os
    import time

    import gymnasium as gym
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, stepping
    from isaaclab.test.benchmark.schema import StartupTime

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

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args):
        app_t1 = time.perf_counter_ns()

        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if args.seed is not None:
            agent_cfg.seed = args.seed
        env_cfg.seed = agent_cfg.seed

        installed_rsl_rl = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl)

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args.checkpoint in _common.CHECKPOINT_SELECTORS:
            resume_path = _common.resolve_checkpoint_selector(
                log_root_path,
                args.checkpoint,
                library="rsl_rl",
                task=args.task,
                checkpoint_pattern=r"model_.*\.pt",
                metadata={"agent": args.agent},
            )
        else:
            resume_path = _common.resolve_play_checkpoint(args.checkpoint, "rsl_rl", args.task)

        cfg = capture.run_config_from_presets(remaining)
        formatter_types = [value.strip() for value in args.benchmark_formatter.split(",") if value.strip()]
        formatter_types = formatter_types or ["omniperf"]

        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_play",
            formatter_type=formatter_types,
            output_path=args.output_path,
            use_recorders=True,
            frametime_recorders=any(t in ("summary", "omniperf") for t in formatter_types),
            output_prefix=f"benchmark_play_{args.task}",
            workflow_metadata={
                "metadata": [
                    {"name": "task", "data": args.task},
                    {"name": "num_envs", "data": args.num_envs},
                    {"name": "num_frames", "data": args.num_frames},
                    {"name": "presets", "data": ",".join(cfg.presets)},
                ]
            },
        )

        env_t0 = time.perf_counter_ns()
        env = gym.make(args.task, cfg=env_cfg)
        env_t1 = time.perf_counter_ns()

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        num_envs = env.unwrapped.num_envs

        # Load the trained policy the same way scripts/reinforcement_learning/rsl_rl/play.py does.
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        with BenchmarkMonitor(benchmark, interval=1.0):
            step_times, reward, ep_length, success_rate = stepping.run_play_loop(env, policy, args.num_frames)

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
        seed = agent_cfg.seed if agent_cfg.seed is not None else 0

        run_identity = builders.build_run_identity(
            run_id=capture.synth_run_id("rsl_rl", cfg.physics_backend, args.task, seed, stamp),
            framework="rsl_rl",
            config=cfg,
            task=args.task,
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
