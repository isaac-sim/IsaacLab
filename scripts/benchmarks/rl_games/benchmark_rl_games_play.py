# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL-Games play-benchmark adapter.

Rolls out a checkpointed RL-Games policy under a :class:`~isaaclab.test.benchmark.BenchmarkMonitor`
and emits a :class:`~isaaclab.test.benchmark.schema.PlayBundle` JSON file. Dispatched from
``scripts/benchmarks/play.py`` via ``--rl_library rl_games``.
"""

from __future__ import annotations

import sys

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

    parser = argparse.ArgumentParser(description="Benchmark RL inference (play) with RL-Games.")
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
        "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
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
    """Run the RL-Games play benchmark and write a :class:`~isaaclab.test.benchmark.schema.PlayBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e. ``sys.argv[1:]``
            after the dispatcher has stripped ``--rl_library``).
    """
    import contextlib
    import math
    import os
    import re
    import time

    import gymnasium as gym
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    from isaaclab.app import launch_simulation
    from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, stepping
    from isaaclab.test.benchmark.schema import StartupTime

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

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
        if args_cli.seed is not None:
            agent_cfg["params"]["seed"] = args_cli.seed
        env_cfg.seed = agent_cfg["params"]["seed"]

        config_name = agent_cfg["params"]["config"]["name"]
        log_root_path = os.path.abspath(os.path.join("logs", "rl_games", config_name))
        if args_cli.checkpoint in _common.CHECKPOINT_SELECTORS:
            resume_path = _common.resolve_checkpoint_selector(
                log_root_path,
                args_cli.checkpoint,
                library="rl_games",
                task=args_cli.task,
                checkpoint_pattern=r".*\.pth",
                other_dirs=["nn"],
                preferred_checkpoint_pattern=rf"{re.escape(config_name)}\.pth",
                metadata={"agent": args_cli.agent},
            )
        else:
            resume_path = _common.resolve_play_checkpoint(args_cli.checkpoint, "rl_games", args_cli.task)

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

        # Read rl_games device / clipping config.
        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        env_t0 = time.perf_counter_ns()
        env = gym.make(args_cli.task, cfg=env_cfg)
        env_t1 = time.perf_counter_ns()

        # Wrap for rl_games (pass obs groups so tasks with non-default/asymmetric observation
        # layouts feed the policy the same observation it was trained on).
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

        # Register with rl_games vecenv registry.
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        num_envs = env.unwrapped.num_envs

        # Load the trained policy the same way isaaclab_rl.entrypoints.backends.play_rl_games does.
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = num_envs
        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()
        _ = agent.get_batch_size(agent.obs_to_torch(env.reset()), 1)
        if agent.is_rnn:
            agent.init_rnn()

        def policy(obs):
            """Map an observation batch to a deterministic action batch via the rl_games player.

            Mirrors the inference path in ``isaaclab_rl.entrypoints.backends.play_rl_games``:
            extracts the ``"obs"`` tensor from the wrapper's dict observation and runs the
            player's deterministic action.

            Args:
                obs: Observation returned by the rl_games-wrapped env (dict or tensor).

            Returns:
                The action tensor to feed ``env.step``.
            """
            if isinstance(obs, dict):
                obs = obs["obs"]
            obs = agent.obs_to_torch(obs)
            return agent.get_action(obs, is_deterministic=agent.is_deterministic)

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
        seed = agent_cfg["params"]["seed"] if agent_cfg["params"]["seed"] is not None else 0

        run_identity = builders.build_run_identity(
            run_id=capture.synth_run_id("rl_games", cfg.physics_backend, args_cli.task, seed, stamp),
            framework="rl_games",
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
