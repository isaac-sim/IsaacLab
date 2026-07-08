# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Profile Isaac Lab startup phases.

Each phase runs in an independent ``cProfile`` session. The schema formatter emits
a :class:`~isaaclab.test.benchmark.StartupBundle`; other selected
formatters receive equivalent measurement phases.

Profiled phases
---------------
* **app_launch**: :func:`~isaaclab.app.launch_simulation` context entry
  (simulation runtime initialization).
* **python_imports**: launcher, task registration, and runtime-library imports.
* **task_config**: :func:`~isaaclab_tasks.utils.resolve_task_config`.
* **env_creation**: :func:`gym.make` + ``env.reset()``.
* **first_step**: first ``env.step()`` call.

Usage example::

    uv run isaaclab benchmark startup \\
        --task Isaac-Cartpole-Direct \\
        --num_envs 16 \\
        presets=newton_mjwarp --headless
"""

from __future__ import annotations

import argparse
import cProfile
import os
import sys
import time
from datetime import datetime, timezone

_START_UTC = datetime.now(timezone.utc).isoformat()
_VALID_PHASES = {"app_launch", "python_imports", "task_config", "env_creation", "first_step"}


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse benchmark arguments and retain Hydra overrides in ``sys.argv``.

    Args:
        argv: Command-line arguments excluding the script path.

    Returns:
        Parsed arguments and the remaining Hydra overrides.
    """
    from isaaclab.app import add_launcher_args

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Profile Isaac Lab startup phases.")
    parser.add_argument("--task", type=str, required=True, help="Gym task id to profile.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Number of top cProfile functions per phase (default: 5 with whitelist, 30 otherwise).",
    )
    parser.add_argument(
        "--benchmark_formatter",
        type=str,
        default="schema",
        help=(
            "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
            " 'omniperf', 'osmo', 'json', 'summary'."
            " Example: 'schema,omniperf'."
        ),
    )
    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--whitelist_config",
        type=str,
        default=None,
        help="Path to YAML file with per-phase fnmatch patterns. Overrides --top_n for listed phases.",
    )
    add_launcher_args(parser)

    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + remaining
    return args, remaining


def _load_whitelist(path: str | None) -> dict[str, list[str]]:
    """Load and validate a startup profiling whitelist.

    Args:
        path: Path to the whitelist YAML file, or ``None``.

    Returns:
        Validated patterns keyed by startup phase.
    """
    if path is None:
        return {}

    import yaml

    try:
        with open(path) as whitelist_file:
            raw = yaml.safe_load(whitelist_file)
    except OSError as exc:
        print(f"[ERROR] Cannot read whitelist config '{path}': {exc}")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"[ERROR] Invalid YAML in whitelist config '{path}': {exc}")
        sys.exit(1)

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        print(
            f"[ERROR] Whitelist config must be a YAML mapping (got {type(raw).__name__})."
            " Expected format: phase_name: [pattern, ...]"
        )
        sys.exit(1)

    unknown = set(raw) - _VALID_PHASES
    if unknown:
        print(
            f"[WARNING] Whitelist config contains unknown phase(s): {unknown}. "
            f"Valid phases: {_VALID_PHASES}. Check for typos."
        )
    for phase_name, patterns in raw.items():
        if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
            print(
                f"[ERROR] Whitelist phase '{phase_name}' must be a list of strings, "
                f"got {type(patterns).__name__}. Check YAML formatting (use '- pattern' syntax)."
            )
            sys.exit(1)
    return raw


def _isaaclab_source_prefixes() -> list[str]:
    """Return package paths included in function-level startup profiles."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    source_dir = os.path.join(repo_root, "source")
    if not os.path.isdir(source_dir):
        print(
            f"[WARNING] IsaacLab source directory not found at '{source_dir}'. Function-level profiling will be empty."
        )
        return []
    return [
        os.path.join(source_dir, directory)
        for directory in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, directory))
    ]


def run(argv: list[str]) -> None:
    """Run the startup benchmark and write the selected formatter outputs.

    Args:
        argv: Command-line arguments excluding the script path.
    """
    imports_profile = cProfile.Profile()
    imports_time_begin = time.perf_counter_ns()
    imports_profile.enable()

    args, hydra_args = _parse_args(argv)

    import gymnasium as gym
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.test.benchmark import BaseIsaacLabBenchmark, builders, capture, stepping
    from isaaclab.test.benchmark.profiling import parse_cprofile_stats
    from isaaclab.test.benchmark.schema import CProfileFunction, StartupPhase

    from isaaclab_tasks.utils import resolve_task_config

    imports_profile.disable()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    imports_time_end = time.perf_counter_ns()

    task_config_profile = cProfile.Profile()
    task_config_time_begin = time.perf_counter_ns()
    task_config_profile.enable()
    try:
        env_cfg, _ = resolve_task_config(args.task, None)
    finally:
        task_config_profile.disable()
    task_config_time_end = time.perf_counter_ns()

    source_prefixes = _isaaclab_source_prefixes()
    whitelist = _load_whitelist(args.whitelist_config)
    if args.top_n is None:
        args.top_n = 5 if whitelist else 30

    app_launch_profile = cProfile.Profile()
    app_launch_time_begin = time.perf_counter_ns()
    app_launch_profile.enable()

    with launch_simulation(env_cfg, args):
        app_launch_profile.disable()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        app_launch_time_end = time.perf_counter_ns()

        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if args.device is not None:
            env_cfg.sim.device = args.device
        if args.seed is not None:
            env_cfg.seed = args.seed

        env = None
        env_creation_profile = cProfile.Profile()
        env_creation_time_begin = time.perf_counter_ns()
        try:
            env_creation_profile.enable()
            try:
                env = gym.make(args.task, cfg=env_cfg)
                env.reset()
            finally:
                env_creation_profile.disable()

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            env_creation_time_end = time.perf_counter_ns()

            actions = stepping.sample_random_actions(env)

            first_step_profile = cProfile.Profile()
            first_step_time_begin = time.perf_counter_ns()
            first_step_profile.enable()
            try:
                env.step(actions)
            finally:
                first_step_profile.disable()

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            first_step_time_end = time.perf_counter_ns()
            end_utc = capture.now_utc_iso()

            phase_profiles: dict[str, tuple[cProfile.Profile, float]] = {
                "app_launch": (app_launch_profile, (app_launch_time_end - app_launch_time_begin) / 1e6),
                "python_imports": (imports_profile, (imports_time_end - imports_time_begin) / 1e6),
                "task_config": (task_config_profile, (task_config_time_end - task_config_time_begin) / 1e6),
                "env_creation": (env_creation_profile, (env_creation_time_end - env_creation_time_begin) / 1e6),
                "first_step": (first_step_profile, (first_step_time_end - first_step_time_begin) / 1e6),
            }

            phases: dict[str, StartupPhase] = {}
            for phase_name, (profile, wall_ms) in phase_profiles.items():
                functions = parse_cprofile_stats(
                    profile, source_prefixes, top_n=args.top_n, whitelist=whitelist.get(phase_name)
                )
                phases[phase_name] = StartupPhase(
                    total_time_s=wall_ms / 1000.0,
                    top_functions=[
                        CProfileFunction(
                            name=label,
                            own_time_s=own_ms / 1000.0,
                            cum_time_s=cumulative_ms / 1000.0,
                            calls=calls,
                        )
                        for label, own_ms, cumulative_ms, calls in functions
                    ],
                )

            cfg = capture.run_config_from_presets(hydra_args, env_cfg=env_cfg)
            stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
            seed = args.seed if args.seed is not None else 0
            run_id = capture.synth_run_id(None, cfg.physics_backend, args.task, seed, stamp)
            run_identity = builders.build_run_identity(
                run_id=run_id,
                framework=None,
                config=cfg,
                task=args.task,
                seed=seed,
                start_utc=_START_UTC,
                end_utc=end_utc,
                num_envs=None,
                max_iterations=None,
            )

            benchmark = BaseIsaacLabBenchmark(
                benchmark_name="benchmark_startup",
                formatter_type=args.benchmark_formatter,
                output_path=args.output_path,
                use_recorders=True,
                output_prefix=f"startup_{args.task}",
                workflow_metadata={
                    "metadata": [
                        {"name": "task", "data": args.task},
                        {"name": "seed", "data": args.seed},
                        {"name": "num_envs", "data": args.num_envs},
                        {"name": "top_n", "data": args.top_n},
                        {"name": "presets", "data": ",".join(cfg.presets)},
                    ]
                },
            )
            benchmark.update_manual_recorders()

            bundle = builders.build_startup_bundle(
                run=run_identity,
                versions=capture.capture_versions(benchmark),
                hardware=capture.capture_hardware(benchmark),
                phases=phases,
                top_n=args.top_n,
                whitelist=args.whitelist_config,
            )
            benchmark.attach_bundle(bundle)
            benchmark._finalize_impl()
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    run(sys.argv[1:])
