# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Profile Isaac Lab startup phases.

Each phase runs in an independent ``cProfile`` session. The schema formatter emits
a :class:`~isaaclab.benchmark.StartupBundle`; other selected
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
        presets=newton_mjwarp

Use ``isaaclab benchmark startup-multigpu`` to profile rank 0 while every GPU launches
concurrently; see :mod:`isaaclab.benchmark.entrypoints.multigpu`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.benchmark import BenchmarkResult

import argparse
import cProfile
import importlib.util
import os
import sys
import time
from datetime import datetime, timezone

_PHASE_ORDER = ("python_imports", "task_config", "app_launch", "env_creation", "first_step")
_VALID_PHASES = set(_PHASE_ORDER)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse benchmark arguments and retain Hydra overrides in ``sys.argv``.

    Args:
        argv: Command-line arguments excluding the script path.

    Returns:
        Parsed arguments and the remaining Hydra overrides.
    """
    from isaaclab.app import add_launcher_args
    from isaaclab.benchmark.distributed import add_distributed_arg

    from isaaclab_tasks.utils import setup_preset_cli

    parser = argparse.ArgumentParser(description="Profile Isaac Lab startup phases.")
    help_requested = "-h" in argv or "--help" in argv
    parser.add_argument("--task", type=str, required=not help_requested, help="Gym task id to profile.")
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
    add_distributed_arg(parser)
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
    """Return installed Isaac Lab package roots included in startup profiles."""
    package_names = (
        "isaaclab",
        "isaaclab_assets",
        "isaaclab_contrib",
        "isaaclab_mimic",
        "isaaclab_newton",
        "isaaclab_ov",
        "isaaclab_physx",
        "isaaclab_rl",
        "isaaclab_tasks",
        "isaaclab_tasks_experimental",
        "isaaclab_teleop",
        "isaaclab_visualizers",
    )
    prefixes: list[str] = []
    for package_name in package_names:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            continue
        locations = spec.submodule_search_locations
        if locations is not None:
            prefixes.extend(os.path.realpath(location) for location in locations)
        elif spec.origin is not None:
            prefixes.append(os.path.realpath(os.path.dirname(spec.origin)))
    return list(dict.fromkeys(prefixes))


def _timer_totals(since: dict[str, float] | None = None) -> dict[str, float]:
    """Return the cumulative time [s] recorded by each named :class:`~isaaclab.utils.timer.Timer`.

    Args:
        since: Earlier snapshot to subtract, keeping only timers that advanced.

    Returns:
        Cumulative (or elapsed, when ``since`` is given) seconds keyed by timer name.
    """
    from isaaclab.utils.timer import Timer

    totals = {name: info["mean"] * info["n"] for name, info in Timer.timing_info.items()}
    if since is None:
        return totals
    elapsed = {name: total - since.get(name, 0.0) for name, total in totals.items()}
    return {name: seconds for name, seconds in elapsed.items() if seconds > 1e-6}


def run(argv: list[str]) -> BenchmarkResult | None:
    """Run the startup benchmark and write the selected formatter outputs.

    Args:
        argv: Command-line arguments excluding the script path.

    Returns:
        Completed startup result, or ``None`` on a distributed rank other than global rank 0.
    """
    start_utc = datetime.now(timezone.utc).isoformat()
    imports_profile = cProfile.Profile()
    imports_time_begin = time.perf_counter_ns()
    imports_profile.enable()

    args, hydra_args = _parse_args(argv)

    import gymnasium as gym
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.benchmark import BaseIsaacLabBenchmark, BenchmarkResult, builders, capture, console, stepping
    from isaaclab.benchmark.distributed import DistributedContext
    from isaaclab.benchmark.profiling import parse_cprofile_stats
    from isaaclab.benchmark.schema import CProfileFunction, StartupPhase

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
    distributed = DistributedContext.from_env(args.distributed, workflow="startup")

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
        # A distributed launch already pinned this rank to its own GPU; honoring --device here
        # would move every rank onto the same one.
        if args.device is not None and not distributed.enabled:
            env_cfg.sim.device = args.device
        if args.seed is not None:
            env_cfg.seed = args.seed

        env = None
        env_creation_profile = cProfile.Profile()
        env_creation_time_begin = time.perf_counter_ns()
        try:
            timers_before = _timer_totals()
            env_creation_profile.enable()
            try:
                env = gym.make(args.task, cfg=env_cfg)
                env_reset_time_begin = time.perf_counter_ns()
                env.reset()
                env_reset_time_end = time.perf_counter_ns()
            finally:
                env_creation_profile.disable()
            env_creation_detail = _timer_totals(since=timers_before)
            env_creation_detail["env_reset"] = (env_reset_time_end - env_reset_time_begin) / 1e9

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            env_creation_time_end = time.perf_counter_ns()

            actions = stepping.sample_random_actions(env)

            first_step_profile = cProfile.Profile()
            first_step_time_begin = time.perf_counter_ns()
            first_step_profile.enable()
            try:
                with torch.inference_mode():
                    env.step(actions)
            finally:
                first_step_profile.disable()

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            first_step_time_end = time.perf_counter_ns()
            end_utc = capture.now_utc_iso()

            # Every rank profiles its own startup so the ranks contend as they would in a real
            # distributed launch, but only rank 0 reports.
            if not distributed.is_main:
                return None

            # Ordered chronologically: the bundle and the console summary preserve this order.
            phase_profiles: dict[str, tuple[cProfile.Profile, float]] = {
                "python_imports": (imports_profile, (imports_time_end - imports_time_begin) / 1e6),
                "task_config": (task_config_profile, (task_config_time_end - task_config_time_begin) / 1e6),
                "app_launch": (app_launch_profile, (app_launch_time_end - app_launch_time_begin) / 1e6),
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
                start_utc=start_utc,
                end_utc=end_utc,
                num_envs=env.unwrapped.num_envs,
                max_iterations=None,
            )

            benchmark = BaseIsaacLabBenchmark(
                benchmark_name="benchmark_startup",
                formatter_type=args.benchmark_formatter,
                output_path=args.output_path,
                use_recorders=True,
                output_prefix=f"benchmark_startup{'_multigpu' if distributed.enabled else ''}_{args.task}",
                workflow_metadata={
                    "metadata": [
                        {"name": "task", "data": args.task},
                        {"name": "seed", "data": args.seed},
                        {"name": "num_envs", "data": args.num_envs},
                        {"name": "top_n", "data": args.top_n},
                        {"name": "presets", "data": ",".join(cfg.presets)},
                        {"name": "world_size", "data": distributed.world_size},
                    ]
                },
            )
            benchmark.update_manual_recorders()

            extra: dict[str, float | int | str | bool] = {
                f"env_creation.{name}_s": seconds for name, seconds in env_creation_detail.items()
            }
            if distributed.enabled:
                extra.update(distributed.bundle_metadata(workload_scope="rank0"))

            bundle = builders.build_startup_bundle(
                run=run_identity,
                versions=capture.capture_versions(benchmark),
                hardware=capture.capture_hardware(benchmark),
                phases=phases,
                top_n=args.top_n,
                whitelist=args.whitelist_config,
                extra=extra,
            )
            benchmark.attach_bundle(bundle)
            output_paths = benchmark.finalize()
            result = BenchmarkResult(bundle=bundle, output_paths=output_paths)
            console.print_startup_report(bundle, output_paths, env_creation_detail)
        finally:
            if env is not None:
                env.close()

    return result


if __name__ == "__main__":
    run(sys.argv[1:])
