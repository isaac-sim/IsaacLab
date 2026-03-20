# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to profile IsaacLab startup phases with cProfile.

Each startup stage (app launch, python imports, env creation, first step) is
wrapped in its own cProfile session. The top functions by own-time are emitted
as SingleMeasurement entries (both own-time and cumulative time) via the
standard benchmark backend.
"""

import argparse
import cProfile
import os
import sys
import time

from isaaclab.app import AppLauncher

# -- CLI arguments -----------------------------------------------------------

parser = argparse.ArgumentParser(description="Profile IsaacLab startup phases.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--top_n",
    type=int,
    default=None,
    help="Number of top functions per phase (default: 30, or 5 with --whitelist_config).",
)
parser.add_argument(
    "--benchmark_backend",
    type=str,
    default="omniperf",
    choices=[
        "json",
        "osmo",
        "omniperf",
        "summary",
        "LocalLogMetrics",
        "JSONFileMetrics",
        "OsmoKPIFile",
        "OmniPerfKPIFile",
    ],
    help="Benchmarking backend options, defaults omniperf",
)
parser.add_argument("--output_path", type=str, default=".", help="Path to output benchmark results.")
parser.add_argument(
    "--whitelist_config",
    type=str,
    default=None,
    help="Path to YAML file with per-phase function whitelist patterns. Overrides --top_n for listed phases.",
)

# append AppLauncher cli args (provides --device, --headless, etc.)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from isaaclab.test.benchmark import BaseIsaacLabBenchmark, SingleMeasurement
from isaaclab.utils.timer import Timer, TimerError

from scripts.benchmarks.utils import (
    get_backend_type,
    get_preset_string,
    parse_cprofile_stats,
)

# -- Python imports (profiled) ------------------------------------------------

imports_profile = cProfile.Profile()
imports_time_begin = time.perf_counter_ns()
imports_profile.enable()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402

from isaaclab_tasks.utils import launch_simulation, resolve_task_config  # noqa: E402

imports_profile.disable()

if torch.cuda.is_available() and torch.cuda.is_initialized():
    torch.cuda.synchronize()
imports_time_end = time.perf_counter_ns()

# -- Resolve task config (outside profiling) ---------------------------------

env_cfg, _agent_cfg = resolve_task_config(args_cli.task, None)

# -- Detect IsaacLab source prefixes for filtering ---------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_source_dir = os.path.join(_REPO_ROOT, "source")
if os.path.isdir(_source_dir):
    _ISAACLAB_PREFIXES = [
        os.path.join(_source_dir, d) for d in os.listdir(_source_dir) if os.path.isdir(os.path.join(_source_dir, d))
    ]
else:
    print(f"[WARNING] IsaacLab source directory not found at '{_source_dir}'. Function-level profiling will be empty.")
    _ISAACLAB_PREFIXES = []

# -- Load whitelist config if provided ---------------------------------------

_WHITELIST: dict[str, list[str]] = {}
if args_cli.whitelist_config is not None:
    import yaml

    try:
        with open(args_cli.whitelist_config) as f:
            raw = yaml.safe_load(f)
    except OSError as e:
        print(f"[ERROR] Cannot read whitelist config '{args_cli.whitelist_config}': {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Invalid YAML in whitelist config '{args_cli.whitelist_config}': {e}")
        sys.exit(1)

    if raw is None:
        _WHITELIST = {}
    elif not isinstance(raw, dict):
        print(
            f"[ERROR] Whitelist config must be a YAML mapping (got {type(raw).__name__})."
            " Expected format: phase_name: [pattern, ...]"
        )
        sys.exit(1)
    else:
        _VALID_PHASES = {"app_launch", "python_imports", "env_creation", "first_step"}
        unknown_phases = set(raw.keys()) - _VALID_PHASES
        if unknown_phases:
            print(
                f"[WARNING] Whitelist config contains unknown phase(s): {unknown_phases}. "
                f"Valid phases: {_VALID_PHASES}. Check for typos."
            )
        for phase_name, patterns in raw.items():
            if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
                print(
                    f"[ERROR] Whitelist phase '{phase_name}' must be a list of strings, "
                    f"got {type(patterns).__name__}. Check YAML formatting (use '- pattern' syntax)."
                )
                sys.exit(1)
        _WHITELIST = raw

# Resolve top_n default: 5 when using whitelist (fallback phases stay compact), 30 otherwise
if args_cli.top_n is None:
    args_cli.top_n = 5 if _WHITELIST else 30

# -- Create the benchmark instance ------------------------------------------

backend_type = get_backend_type(args_cli.benchmark_backend)
benchmark = BaseIsaacLabBenchmark(
    benchmark_name="benchmark_startup",
    backend_type=backend_type,
    output_path=args_cli.output_path,
    use_recorders=True,
    frametime_recorders=False,
    output_prefix=f"benchmark_startup_{args_cli.task}",
    workflow_metadata={
        "metadata": [
            {"name": "task", "data": args_cli.task},
            {"name": "seed", "data": args_cli.seed},
            {"name": "num_envs", "data": args_cli.num_envs},
            {"name": "top_n", "data": args_cli.top_n},
            {"name": "presets", "data": get_preset_string(hydra_args)},
        ]
    },
)


# -- Main profiling logic ---------------------------------------------------


def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    app_launch_profile: cProfile.Profile,
    app_launch_wall_ms: float,
):
    """Profile env creation and first step, then log all phase measurements.

    Args:
        env_cfg: Resolved environment configuration for the task.
        app_launch_profile: cProfile session from the app-launch phase.
        app_launch_wall_ms: Wall-clock duration of the app-launch phase [ms].
    """

    # Override config with CLI args
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed

    # -- Env creation (gym.make + env.reset) profiled ---------------------------

    env_creation_profile = cProfile.Profile()
    env_creation_time_begin = time.perf_counter_ns()
    env_creation_profile.enable()
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
    finally:
        env_creation_profile.disable()

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    env_creation_time_end = time.perf_counter_ns()

    try:
        # -- First step profiled ------------------------------------------------

        # Sample random actions
        actions = (
            torch.rand(env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)
            * 2.0
            - 1.0
        )

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

        # -- Parse all profiles and log measurements ----------------------------

        imports_wall_ms = (imports_time_end - imports_time_begin) / 1e6
        env_creation_wall_ms = (env_creation_time_end - env_creation_time_begin) / 1e6
        first_step_wall_ms = (first_step_time_end - first_step_time_begin) / 1e6

        # Collect Timer-based sub-timings for env_creation phase (may not exist for all environment types)
        scene_creation_ms = None
        try:
            scene_creation_ms = Timer.get_timer_info("scene_creation") * 1000
        except TimerError:
            print("[INFO] Timer 'scene_creation' not available; sub-timing will be omitted.")

        simulation_start_ms = None
        try:
            simulation_start_ms = Timer.get_timer_info("simulation_start") * 1000
        except TimerError:
            print("[INFO] Timer 'simulation_start' not available; sub-timing will be omitted.")

        phases = {
            "app_launch": {
                "profile": app_launch_profile,
                "wall_clock_ms": app_launch_wall_ms,
                "extra_measurements": [],
            },
            "python_imports": {
                "profile": imports_profile,
                "wall_clock_ms": imports_wall_ms,
                "extra_measurements": [],
            },
            "env_creation": {
                "profile": env_creation_profile,
                "wall_clock_ms": env_creation_wall_ms,
                "extra_measurements": [
                    (name, val)
                    for name, val in [
                        ("Scene Creation Time", scene_creation_ms),
                        ("Simulation Start Time", simulation_start_ms),
                    ]
                    if val is not None
                ],
            },
            "first_step": {
                "profile": first_step_profile,
                "wall_clock_ms": first_step_wall_ms,
                "extra_measurements": [],
            },
        }

        # Parse profiles and log measurements to benchmark
        for phase_name, phase_data in phases.items():
            phase_whitelist = _WHITELIST.get(phase_name)
            functions = parse_cprofile_stats(
                phase_data["profile"], _ISAACLAB_PREFIXES, top_n=args_cli.top_n, whitelist=phase_whitelist
            )
            wall_ms = phase_data["wall_clock_ms"]
            extras = phase_data["extra_measurements"]

            # Log wall-clock time
            benchmark.add_measurement(
                phase_name, measurement=SingleMeasurement(name="Wall Clock Time", value=wall_ms, unit="ms")
            )

            # Log extra sub-timings
            for extra_name, extra_val in extras:
                benchmark.add_measurement(
                    phase_name, measurement=SingleMeasurement(name=extra_name, value=extra_val, unit="ms")
                )

            # Log per-function measurements (tottime + cumtime)
            for label, tottime_ms, cumtime_ms in functions:
                benchmark.add_measurement(
                    phase_name, measurement=SingleMeasurement(name=label, value=round(tottime_ms, 2), unit="ms")
                )
                benchmark.add_measurement(
                    phase_name,
                    measurement=SingleMeasurement(name=f"{label} (cumtime)", value=round(cumtime_ms, 2), unit="ms"),
                )

        # Finalize benchmark output
        benchmark.update_manual_recorders()
        benchmark._finalize_impl()
    finally:
        env.close()


if __name__ == "__main__":
    # -- App launch (profiled) --------------------------------------------------

    app_launch_profile = cProfile.Profile()
    app_launch_time_begin = time.perf_counter_ns()
    app_launch_profile.enable()

    with launch_simulation(env_cfg, args_cli):
        app_launch_profile.disable()

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        app_launch_time_end = time.perf_counter_ns()

        app_launch_wall_ms = (app_launch_time_end - app_launch_time_begin) / 1e6
        main(env_cfg, app_launch_profile, app_launch_wall_ms)
