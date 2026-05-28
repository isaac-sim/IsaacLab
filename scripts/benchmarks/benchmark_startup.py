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
from datetime import datetime, timezone

from isaaclab.app import AppLauncher

from isaaclab_tasks.utils import fold_preset_tokens, setup_preset_cli

# Wall-clock start of the entire script, captured as early as possible so the
# startup bundle can report a total duration that covers all phases.
_SCRIPT_START_DT = datetime.now(timezone.utc)

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
parser.add_argument(
    "--schema_v1_output",
    type=str,
    default=None,
    help="If set, write a schema-v1 startup.json to this path.",
)
parser.add_argument(
    "--backend",
    choices=["physx", "newton"],
    default=None,
    help="Physics backend tag recorded in the bundle. Defaults to 'physx' if omitted.",
)
parser.add_argument(
    "--run_id",
    type=str,
    default=None,
    help="Run identity string to embed in the bundle. If omitted, a synthetic run_id is generated.",
)

# append AppLauncher cli args (provides --device, --headless, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
hydra_args = fold_preset_tokens(hydra_args)
sys.argv = [sys.argv[0]] + hydra_args

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from isaaclab.test.benchmark import BaseIsaacLabBenchmark, SingleMeasurement
from isaaclab.utils.timer import Timer, TimerError

from scripts.benchmarks._schema_helpers import capture_hardware, capture_versions, synth_run_id
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

from scripts.benchmarks._action_sampling import sample_random_actions  # noqa: E402

imports_profile.disable()

if torch.cuda.is_available() and torch.cuda.is_initialized():
    torch.cuda.synchronize()
imports_time_end = time.perf_counter_ns()

# -- Resolve task config (profiled) ------------------------------------------

task_config_profile = cProfile.Profile()
task_config_time_begin = time.perf_counter_ns()
task_config_profile.enable()

env_cfg, _agent_cfg = resolve_task_config(args_cli.task, None)

task_config_profile.disable()
task_config_time_end = time.perf_counter_ns()

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
        _VALID_PHASES = {"app_launch", "python_imports", "task_config", "env_creation", "first_step"}
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

env_cfg.seed = args_cli.seed if args_cli.seed is not None else env_cfg.seed

backend_type = get_backend_type(args_cli.benchmark_backend)
benchmark = BaseIsaacLabBenchmark(
    benchmark_name="benchmark_startup",
    backend_type=backend_type,
    output_path=args_cli.output_path,
    use_recorders=True,
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


# -- Schema v1 helpers ------------------------------------------------------


def _build_startup_bundle(
    phases_data: dict,
    run_start_dt: datetime,
    run_end_dt: datetime,
    status: str,
    versions,
    hardware,
):
    """Build a schema-v1 StartupBundle from the collected phase data.

    Args:
        phases_data: The same ``phases`` dict ``main()`` builds for legacy logging.
        run_start_dt: UTC timestamp when the whole script started.
        run_end_dt: UTC timestamp when the whole script finished.
        status: Completion status of the run (``"completed"`` or ``"crashed"``).
        versions: Pre-captured :class:`Versions` (must be captured before
            ``benchmark._finalize_impl()`` which clears the recorders).
        hardware: Pre-captured :class:`Hardware`.

    Returns:
        A :class:`StartupBundle` ready to be passed to :func:`write_bundle_file`.
    """
    from isaaclab.benchmark.schema import (
        CProfileFunction,
        StartupBundle,
        StartupConfig,
        StartupPhase,
        StartupRunIdentity,
    )

    # Startup profiling is framework-agnostic; callers that wrap multiple
    # framework runs pass the real framework via --run_id. We record "rsl_rl"
    # as a schema placeholder when invoked standalone (the field is required).
    framework = "rsl_rl"
    backend = args_cli.backend or "physx"

    phases_out: dict[str, StartupPhase] = {}
    for name, data in phases_data.items():
        top_funcs: list[CProfileFunction] = []
        for label, tottime_ms, cumtime_ms, ncalls in parse_cprofile_stats(
            data["profile"], _ISAACLAB_PREFIXES, top_n=args_cli.top_n, whitelist=_WHITELIST.get(name)
        ):
            top_funcs.append(
                CProfileFunction(
                    name=label,
                    own_time_s=tottime_ms / 1000.0,
                    cum_time_s=cumtime_ms / 1000.0,
                    calls=ncalls,
                )
            )
        phases_out[name] = StartupPhase(
            total_time_s=data["wall_clock_ms"] / 1000.0,
            top_functions=top_funcs,
        )

    seed = args_cli.seed if args_cli.seed is not None else 0
    run_id = args_cli.run_id or synth_run_id(framework, backend, args_cli.task, seed)

    return StartupBundle(
        run=StartupRunIdentity(
            run_id=run_id,
            framework=framework,
            backend=backend,
            task=args_cli.task,
            seed=seed,
            start_time_utc=run_start_dt.isoformat().replace("+00:00", "Z"),
            end_time_utc=run_end_dt.isoformat().replace("+00:00", "Z"),
            duration_s=(run_end_dt - run_start_dt).total_seconds(),
            status=status,
        ),
        versions=versions,
        hardware=hardware,
        phases=phases_out,
        config=StartupConfig(top_n=args_cli.top_n, whitelist=args_cli.whitelist_config),
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
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else env_cfg.seed

    # -- Env creation (gym.make + env.reset) profiled ---------------------------

    env = None
    env_creation_profile = cProfile.Profile()
    env_creation_time_begin = time.perf_counter_ns()
    env_creation_profile.enable()
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
    finally:
        env_creation_profile.disable()

    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        env_creation_time_end = time.perf_counter_ns()
        # -- First step profiled ------------------------------------------------

        # Sample random actions from the action space(s). Returns a tensor for
        # single-agent envs and a per-agent dict for multi-agent (DirectMARLEnv)
        # envs — env.step accepts the matching shape.
        actions = sample_random_actions(env)

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
        task_config_wall_ms = (task_config_time_end - task_config_time_begin) / 1e6
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
            "task_config": {
                "profile": task_config_profile,
                "wall_clock_ms": task_config_wall_ms,
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
            for label, tottime_ms, cumtime_ms, _ncalls in functions:
                benchmark.add_measurement(
                    phase_name, measurement=SingleMeasurement(name=label, value=round(tottime_ms, 2), unit="ms")
                )
                benchmark.add_measurement(
                    phase_name,
                    measurement=SingleMeasurement(name=f"{label} (cumtime)", value=round(cumtime_ms, 2), unit="ms"),
                )

        # Capture versions/hardware BEFORE finalize, which clears the recorders.
        versions_v1 = None
        hardware_v1 = None
        if args_cli.schema_v1_output is not None:
            benchmark.update_manual_recorders()
            versions_v1 = capture_versions(benchmark)
            hardware_v1 = capture_hardware(benchmark)

        # Finalize benchmark output (nulls out _manual_recorders).
        benchmark.update_manual_recorders()
        benchmark._finalize_impl()

        if args_cli.schema_v1_output is not None:
            from isaaclab.benchmark.schema import write_bundle_file

            bundle = _build_startup_bundle(
                phases,
                _SCRIPT_START_DT,
                datetime.now(timezone.utc),
                status="completed",
                versions=versions_v1,
                hardware=hardware_v1,
            )
            write_bundle_file(bundle, args_cli.schema_v1_output)
    finally:
        if env is not None:
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
