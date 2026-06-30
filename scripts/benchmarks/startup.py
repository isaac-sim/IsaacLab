# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Profile Isaac Lab startup phases.

Each phase runs in an independent ``cProfile`` session. The schema formatter emits
a :class:`~isaaclab.test.benchmark.schema.StartupBundle`; other selected
formatters receive equivalent measurement phases.

Profiled phases
---------------
* **app_launch**: :func:`~isaaclab.app.launch_simulation` call (Isaac Sim
  fabric startup).
* **python_imports**: launcher, task registration, and runtime-library imports.
* **task_config**: :func:`~isaaclab_tasks.utils.resolve_task_config`.
* **env_creation**: :func:`gym.make` + ``env.reset()``.
* **first_step**: first ``env.step()`` call.

Usage example::

    ./isaaclab.sh -p scripts/benchmarks/startup.py \\
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

_start_utc = datetime.now(timezone.utc).isoformat()
_imports_profile = cProfile.Profile()
_imports_time_begin = time.perf_counter_ns()
_imports_profile.enable()

from isaaclab.app import AppLauncher

from isaaclab_tasks.utils import setup_preset_cli

_parser = argparse.ArgumentParser(description="Profile Isaac Lab startup phases.")
_parser.add_argument("--task", type=str, required=True, help="Gym task id to profile.")
_parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
_parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
_parser.add_argument(
    "--top_n",
    type=int,
    default=None,
    help="Number of top cProfile functions per phase (default: 5 with whitelist, 30 otherwise).",
)
_parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="schema",
    help=(
        "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
        " 'omniperf', 'osmo', 'json', 'summary'."
        " Example: 'schema,omniperf'."
    ),
)
_parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
_parser.add_argument(
    "--whitelist_config",
    type=str,
    default=None,
    help="Path to YAML file with per-phase fnmatch patterns. Overrides --top_n for listed phases.",
)

AppLauncher.add_app_launcher_args(_parser)
args_cli, _hydra_args = setup_preset_cli(_parser)
sys.argv = [sys.argv[0]] + _hydra_args


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.test.benchmark import (  # noqa: E402
    BaseIsaacLabBenchmark,  # noqa: E402
    builders,
    capture,
    stepping,
)
from isaaclab.test.benchmark.profiling import parse_cprofile_stats  # noqa: E402
from isaaclab.test.benchmark.schema import CProfileFunction, StartupPhase  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

_imports_profile.disable()

if torch.cuda.is_available() and torch.cuda.is_initialized():
    torch.cuda.synchronize()
_imports_time_end = time.perf_counter_ns()


_task_config_profile = cProfile.Profile()
_task_config_time_begin = time.perf_counter_ns()
_task_config_profile.enable()

_env_cfg, _agent_cfg = resolve_task_config(args_cli.task, None)

_task_config_profile.disable()
_task_config_time_end = time.perf_counter_ns()


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_source_dir = os.path.join(_REPO_ROOT, "source")
if os.path.isdir(_source_dir):
    _ISAACLAB_PREFIXES = [
        os.path.join(_source_dir, d) for d in os.listdir(_source_dir) if os.path.isdir(os.path.join(_source_dir, d))
    ]
else:
    print(f"[WARNING] IsaacLab source directory not found at '{_source_dir}'. Function-level profiling will be empty.")
    _ISAACLAB_PREFIXES = []


_WHITELIST: dict[str, list[str]] = {}
if args_cli.whitelist_config is not None:
    import yaml

    try:
        with open(args_cli.whitelist_config) as _wf:
            _raw = yaml.safe_load(_wf)
    except OSError as e:
        print(f"[ERROR] Cannot read whitelist config '{args_cli.whitelist_config}': {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Invalid YAML in whitelist config '{args_cli.whitelist_config}': {e}")
        sys.exit(1)

    if _raw is None:
        _WHITELIST = {}
    elif not isinstance(_raw, dict):
        print(
            f"[ERROR] Whitelist config must be a YAML mapping (got {type(_raw).__name__})."
            " Expected format: phase_name: [pattern, ...]"
        )
        sys.exit(1)
    else:
        _VALID_PHASES = {"app_launch", "python_imports", "task_config", "env_creation", "first_step"}
        _unknown = set(_raw.keys()) - _VALID_PHASES
        if _unknown:
            print(
                f"[WARNING] Whitelist config contains unknown phase(s): {_unknown}. "
                f"Valid phases: {_VALID_PHASES}. Check for typos."
            )
        for _phase_name, _patterns in _raw.items():
            if not isinstance(_patterns, list) or not all(isinstance(_p, str) for _p in _patterns):
                print(
                    f"[ERROR] Whitelist phase '{_phase_name}' must be a list of strings, "
                    f"got {type(_patterns).__name__}. Check YAML formatting (use '- pattern' syntax)."
                )
                sys.exit(1)
        _WHITELIST = _raw

if args_cli.top_n is None:
    args_cli.top_n = 5 if _WHITELIST else 30


def _run_main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    app_launch_profile: cProfile.Profile,
    app_launch_wall_ms: float,
) -> None:
    """Profile environment creation and the first step while simulation is active.

    Args:
        env_cfg: Resolved environment configuration for the task.
        app_launch_profile: Completed cProfile session for the app-launch phase.
        app_launch_wall_ms: Wall-clock duration of the app-launch phase [ms].
    """
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    env = None
    env_creation_profile = cProfile.Profile()
    env_creation_time_begin = time.perf_counter_ns()
    try:
        env_creation_profile.enable()
        try:
            env = gym.make(args_cli.task, cfg=env_cfg)
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

        imports_wall_ms = (_imports_time_end - _imports_time_begin) / 1e6
        task_config_wall_ms = (_task_config_time_end - _task_config_time_begin) / 1e6
        env_creation_wall_ms = (env_creation_time_end - env_creation_time_begin) / 1e6
        first_step_wall_ms = (first_step_time_end - first_step_time_begin) / 1e6

        _phase_raw: dict[str, tuple[cProfile.Profile, float]] = {
            "app_launch": (app_launch_profile, app_launch_wall_ms),
            "python_imports": (_imports_profile, imports_wall_ms),
            "task_config": (_task_config_profile, task_config_wall_ms),
            "env_creation": (env_creation_profile, env_creation_wall_ms),
            "first_step": (first_step_profile, first_step_wall_ms),
        }

        phases: dict[str, StartupPhase] = {}
        for phase_name, (profile, wall_ms) in _phase_raw.items():
            functions = parse_cprofile_stats(
                profile, _ISAACLAB_PREFIXES, top_n=args_cli.top_n, whitelist=_WHITELIST.get(phase_name)
            )
            phases[phase_name] = StartupPhase(
                total_time_s=wall_ms / 1000.0,
                top_functions=[
                    CProfileFunction(
                        name=lbl,
                        own_time_s=tot_ms / 1000.0,
                        cum_time_s=cum_ms / 1000.0,
                        calls=ncalls,
                    )
                    for (lbl, tot_ms, cum_ms, ncalls) in functions
                ],
            )

        cfg = capture.run_config_from_presets(_hydra_args)

        end_utc = capture.now_utc_iso()
        stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]

        seed = args_cli.seed if args_cli.seed is not None else 0
        run_id = capture.synth_run_id(None, cfg.physics_backend, args_cli.task, seed, stamp)

        run = builders.build_run_identity(
            run_id=run_id,
            framework=None,
            config=cfg,
            task=args_cli.task,
            seed=seed,
            start_utc=_start_utc,
            end_utc=end_utc,
            num_envs=None,
            max_iterations=None,
        )

        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_startup",
            formatter_type=args_cli.benchmark_formatter,
            output_path=args_cli.output_path,
            use_recorders=True,
            output_prefix=f"startup_{args_cli.task}",
            workflow_metadata={
                "metadata": [
                    {"name": "task", "data": args_cli.task},
                    {"name": "seed", "data": args_cli.seed},
                    {"name": "num_envs", "data": args_cli.num_envs},
                    {"name": "top_n", "data": args_cli.top_n},
                    {"name": "presets", "data": ",".join(cfg.presets)},
                ]
            },
        )

        benchmark.update_manual_recorders()

        versions = capture.capture_versions(benchmark)
        hardware = capture.capture_hardware(benchmark)

        bundle = builders.build_startup_bundle(
            run=run,
            versions=versions,
            hardware=hardware,
            phases=phases,
            top_n=args_cli.top_n,
            whitelist=args_cli.whitelist_config,
        )

        benchmark.attach_bundle(bundle)

        benchmark._finalize_impl()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    _app_launch_profile = cProfile.Profile()
    _app_launch_time_begin = time.perf_counter_ns()
    _app_launch_profile.enable()

    with launch_simulation(_env_cfg, args_cli):
        _app_launch_profile.disable()

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        _app_launch_time_end = time.perf_counter_ns()

        _app_launch_wall_ms = (_app_launch_time_end - _app_launch_time_begin) / 1e6
        _run_main(_env_cfg, _app_launch_profile, _app_launch_wall_ms)
