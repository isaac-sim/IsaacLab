# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to benchmark RL agent with SKRL.

Mirrors :mod:`scripts.benchmarks.benchmark_rsl_rl` but uses SKRL's PPO Runner.
The v1.0 ``training.json`` output is identical in shape; only the
``framework`` field switches to ``"skrl"``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


def _native_backend_matches(raw_cfg, requested: str) -> bool:
    """Return ``True`` iff ``raw_cfg.sim.physics`` matches the requested backend.

    Returns ``False`` for sim-level :class:`PresetCfg` wrappers: presets carry
    multiple backends and the preset system handles selection downstream.
    """
    sim = getattr(raw_cfg, "sim", None)
    if sim is None:
        return False
    from isaaclab_tasks.utils.hydra import PresetCfg

    if isinstance(sim, PresetCfg):
        return False  # preset system handles it; presets_available is the source of truth
    physics = getattr(sim, "physics", None)
    # SimulationCfg.physics defaults to None which means PhysxCfg().
    if physics is None:
        return requested == "physx"
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    try:
        from isaaclab_ovphysx.physics import OvPhysxCfg
    except ImportError:
        OvPhysxCfg = None
    if isinstance(physics, PhysxCfg):
        return requested == "physx"
    if isinstance(physics, NewtonCfg):
        return requested == "newton"
    if OvPhysxCfg is not None and isinstance(physics, OvPhysxCfg):
        return requested == "ovphysx"
    return False


# -- CLI arguments -----------------------------------------------------------

parser = argparse.ArgumentParser(description="Benchmark an RL agent with SKRL.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10, help="RL policy training iterations.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the SKRL agent.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the SKRL agent.",
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
    "--backend",
    choices=["physx", "newton"],
    default=None,
    help=(
        "Physics backend to run with. Drives both the bundle tag and "
        "hydra `presets=<backend>`. Pass an explicit `presets=...` on "
        "the CLI to override."
    ),
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help=(
        "Absolute path where the training framework writes its outputs "
        "(TB events, checkpoints, params). When unset, falls back to "
        "the default logs/<framework>/<experiment>/<timestamp>/ path. "
        "Useful for downstream tooling that wants to collect outputs "
        "into a pre-allocated directory."
    ),
)
parser.add_argument(
    "--run_id",
    type=str,
    default=None,
    help="Run identity string to embed in the bundle. If omitted, a synthetic run_id is generated.",
)
parser.add_argument(
    "--schema_v1_output",
    type=str,
    default=None,
    help="If set, write a schema-v1 training.json to this path.",
)
parser.add_argument(
    "--ema_alpha",
    type=float,
    default=0.05,
    help="EMA smoothing factor for reward/ep_length (default 0.05, ~20-sample window).",
)
parser.add_argument(
    "--no_series",
    action="store_true",
    default=False,
    help="Omit per-iteration series from training.json (leaves final_raw + final_ema only).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Map --backend X to hydra presets=X so the physics preset is applied
# at config-resolve time.  Validate the request first: if the task does
# not advertise an X physics preset, exit fast with a stable
# ``preset_unsupported`` stderr prefix.  An explicit presets=... on the
# CLI bypasses validation (operator override).
if args_cli.backend is not None:
    existing_presets = [a for a in hydra_args if a.startswith("presets=")]
    if existing_presets:
        print(f"[WARNING] --backend={args_cli.backend} ignored because {existing_presets[0]} was explicitly passed.")
    else:
        from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
        from isaaclab_tasks.utils.preset_target import PresetTarget

        preset_map = enumerate_task_presets(args_cli.task)
        physics_presets = preset_map.get(PresetTarget.PHYSICS, []) if preset_map is not None else []
        if args_cli.backend in physics_presets:
            hydra_args = [f"presets={args_cli.backend}"] + hydra_args
        else:
            # No advertised <backend> physics preset. The task may still run on
            # that backend natively (sim.physics is already that type), in which
            # case no injection is needed; otherwise the request is unsupported.
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

            try:
                _raw_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
            except Exception as exc:  # noqa: BLE001 — fall through to unchecked injection
                print(
                    f"[WARNING] could not load raw cfg for {args_cli.task!r} "
                    f"to validate preset support ({type(exc).__name__}: {exc}); "
                    f"injecting presets={args_cli.backend} unchecked.",
                    file=sys.stderr,
                )
                hydra_args = [f"presets={args_cli.backend}"] + hydra_args
            else:
                if _native_backend_matches(_raw_cfg, args_cli.backend):
                    print(
                        f"[INFO] task {args_cli.task!r} has no '{args_cli.backend}' "
                        f"preset; running on native {args_cli.backend} backend (no "
                        f"injection).",
                        file=sys.stderr,
                    )
                    # No injection — hydra_args unchanged.
                else:
                    sys.stderr.write(
                        f"[ERROR] preset_unsupported: task {args_cli.task!r} has no "
                        f"{args_cli.backend!r} physics preset and does not run on "
                        f"{args_cli.backend!r} natively.\n"
                    )
                    sys.exit(2)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

imports_time_begin = time.perf_counter_ns()

from datetime import datetime, timezone

_SCRIPT_START_DT = datetime.now(timezone.utc)

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import launch_simulation, resolve_task_config

imports_time_end = time.perf_counter_ns()

from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor
from isaaclab.utils.timer import Timer

from scripts.benchmarks._schema_helpers import capture_hardware, capture_versions, synth_run_id
from scripts.benchmarks.utils import (
    get_backend_type,
    get_preset_string,
    log_app_start_time,
    log_python_imports_time,
    log_rl_policy_episode_lengths,
    log_rl_policy_rewards,
    log_runtime_step_times,
    log_scene_creation_time,
    log_simulation_start_time,
    log_task_start_time,
    log_total_start_time,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Resolve SKRL agent entry point (matches scripts/reinforcement_learning/skrl/train.py).
# For multi-agent (DirectMARLEnv) tasks, plain PPO can't be used — its observation
# preprocessor and policy assume a single Tensor obs, but DirectMARLEnv emits a
# per-agent dict. Auto-promote the default ``ppo`` algorithm to ``ippo`` so the
# task gets the right multi-agent variant; explicit ``--algorithm`` overrides
# (e.g. user passes ``mappo``) are honoured as-is.
_algorithm = args_cli.algorithm.lower()
if _algorithm == "ppo":
    try:
        from isaaclab.envs import DirectMARLEnvCfg as _DirectMARLEnvCfg

        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry as _peek_cfg

        _peek = _peek_cfg(args_cli.task, "env_cfg_entry_point")
        if isinstance(_peek, type) and issubclass(_peek, _DirectMARLEnvCfg):
            _is_marl = True
        else:
            _is_marl = isinstance(_peek, _DirectMARLEnvCfg)
    except Exception:  # noqa: BLE001 — best-effort detection; fall through to PPO if peek fails
        _is_marl = False
    if _is_marl:
        print(f"[INFO] {args_cli.task!r} is a multi-agent task; promoting --algorithm ppo -> ippo.", file=sys.stderr)
        _algorithm = "ippo"
_agent_cfg_entry_point = "skrl_cfg_entry_point" if _algorithm == "ppo" else f"skrl_{_algorithm}_cfg_entry_point"

backend_type = get_backend_type(args_cli.benchmark_backend)
benchmark = BaseIsaacLabBenchmark(
    benchmark_name="benchmark_skrl_train",
    backend_type=backend_type,
    output_path=args_cli.output_path,
    use_recorders=True,
    frametime_recorders=backend_type in ("summary", "omniperf"),
    output_prefix=f"benchmark_skrl_train_{args_cli.task}",
    workflow_metadata={
        "metadata": [
            {"name": "task", "data": args_cli.task},
            {"name": "seed", "data": args_cli.seed},
            {"name": "num_envs", "data": args_cli.num_envs},
            {"name": "max_iterations", "data": args_cli.max_iterations},
            {"name": "algorithm", "data": args_cli.algorithm},
            {"name": "presets", "data": get_preset_string(hydra_args)},
        ]
    },
)


def _compute_ema(series: list[float], alpha: float) -> float:
    """Exponentially weighted moving average over a per-iteration series.

    Args:
        series: Per-iteration scalar values.
        alpha: Smoothing factor in [0, 1]; smaller values give more smoothing.

    Returns:
        Final EMA value; 0.0 for an empty series.
    """
    if not series:
        return 0.0
    ema = float(series[0])
    for y in series[1:]:
        ema = alpha * float(y) + (1.0 - alpha) * ema
    return ema


def _find_measurement(measurements, name: str) -> float | None:
    """Return the value of the first SingleMeasurement with matching ``name``."""
    for meas in measurements:
        if meas.name == name:
            return float(meas.value)
    return None


def _capture_resources(bm: BaseIsaacLabBenchmark):
    """Build a schema-v1 :class:`Resources` from GPU/CPU/Memory recorders."""
    from isaaclab.benchmark.schema import MeanStd, MeanStdPeak, Resources

    gpu_m = bm._manual_recorders["GPUInfo"].get_data().measurements
    cpu_m = bm._manual_recorders["CPUInfo"].get_data().measurements
    mem_m = bm._manual_recorders["MemoryInfo"].get_data().measurements

    gpu_util_mean = _find_measurement(gpu_m, "GPU Utilization") or 0.0
    gpu_util_std = _find_measurement(gpu_m, "GPU Utilization std") or 0.0
    gpu_mem_mean = _find_measurement(gpu_m, "GPU Memory Used") or 0.0
    gpu_mem_std = _find_measurement(gpu_m, "GPU Memory Used std") or 0.0
    gpu_mem_peak = _find_measurement(gpu_m, "GPU Memory Used peak") or 0.0
    cpu_util_mean = _find_measurement(cpu_m, "CPU Utilization") or 0.0
    cpu_util_std = _find_measurement(cpu_m, "CPU Utilization std") or 0.0
    ram_mean = _find_measurement(mem_m, "System Memory RSS") or 0.0
    ram_std = _find_measurement(mem_m, "System Memory RSS std") or 0.0
    ram_peak = _find_measurement(mem_m, "System Memory RSS peak") or 0.0

    return Resources(
        gpu_util_pct=MeanStd(mean=gpu_util_mean, std=gpu_util_std),
        gpu_mem_gb=MeanStdPeak(mean=gpu_mem_mean, std=gpu_mem_std, peak=gpu_mem_peak),
        cpu_util_pct=MeanStd(mean=cpu_util_mean, std=cpu_util_std),
        ram_gb=MeanStdPeak(mean=ram_mean, std=ram_std, peak=ram_peak),
    )


def _build_training_bundle(
    reward_series: list[float],
    ep_len_series: list[float],
    iter_times_s: list[float],
    num_envs: int,
    steps_per_iter: int,
    args,
    versions,
    hardware,
    resources,
    run_start_dt: datetime,
    run_end_dt: datetime,
    status: str,
    app_launch_s: float,
    env_creation_s: float,
    first_step_s: float,
):
    """Build a schema-v1 :class:`TrainingBundle` for an SKRL run."""
    from isaaclab.benchmark.schema import (
        Learning,
        LearningCurve,
        MeanStd,
        RunIdentity,
        Runtime,
        StartupPhaseTimes,
        TrainingBundle,
    )

    def _ms(xs):
        return MeanStd(
            mean=float(np.mean(xs)) if xs else 0.0,
            std=float(np.std(xs)) if xs else 0.0,
        )

    env_steps_per_s_series = [num_envs * steps_per_iter / t if t > 0 else 0.0 for t in iter_times_s]
    iters_per_s_series = [1.0 / t if t > 0 else 0.0 for t in iter_times_s]

    backend = args.backend or "physx"
    run_id = args.run_id or synth_run_id("skrl", backend, args.task, args.seed)

    return TrainingBundle(
        run=RunIdentity(
            run_id=run_id,
            framework="skrl",
            backend=backend,
            task=args.task,
            seed=args.seed,
            num_envs=num_envs,
            max_iterations=args.max_iterations,
            start_time_utc=run_start_dt.isoformat().replace("+00:00", "Z"),
            end_time_utc=run_end_dt.isoformat().replace("+00:00", "Z"),
            duration_s=(run_end_dt - run_start_dt).total_seconds(),
            status=status,
        ),
        versions=versions,
        hardware=hardware,
        runtime=Runtime(
            startup_phase_times_s=StartupPhaseTimes(
                app_launch=app_launch_s,
                env_creation=env_creation_s,
                first_step=first_step_s,
            ),
            iterations_completed=len(iter_times_s),
            total_wall_time_s=sum(iter_times_s),
            steps_per_iteration=steps_per_iter,
            iteration_time_s=_ms(iter_times_s),
            env_steps_per_s=_ms(env_steps_per_s_series),
            iterations_per_s=_ms(iters_per_s_series),
        ),
        resources=resources,
        learning=Learning(
            ema_alpha=args.ema_alpha,
            reward=LearningCurve(
                final_raw=reward_series[-1] if reward_series else 0.0,
                final_ema=_compute_ema(reward_series, args.ema_alpha),
                series_per_iter=None if args.no_series else reward_series,
            ),
            ep_length=LearningCurve(
                final_raw=ep_len_series[-1] if ep_len_series else 0.0,
                final_ema=_compute_ema(ep_len_series, args.ema_alpha),
                series_per_iter=None if args.no_series else ep_len_series,
            ),
        ),
    )


def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
    app_start_time_begin: int,
    app_start_time_end: int,
):
    """Train an SKRL agent and emit a v1 schema bundle on success."""
    from skrl.utils.runner.torch import Runner

    # Override configuration with non-hydra CLI arguments.
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # Derive total timesteps from max_iterations (same formula as train.py).
    rollouts = int(agent_cfg["agent"]["rollouts"])
    agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * rollouts

    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    if args_cli.log_dir is not None:
        # Decompose so both `directory` and `experiment_name` are non-empty —
        # SKRL's BaseAgent synthesizes a timestamp+classname subdir when
        # `experiment_name` is falsy. Splitting <log_dir> into dirname/basename
        # makes ``os.path.join(directory, experiment_name)`` recompose to
        # <log_dir> exactly.
        log_dir = os.path.abspath(args_cli.log_dir)
        agent_cfg["agent"]["experiment"]["directory"] = os.path.dirname(log_dir) or "."
        agent_cfg["agent"]["experiment"]["experiment_name"] = os.path.basename(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{_algorithm}_{args_cli.ml_framework}"
        if agent_cfg["agent"]["experiment"]["experiment_name"]:
            log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
        agent_cfg["agent"]["experiment"]["directory"] = log_root_path
        agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
        log_dir = os.path.join(log_root_path, log_dir)
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.log_dir = log_dir

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    task_startup_time_begin = time.perf_counter_ns()
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    task_startup_time_end = time.perf_counter_ns()

    from scripts.benchmarks.skrl_benchmark_trainer import BenchmarkTrainer

    class _BenchmarkRunner(Runner):
        """Runner variant that builds a BenchmarkTrainer instead of a stock SequentialTrainer.

        Using a Runner subclass (rather than swapping ``Runner._trainer`` after
        construction) ensures SKRL's ``agent.init()`` — which creates a
        ``SummaryWriter`` — fires exactly once. Swapping after-the-fact would
        call ``agent.init()`` twice and leave an orphaned TB events file in
        the log dir.
        """

        def _generate_trainer(self, env, cfg, agent):
            # Mirror stock Runner._generate_trainer: pop 'class', pass cfg["trainer"].
            cfg["trainer"].pop("class", None)
            return BenchmarkTrainer(env=env, agents=agent, cfg=cfg["trainer"])

    runner = _BenchmarkRunner(env, agent_cfg)
    benchmark_trainer = runner._trainer

    with BenchmarkMonitor(benchmark, interval=1.0):
        runner.run()

    # Final recorder update after training completes.
    benchmark.update_manual_recorders()

    iter_times_s = benchmark_trainer.iter_times_s
    reward_series = benchmark_trainer.iter_rewards
    ep_len_series = benchmark_trainer.iter_ep_lengths
    per_iter_s = (sum(iter_times_s) / len(iter_times_s)) if iter_times_s else 0.0

    rl_training_times = {
        "Collection Time": iter_times_s,
        "Learning Time": [0.0] * len(iter_times_s),
        "Total FPS": [(args_cli.num_envs * rollouts / t) if t > 0 else 0.0 for t in iter_times_s],
    }

    log_app_start_time(benchmark, (app_start_time_end - app_start_time_begin) / 1e6)
    log_python_imports_time(benchmark, (imports_time_end - imports_time_begin) / 1e6)
    log_task_start_time(benchmark, (task_startup_time_end - task_startup_time_begin) / 1e6)
    log_scene_creation_time(benchmark, Timer.get_timer_info("scene_creation") * 1000)
    log_simulation_start_time(benchmark, Timer.get_timer_info("simulation_start") * 1000)
    log_total_start_time(benchmark, (task_startup_time_end - app_start_time_begin) / 1e6)
    if iter_times_s:
        log_runtime_step_times(benchmark, rl_training_times, compute_stats=True)
    if reward_series:
        log_rl_policy_rewards(benchmark, reward_series)
    if ep_len_series:
        log_rl_policy_episode_lengths(benchmark, ep_len_series)

    # Capture v1 state before _finalize_impl clears the recorders.
    versions_v1 = None
    hardware_v1 = None
    resources_v1 = None
    if args_cli.schema_v1_output is not None:
        versions_v1 = capture_versions(benchmark)
        hardware_v1 = capture_hardware(benchmark)
        resources_v1 = _capture_resources(benchmark)

    benchmark._finalize_impl()

    if args_cli.schema_v1_output is not None:
        from isaaclab.benchmark.schema import write_bundle_file

        bundle = _build_training_bundle(
            reward_series=reward_series,
            ep_len_series=ep_len_series,
            iter_times_s=iter_times_s,
            num_envs=env.unwrapped.num_envs,
            steps_per_iter=rollouts,
            args=args_cli,
            versions=versions_v1,
            hardware=hardware_v1,
            resources=resources_v1,
            run_start_dt=_SCRIPT_START_DT,
            run_end_dt=datetime.now(timezone.utc),
            status="completed",
            app_launch_s=(app_start_time_end - app_start_time_begin) / 1e9,
            env_creation_s=(task_startup_time_end - task_startup_time_begin) / 1e9,
            first_step_s=per_iter_s,
        )
        write_bundle_file(bundle, args_cli.schema_v1_output)

    env.close()


if __name__ == "__main__":
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, _agent_cfg_entry_point)

    app_start_time_begin = time.perf_counter_ns()
    with launch_simulation(env_cfg, args_cli):
        app_start_time_end = time.perf_counter_ns()
        main(env_cfg, agent_cfg, app_start_time_begin, app_start_time_end)
