# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to benchmark RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

from isaaclab_tasks.utils import fold_preset_tokens, setup_preset_cli

from scripts.benchmarks.early_stop import (
    RslRlEarlyStopWrapper,
    add_success_cli_args,
    build_success_kwargs,
    get_success_tracker,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip


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


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
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
    "--reward_threshold", type=float, default=None, help="Reward threshold for convergence (overrides config)."
)
parser.add_argument(
    "--check_convergence", action="store_true", help="Check reward convergence using thresholds from configs.yaml."
)
parser.add_argument(
    "--convergence_config", type=str, default="full", help="Config mode for convergence thresholds (default: full)."
)
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
add_success_cli_args(parser)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
hydra_args = fold_preset_tokens(hydra_args)
if args_cli.video:
    args_cli.enable_cameras = True

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

# Re-set sys.argv so the --backend coercion above propagates to Hydra.
sys.argv = [sys.argv[0]] + hydra_args

imports_time_begin = time.perf_counter_ns()

import contextlib
import importlib.metadata as metadata
from datetime import datetime, timezone

_SCRIPT_START_DT = datetime.now(timezone.utc)

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, launch_simulation, resolve_task_config

imports_time_end = time.perf_counter_ns()

from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor
from isaaclab.utils.timer import Timer

from scripts.benchmarks._schema_helpers import capture_hardware, capture_versions, synth_run_id
from scripts.benchmarks.utils import (
    get_backend_type,
    get_preset_string,
    log_app_start_time,
    log_python_imports_time,
    log_rl_training_metrics,
    log_runtime_step_times,
    log_scene_creation_time,
    log_simulation_start_time,
    log_success,
    log_task_start_time,
    log_total_start_time,
    parse_tf_logs,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Create the benchmark
backend_type = get_backend_type(args_cli.benchmark_backend)
benchmark = BaseIsaacLabBenchmark(
    benchmark_name="benchmark_rsl_rl_train",
    backend_type=backend_type,
    output_path=args_cli.output_path,
    use_recorders=True,
    frametime_recorders=backend_type in ("summary", "omniperf"),
    output_prefix=f"benchmark_rsl_rl_train_{args_cli.task}",
    workflow_metadata={
        "metadata": [
            {"name": "task", "data": args_cli.task},
            {"name": "seed", "data": args_cli.seed},
            {"name": "num_envs", "data": args_cli.num_envs},
            {"name": "max_iterations", "data": args_cli.max_iterations},
            {"name": "presets", "data": get_preset_string(hydra_args)},
        ]
    },
)


def _compute_ema(series: list[float], alpha: float) -> float:
    """Exponentially weighted moving average over a per-iteration series.

    Returns the final EMA value: ``x_0`` initialised to ``series[0]`` and updated
    as ``x_t = alpha * y_t + (1 - alpha) * x_{t-1}``. Empty series returns 0.0.

    Args:
        series: Per-iteration scalar values (reward or episode length).
        alpha: Smoothing factor in [0, 1]. Smaller values give more smoothing.

    Returns:
        Final EMA value after walking the full series.
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
    """Build a schema-v1 :class:`Resources` dataclass from GPU/CPU/Memory recorders."""
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
    log_data,
    agent_cfg,
    env,
    args,
    framework: str,
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
    """Build a schema-v1 :class:`TrainingBundle` from tensorboard-parsed training data."""
    import numpy as np

    from isaaclab.benchmark.schema import (
        Learning,
        LearningCurve,
        MeanStd,
        RunIdentity,
        Runtime,
        StartupPhaseTimes,
        TrainingBundle,
    )

    reward_series = [float(x) for x in log_data.get("Train/mean_reward", [])]
    ep_len_series = [float(x) for x in log_data.get("Train/mean_episode_length", [])]

    num_envs = env.unwrapped.num_envs
    steps_per_iter = agent_cfg.num_steps_per_env
    total_fps = list(log_data.get("Perf/total_fps", []) or [])
    iter_times = [num_envs * steps_per_iter / fps if fps > 0 else 0.0 for fps in total_fps]

    def _ms(xs):
        return MeanStd(
            mean=float(np.mean(xs)) if xs else 0.0,
            std=float(np.std(xs)) if xs else 0.0,
        )

    env_steps_per_s_series = [num_envs * steps_per_iter / t if t > 0 else 0.0 for t in iter_times]
    iters_per_s_series = [1.0 / t if t > 0 else 0.0 for t in iter_times]

    backend = args.backend or "physx"
    run_id = args.run_id or synth_run_id(framework, backend, args.task, args.seed)

    return TrainingBundle(
        run=RunIdentity(
            run_id=run_id,
            framework=framework,
            backend=backend,
            task=args.task,
            seed=args.seed,
            num_envs=num_envs,
            max_iterations=agent_cfg.max_iterations,
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
            iterations_completed=len(iter_times),
            total_wall_time_s=sum(iter_times),
            steps_per_iteration=steps_per_iter,
            iteration_time_s=_ms(iter_times),
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
    agent_cfg: RslRlOnPolicyRunnerCfg,
    app_start_time_begin: int,
    app_start_time_end: int,
):
    """Train with RSL-RL agent."""
    # parse configuration
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    # For distributed training, launch_simulation() already resolved the
    # correct per-rank device; only apply a CLI --device override for
    # non-distributed runs (the default "cuda:0" would clobber the
    # per-rank device otherwise).
    if not args_cli.distributed:
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    # env_cfg.sim.device is already resolved by launch_simulation().
    world_rank = 0
    world_size = 1
    if args_cli.distributed:
        agent_cfg.device = env_cfg.sim.device

        # use global rank for seed diversity across all nodes
        world_rank = int(os.getenv("RANK", "0"))
        seed = agent_cfg.seed + world_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed
        world_size = int(os.getenv("WORLD_SIZE", 1))

    if args_cli.log_dir is not None:
        # Explicit override: write straight into the given dir.
        log_dir = os.path.abspath(args_cli.log_dir)
        log_root_path = os.path.dirname(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[INFO] Logging experiment in directory: {log_dir}")
    else:
        # Default: auto-generate logs/<framework>/<experiment>/<timestamp>/
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    task_startup_time_begin = time.perf_counter_ns()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    task_startup_time_end = time.perf_counter_ns()

    # handle deprecated configurations (e.g. legacy policy -> actor/critic migration)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # always track the success metric; early-stop only if --check_success
    early_stop_ctx = RslRlEarlyStopWrapper(
        env, runner, num_steps_per_env=agent_cfg.num_steps_per_env, **build_success_kwargs(args_cli)
    )

    # run training with continuous benchmark monitoring
    with early_stop_ctx, BenchmarkMonitor(benchmark, interval=1.0):
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    if world_rank == 0:
        # Final update after training completes
        benchmark.update_manual_recorders()

        # parse tensorboard file stats
        log_data = parse_tf_logs(log_dir)

        # prepare RL timing dict
        collection_fps = (
            1
            / (np.array(log_data["Perf/collection_time"]))
            * env.unwrapped.num_envs
            * agent_cfg.num_steps_per_env
            * world_size
        )
        rl_training_times = {
            "Collection Time": (np.array(log_data["Perf/collection_time"]) / 1000).tolist(),
            "Learning Time": (np.array(log_data["Perf/learning_time"]) / 1000).tolist(),
            "Collection FPS": collection_fps.tolist(),
            "Total FPS": log_data["Perf/total_fps"] * world_size,
        }

        # log additional metrics to benchmark services
        log_app_start_time(benchmark, (app_start_time_end - app_start_time_begin) / 1e6)
        log_python_imports_time(benchmark, (imports_time_end - imports_time_begin) / 1e6)
        log_task_start_time(benchmark, (task_startup_time_end - task_startup_time_begin) / 1e6)
        log_scene_creation_time(benchmark, Timer.get_timer_info("scene_creation") * 1000)
        log_simulation_start_time(benchmark, Timer.get_timer_info("simulation_start") * 1000)
        log_total_start_time(benchmark, (task_startup_time_end - app_start_time_begin) / 1e6)
        log_runtime_step_times(benchmark, rl_training_times, compute_stats=True)
        log_rl_training_metrics(
            benchmark,
            log_data,
            reward_tag="Train/mean_reward",
            episode_length_tag="Train/mean_episode_length",
            task=args_cli.task,
            workflow="rsl_rl",
            should_check_convergence=args_cli.check_convergence,
            reward_threshold=args_cli.reward_threshold,
            convergence_config=args_cli.convergence_config,
        )

        tracker = get_success_tracker(args_cli, early_stop_ctx.tracker, log_data)
        log_success(benchmark, tracker, framework_iteration_count=early_stop_ctx.framework_iteration_count)

        # Capture v1 state before _finalize_impl nulls out _manual_recorders.
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

            # Proxy for first-step time: the first iteration's collection+learning time.
            # Pending a dedicated first-step timer in runner.learn().
            first_step_s = 0.0
            with contextlib.suppress(IndexError, KeyError, ValueError):
                first_step_s = float(rl_training_times["Collection Time"][0]) + float(
                    rl_training_times["Learning Time"][0]
                )

            bundle = _build_training_bundle(
                log_data=log_data,
                agent_cfg=agent_cfg,
                env=env,
                args=args_cli,
                framework="rsl_rl",
                versions=versions_v1,
                hardware=hardware_v1,
                resources=resources_v1,
                run_start_dt=_SCRIPT_START_DT,
                run_end_dt=datetime.now(timezone.utc),
                status="completed",
                app_launch_s=(app_start_time_end - app_start_time_begin) / 1e9,
                env_creation_s=(task_startup_time_end - task_startup_time_begin) / 1e9,
                first_step_s=first_step_s,
            )
            write_bundle_file(bundle, args_cli.schema_v1_output)

    # close the simulator
    env.close()


if __name__ == "__main__":
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    app_start_time_begin = time.perf_counter_ns()
    with launch_simulation(env_cfg, args_cli):
        app_start_time_end = time.perf_counter_ns()
        main(env_cfg, agent_cfg, app_start_time_begin, app_start_time_end)
