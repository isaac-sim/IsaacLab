# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to benchmark non-RL environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark non-RL environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--num_frames", type=int, default=100, help="Number of environment frames to run benchmark for.")
parser.add_argument(
    "--benchmark_backend",
    type=str,
    default="OmniPerfKPIFile",
    choices=["LocalLogMetrics", "JSONFileMetrics", "OsmoKPIFile", "OmniPerfKPIFile"],
    help="Benchmarking backend options, defaults OmniPerfKPIFile",
)
parser.add_argument("--output_folder", type=str, default=None, help="Output folder for the benchmark.")
parser.add_argument(
    "--kit",
    action="store_true",
    default=False,
    help="Enable Isaac Sim Kit and use isaacsim.benchmark.services. Default: False (uses standalone benchmark).",
)

# Conditionally add AppLauncher args only if --kit is enabled
if "--kit" in sys.argv:
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_start_time_begin = time.perf_counter_ns()

# Conditionally launch Isaac Sim
simulation_app = None
app_launcher = None
if args_cli.kit:
    # Force Omniverse mode by setting environment variable
    # This ensures SimulationApp is launched even without explicit visualizers
    os.environ["LAUNCH_OV_APP"] = "1"

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

app_start_time_end = time.perf_counter_ns()

"""Rest everything follows."""

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

# Import benchmark infrastructure based on kit flag
if args_cli.kit:
    # enable benchmarking extension
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.benchmark.services")

    # Set the benchmark settings according to the inputs
    import carb

    settings = carb.settings.get_settings()
    settings.set("/exts/isaacsim.benchmark.services/metrics/metrics_output_folder", args_cli.output_folder)
    settings.set("/exts/isaacsim.benchmark.services/metrics/randomize_filename_prefix", True)

    from isaacsim.benchmark.services import BaseIsaacBenchmark

    from scripts.benchmarks.utils.benchmark_utils import create_kit_logging_functions, get_timer_value

    # Get all logging functions for kit mode
    log_funcs = create_kit_logging_functions()
else:
    # Use standalone benchmark services
    from scripts.benchmarks.utils.benchmark_utils import create_standalone_logging_functions, get_timer_value
    from scripts.benchmarks.utils.standalone_benchmark import StandaloneBenchmark

    # Get all logging functions for standalone mode
    log_funcs = create_standalone_logging_functions()

# Extract individual functions from the dictionary for easier use
get_isaaclab_version = log_funcs["get_isaaclab_version"]
get_mujoco_warp_version = log_funcs["get_mujoco_warp_version"]
get_newton_version = log_funcs["get_newton_version"]
log_app_start_time = log_funcs["log_app_start_time"]
log_python_imports_time = log_funcs["log_python_imports_time"]
log_task_start_time = log_funcs["log_task_start_time"]
log_scene_creation_time = log_funcs["log_scene_creation_time"]
log_simulation_start_time = log_funcs["log_simulation_start_time"]
log_total_start_time = log_funcs["log_total_start_time"]
log_runtime_step_times = log_funcs["log_runtime_step_times"]

imports_time_begin = time.perf_counter_ns()

import gymnasium as gym
import numpy as np
import torch
from datetime import datetime

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

imports_time_end = time.perf_counter_ns()


# Create the benchmark
if args_cli.kit:
    benchmark = BaseIsaacBenchmark(
        benchmark_name="benchmark_non_rl",
        workflow_metadata={
            "metadata": [
                {"name": "task", "data": args_cli.task},
                {"name": "seed", "data": args_cli.seed},
                {"name": "num_envs", "data": args_cli.num_envs},
                {"name": "num_frames", "data": args_cli.num_frames},
                {"name": "Mujoco Warp Info", "data": get_mujoco_warp_version()},
                {"name": "Isaac Lab Info", "data": get_isaaclab_version()},
                {"name": "Newton Info", "data": get_newton_version()},
            ]
        },
        backend_type=args_cli.benchmark_backend,
    )
else:
    benchmark = StandaloneBenchmark(
        benchmark_name="benchmark_non_rl",
        workflow_metadata={
            "metadata": [
                {"name": "task", "data": args_cli.task},
                {"name": "seed", "data": args_cli.seed},
                {"name": "num_envs", "data": args_cli.num_envs},
                {"name": "num_frames", "data": args_cli.num_frames},
                {"name": "Mujoco Warp Info", "data": get_mujoco_warp_version()},
                {"name": "Isaac Lab Info", "data": get_isaaclab_version()},
                {"name": "Newton Info", "data": get_newton_version()},
            ]
        },
        backend_type=args_cli.benchmark_backend,
        output_folder=args_cli.output_folder,
        randomize_filename_prefix=True,
    )


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Benchmark without RL in the loop."""

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.kit and hasattr(args_cli, "device") and args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # process distributed
    world_size = 1
    world_rank = 0
    if args_cli.distributed:
        if args_cli.kit:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            world_size = int(os.getenv("WORLD_SIZE", 1))
            world_rank = app_launcher.global_rank
        else:
            print("[WARNING] Distributed mode is only supported with --kit flag.")

    task_startup_time_begin = time.perf_counter_ns()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        log_root_path = os.path.abspath(f"benchmark/{args_cli.task}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    task_startup_time_end = time.perf_counter_ns()

    env.reset()

    benchmark.set_phase("sim_runtime")

    # counter for number of frames to run for
    num_frames = 0
    # log frame times
    step_times = []

    # Run loop depends on whether we're using kit or not
    if args_cli.kit:
        while simulation_app.is_running():
            while num_frames < args_cli.num_frames:
                # get upper and lower bounds of action space, sample actions randomly on this interval
                action_high = 1
                action_low = -1
                actions = (action_high - action_low) * torch.rand(
                    env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device
                ) - action_high

                # env stepping
                env_step_time_begin = time.perf_counter_ns()
                _ = env.step(actions)
                end_step_time_end = time.perf_counter_ns()
                step_times.append(end_step_time_end - env_step_time_begin)

                num_frames += 1

            # terminate
            break
    else:
        # Standalone mode - simple loop
        while num_frames < args_cli.num_frames:
            # get upper and lower bounds of action space, sample actions randomly on this interval
            action_high = 1
            action_low = -1
            actions = (action_high - action_low) * torch.rand(
                env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device
            ) - action_high

            # env stepping
            env_step_time_begin = time.perf_counter_ns()
            _ = env.step(actions)
            end_step_time_end = time.perf_counter_ns()
            step_times.append(end_step_time_end - env_step_time_begin)

            num_frames += 1

    if world_rank == 0:
        benchmark.store_measurements()

        # compute stats
        step_times = np.array(step_times) / 1e6  # ns to ms
        fps = 1.0 / (step_times / 1000)
        effective_fps = fps * env.unwrapped.num_envs * world_size

        # prepare step timing dict
        environment_step_times = {
            "Environment step times": step_times.tolist(),
            "Environment step FPS": fps.tolist(),
            "Environment step effective FPS": effective_fps.tolist(),
        }

        log_app_start_time(benchmark, (app_start_time_end - app_start_time_begin) / 1e6)
        log_python_imports_time(benchmark, (imports_time_end - imports_time_begin) / 1e6)
        log_task_start_time(benchmark, (task_startup_time_end - task_startup_time_begin) / 1e6)

        # Timer may not be available in standalone mode
        scene_creation_time = get_timer_value("scene_creation")
        simulation_start_time = get_timer_value("simulation_start")

        log_scene_creation_time(benchmark, scene_creation_time * 1000 if scene_creation_time else None)
        log_simulation_start_time(benchmark, simulation_start_time * 1000 if simulation_start_time else None)
        log_total_start_time(benchmark, (task_startup_time_end - app_start_time_begin) / 1e6)
        log_runtime_step_times(benchmark, environment_step_times, compute_stats=True)

        benchmark.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    if simulation_app is not None:
        simulation_app.close()
