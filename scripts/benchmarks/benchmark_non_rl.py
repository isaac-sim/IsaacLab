# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to benchmark non-RL environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_start_time_begin = time.perf_counter_ns()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

app_start_time_end = time.perf_counter_ns()

"""Rest everything follows."""

# enable benchmarking extension
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.benchmark.services")
from isaacsim.benchmark.services import BaseIsaacBenchmark

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from isaaclab.utils.timer import Timer
from scripts.benchmarks.utils import (
    log_app_start_time,
    log_python_imports_time,
    log_runtime_step_times,
    log_scene_creation_time,
    log_simulation_start_time,
    log_task_start_time,
    log_total_start_time,
)

imports_time_begin = time.perf_counter_ns()

import gymnasium as gym
import numpy as np
import os
import torch
from datetime import datetime

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

imports_time_end = time.perf_counter_ns()


# Create the benchmark
benchmark = BaseIsaacBenchmark(
    benchmark_name="benchmark_non_rl",
    workflow_metadata={
        "metadata": [
            {"name": "task", "data": args_cli.task},
            {"name": "seed", "data": args_cli.seed},
            {"name": "num_envs", "data": args_cli.num_envs},
            {"name": "num_frames", "data": args_cli.num_frames},
        ]
    },
    backend_type=args_cli.benchmark_backend,
)


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Benchmark without RL in the loop."""

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # process distributed
    world_size = 1
    world_rank = 0
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        world_size = int(os.getenv("WORLD_SIZE", 1))
        world_rank = app_launcher.global_rank

    task_startup_time_begin = time.perf_counter_ns()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        log_root_path = os.path.abs(f"benchmark/{args_cli.task}")
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
        log_scene_creation_time(benchmark, Timer.get_timer_info("scene_creation") * 1000)
        log_simulation_start_time(benchmark, Timer.get_timer_info("simulation_start") * 1000)
        log_total_start_time(benchmark, (task_startup_time_end - app_start_time_begin) / 1e6)
        log_runtime_step_times(benchmark, environment_step_times, compute_stats=True)

        benchmark.stop()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
