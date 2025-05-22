# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import isaaclab
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_steps", type=int, default=5000, help="Number of steps to simulate.")
parser.add_argument("--check_every", type=int, default=100, help="Interval of steps to check time.")
parser.add_argument("--results_dir", type=str, default="results", help="Directory to save the results.")
parser.add_argument("--save_results", action="store_true", default=False, help="Save the results.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import time
import torch
import os
from datetime import datetime
import json

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    step_interval = args_cli.check_every
    num_steps = args_cli.num_steps

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    start_time = time.time()
    start_time_batch = time.time()
    print(f"[INFO]: Starting Benchmark for {num_steps} steps.")
    time_per_step_batch = []
    for i in range(1, num_steps+1, 1):
        # apply random actions
        if i % step_interval == 0:
            check_time = time.time()
            time_per_step_batch.append(check_time - start_time_batch)
            start_time_batch = time.time()
            print(f"[INFO]: Time per step batch: {time_per_step_batch[-1]} seconds")

        with torch.inference_mode():
            actions = torch.rand(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    end_time = time.time()
    print("--------------------------------")
    print(f"[INFO]: Running task: {args_cli.task}")
    print(f"[INFO]: Number of environments: {args_cli.num_envs}")
    print("--------------------------------")
    print(f"[INFO]: Total time taken: {end_time - start_time} seconds")
    print(f"[INFO]: Time per step: {(end_time - start_time) / args_cli.num_steps} seconds")
    print(f"[INFO]: Average time per step batch: {np.mean(time_per_step_batch)} seconds")
    print(f"[INFO]: Std dev of time per step batch: {np.std(time_per_step_batch)} seconds")
    print("--------------------------------")
    total_num_steps = num_steps * args_cli.num_envs
    print(f"[INFO]: Total number of steps: {total_num_steps}")
    print(f"[INFO]: Time per step x num_envs: {(end_time - start_time) / total_num_steps} seconds")
    print("--------------------------------")
    # Make a dict with the results
    if args_cli.save_results:
        results = {
            "isaaclab_version": isaaclab.__version__,
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "step_interval": step_interval,
            "num_steps": num_steps,
            "total_time": end_time - start_time,
            "avg_time_per_step_batch": np.mean(time_per_step_batch),
            "std_time_per_step_batch": np.std(time_per_step_batch),
            "total_num_steps": total_num_steps,
            "time_per_step_num_envs": (end_time - start_time) / total_num_steps,
            "device": args_cli.device,
        }
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{args_cli.task}_envs_{args_cli.num_envs}_steps_{num_steps}_interval_{step_interval}_{date_time}.json"
        if not os.path.exists(args_cli.results_dir):
            os.makedirs(args_cli.results_dir)
        # Save the results to a json file
        with open(os.path.join(args_cli.results_dir, name), "w") as f:
            json.dump(results, f)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
