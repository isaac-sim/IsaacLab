# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""

# Launching Isaac Sim Simulator first.

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import contextlib
import gymnasium as gym
import numpy as np
import os
import random
import torch

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.datasets import HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# global variable to keep track of the data generation statistics
num_success = 0
num_failures = 0
num_attempts = 0


async def run_data_generator(env, env_id, env_action_queue, data_generator, success_term, pause_subtask=False):
    """Run data generator."""
    global num_success, num_failures, num_attempts
    while True:
        results = await data_generator.generate(
            env_id=env_id,
            success_term=success_term,
            env_action_queue=env_action_queue,
            select_src_per_subtask=env.unwrapped.cfg.datagen_config.generation_select_src_per_subtask,
            transform_first_robot_pose=env.unwrapped.cfg.datagen_config.generation_transform_first_robot_pose,
            interpolate_from_last_target_pose=env.unwrapped.cfg.datagen_config.generation_interpolate_from_last_target_pose,
            pause_subtask=pause_subtask,
        )
        if bool(results["success"]):
            num_success += 1
        else:
            num_failures += 1
        num_attempts += 1


def env_loop(env, env_action_queue, shared_datagen_info_pool, asyncio_event_loop):
    """Main loop for the environment."""
    global num_success, num_failures, num_attempts
    prev_num_attempts = 0
    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:

            actions = torch.zeros(env.unwrapped.action_space.shape)

            # get actions from all the data generators
            for i in range(env.unwrapped.num_envs):
                # an async-blocking call to get an action from a data generator
                env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                actions[env_id] = action

            # perform action on environment
            env.step(actions)

            # mark done so the data generators can continue with the step results
            for i in range(env.unwrapped.num_envs):
                env_action_queue.task_done()

            if prev_num_attempts != num_attempts:
                prev_num_attempts = num_attempts
                print("")
                print("*" * 50)
                print(f"have {num_success} successes out of {num_attempts} trials so far")
                print(f"have {num_failures} failures out of {num_attempts} trials so far")
                print("*" * 50)

                # termination condition is on enough successes if @guarantee_success or enough attempts otherwise
                generation_guarantee = env.unwrapped.cfg.datagen_config.generation_guarantee
                generation_num_trials = env.unwrapped.cfg.datagen_config.generation_num_trials
                check_val = num_success if generation_guarantee else num_attempts
                if check_val >= generation_num_trials:
                    print(f"Reached {generation_num_trials} successes/attempts. Exiting.")
                    break

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

    env.close()


def main():
    num_envs = args_cli.num_envs

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get env name from the dataset file
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The dataset file {args_cli.input_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env_name = dataset_file_handler.get_env_name()

    if args_cli.task is not None:
        env_name = args_cli.task
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    # parse configuration
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs)

    # Override the datagen_config.generation_num_trials with the value from the command line arg
    if args_cli.generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = args_cli.generation_num_trials

    env_cfg.env_name = env_name

    # extract success checking function to invoke manually
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    # data generator is in charge of resetting the environment
    env_cfg.terminations = None

    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # create environment
    env = gym.make(env_name, cfg=env_cfg)

    # set seed for generation
    random.seed(env.unwrapped.cfg.datagen_config.seed)
    np.random.seed(env.unwrapped.cfg.datagen_config.seed)
    torch.manual_seed(env.unwrapped.cfg.datagen_config.seed)

    # reset before starting
    env.reset()

    # Set up asyncio stuff
    asyncio_event_loop = asyncio.get_event_loop()
    env_action_queue = asyncio.Queue()

    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = DataGenInfoPool(
        env.unwrapped, env.unwrapped.cfg, env.unwrapped.device, asyncio_lock=shared_datagen_info_pool_lock
    )
    shared_datagen_info_pool.load_from_dataset_file(args_cli.input_file)
    print(f"Loaded {shared_datagen_info_pool.num_datagen_infos} to datagen info pool")

    # make data generator object
    data_generator = DataGenerator(env=env.unwrapped, src_demo_datagen_info_pool=shared_datagen_info_pool)
    data_generator_asyncio_tasks = []
    for i in range(num_envs):
        data_generator_asyncio_tasks.append(
            asyncio_event_loop.create_task(
                run_data_generator(
                    env, i, env_action_queue, data_generator, success_term, pause_subtask=args_cli.pause_subtask
                )
            )
        )

    try:
        asyncio.ensure_future(asyncio.gather(*data_generator_asyncio_tasks))
    except asyncio.CancelledError:
        print("Tasks were cancelled.")

    env_loop(env, env_action_queue, shared_datagen_info_pool, asyncio_event_loop)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # close sim app
    simulation_app.close()
