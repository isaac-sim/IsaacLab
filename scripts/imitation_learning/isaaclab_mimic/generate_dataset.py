# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation, scan
from isaaclab.utils.string import list_intersection, string_to_callable

parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--max_num_failures", type=int, default=None, help="Stop after this many failed generation attempts."
)
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
    help="Pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--use_skillgen",
    action="store_true",
    default=False,
    help="Use skillgen to generate motion trajectories",
)
parser.add_argument(
    "--disable_dataset_compression",
    action="store_true",
    default=False,
    help="Disables dataset compression",
)
parser.add_argument(
    "--external_callback",
    default=None,
    help="Fully qualified path to an externally defined callback.",
)

add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.max_num_failures is not None and args_cli.max_num_failures < 1:
    parser.error("--max_num_failures must be positive")

# Mimic environments may use camera observations or an RTX renderer. Request
# rendering support here so callers do not need a legacy CLI flag.
args_cli.enable_cameras = True

# Call an external callback if requested.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# Error on unrecognized arguments.
unrecognized_args = list_intersection(remaining_args, remaining_args_env_registration)
if unrecognized_args:
    parser.error(f"unrecognized arguments: {' '.join(unrecognized_args)}")

import asyncio
import inspect
import logging
import os
import random

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.datasets import HDF5DatasetFileHandler

import isaaclab_mimic.envs  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# import logger
logger = logging.getLogger(__name__)


def main():
    num_envs = args_cli.num_envs

    # Get env name
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    if not task_name:
        if not os.path.exists(args_cli.input_file):
            raise FileNotFoundError(f"The dataset file {args_cli.input_file} does not exist.")
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(args_cli.input_file)
        task_name = dataset_file_handler.get_env_name()
        if task_name is None:
            raise ValueError("Environment name not found in dataset")
    env_name = task_name

    # Launch the runtime the task needs before loading the mimic helpers, which import USD
    with launch_simulation(parse_env_cfg(env_name, device=args_cli.device, num_envs=num_envs), args_cli):
        generate(env_name)


def generate(env_name: str):
    """Configure and create the environment, then run the (optionally SkillGen-based) data generation."""
    from isaaclab.envs import ManagerBasedRLMimicEnv

    from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
    from isaaclab_mimic.datagen.utils import setup_output_paths

    num_envs = args_cli.num_envs

    # Setup output paths
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)

    # Configure environment
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
        dataset_compression=not args_cli.disable_dataset_compression,
    )
    # resolve the automatic physics and renderer selections of this config the same way the launch did
    scan(env_cfg, args_cli)
    if args_cli.max_num_failures is not None:
        env_cfg.datagen_config.max_num_failures = args_cli.max_num_failures

    # Create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # Check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        logger.warning(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # Set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # Reset before starting
    env.reset()

    motion_planners = None
    try:
        if args_cli.use_skillgen:
            from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
            from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

            # Create one motion planner per environment
            motion_planners = {}
            for env_id in range(num_envs):
                print(f"Initializing motion planner for environment {env_id}")
                # Create a config instance from the task name
                planner_config = CuroboPlannerCfg.from_task_name(env_name)

                # Ensure visualization is only enabled for the first environment
                # If not, sphere and plan visualization will be too slow in isaac lab
                # It is efficient to visualize the spheres and plan for the first environment in rerun
                if env_id != 0:
                    planner_config.visualize_spheres = False
                    planner_config.visualize_plan = False

                motion_planners[env_id] = CuroboPlanner(
                    env=env,
                    robot=env.scene["robot"],
                    config=planner_config,  # Pass the config object
                    env_id=env_id,  # Pass environment ID
                )

            env.cfg.datagen_config.use_skillgen = True

        # Setup and run async data generation
        async_components = setup_async_generation(
            env=env,
            num_envs=args_cli.num_envs,
            input_file=args_cli.input_file,
            success_term=success_term,
            pause_subtask=args_cli.pause_subtask,
            motion_planners=motion_planners,  # Pass the motion planners dictionary
        )

        try:
            data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
            env_loop(
                env,
                async_components["reset_queue"],
                async_components["action_queue"],
                async_components["info_pool"],
                async_components["event_loop"],
                data_gen_tasks=data_gen_tasks,
            )
        except asyncio.CancelledError:
            print("Tasks were cancelled.")
        finally:
            # Cancel all async tasks when env_loop finishes
            data_gen_tasks.cancel()
            try:
                # Wait for tasks to be cancelled
                async_components["event_loop"].run_until_complete(data_gen_tasks)
            except asyncio.CancelledError:
                print("Remaining async tasks cancelled and cleaned up.")
            except Exception as e:
                print(f"Error cancelling remaining async tasks: {e}")
            # Cleanup of motion planners and their visualizers
            if motion_planners is not None:
                for env_id, planner in motion_planners.items():
                    if getattr(planner, "plan_visualizer", None) is not None:
                        print(f"Closing plan visualizer for environment {env_id}")
                        planner.plan_visualizer.close()
                        planner.plan_visualizer = None
                motion_planners.clear()
    finally:
        # Close env after async tasks are done so success_term is never called on a closed env
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
