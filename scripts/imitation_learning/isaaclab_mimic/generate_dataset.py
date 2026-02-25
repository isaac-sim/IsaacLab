# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""

"""Launch Isaac Sim Simulator first."""

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
parser.add_argument(
    "--use_skillgen",
    action="store_true",
    default=False,
    help="use skillgen to generate motion trajectories",
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
import inspect
import logging
import random

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


def main():
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )

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
    # Close sim app
    simulation_app.close()
