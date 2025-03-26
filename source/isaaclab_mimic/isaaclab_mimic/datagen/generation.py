# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import torch
from typing import Any

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# global variable to keep track of the data generation statistics
num_success = 0
num_failures = 0
num_attempts = 0


async def run_data_generator(
    env: ManagerBasedEnv,
    env_id: int,
    env_action_queue: asyncio.Queue,
    data_generator: DataGenerator,
    success_term: Any,
    pause_subtask: bool = False,
):
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


def env_loop(
    env: ManagerBasedEnv,
    env_action_queue: asyncio.Queue,
    shared_datagen_info_pool: DataGenInfoPool,
    asyncio_event_loop: asyncio.AbstractEventLoop,
) -> None:
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


def setup_env_config(
    env_name: str,
    output_dir: str,
    output_file_name: str,
    num_envs: int,
    device: str,
    generation_num_trials: int | None = None,
) -> tuple[Any, Any]:
    """Configure the environment for data generation.

    Args:
        env_name: Name of the environment
        output_dir: Directory to save output
        output_file_name: Name of output file
        num_envs: Number of environments to run
        device: Device to run on
        generation_num_trials: Optional override for number of trials

    Returns:
        tuple containing:
            - env_cfg: The environment configuration
            - success_term: The success termination condition

    Raises:
        NotImplementedError: If no success termination term found
    """
    env_cfg = parse_env_cfg(env_name, device=device, num_envs=num_envs)

    if generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = generation_num_trials

    env_cfg.env_name = env_name

    # Extract success checking function
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    # Configure for data generation
    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False

    # Setup recorders
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def setup_async_generation(
    env: Any, num_envs: int, input_file: str, success_term: Any, pause_subtask: bool = False
) -> dict[str, Any]:
    """Setup async data generation tasks.

    Args:
        env: The environment instance
        num_envs: Number of environments to run
        input_file: Path to input dataset file
        success_term: Success termination condition
        pause_subtask: Whether to pause after subtasks

    Returns:
        List of asyncio tasks for data generation
    """
    asyncio_event_loop = asyncio.get_event_loop()
    env_action_queue = asyncio.Queue()
    shared_datagen_info_pool_lock = asyncio.Lock()
    shared_datagen_info_pool = DataGenInfoPool(
        env.unwrapped, env.unwrapped.cfg, env.unwrapped.device, asyncio_lock=shared_datagen_info_pool_lock
    )
    shared_datagen_info_pool.load_from_dataset_file(input_file)
    print(f"Loaded {shared_datagen_info_pool.num_datagen_infos} to datagen info pool")

    # Create and schedule data generator tasks
    data_generator = DataGenerator(env=env.unwrapped, src_demo_datagen_info_pool=shared_datagen_info_pool)
    data_generator_asyncio_tasks = []
    for i in range(num_envs):
        task = asyncio_event_loop.create_task(
            run_data_generator(env, i, env_action_queue, data_generator, success_term, pause_subtask=pause_subtask)
        )
        data_generator_asyncio_tasks.append(task)

    return {
        "tasks": data_generator_asyncio_tasks,
        "event_loop": asyncio_event_loop,
        "action_queue": env_action_queue,
        "info_pool": shared_datagen_info_pool,
    }
