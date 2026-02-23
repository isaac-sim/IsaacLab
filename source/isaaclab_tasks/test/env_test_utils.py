# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared test utilities for Isaac Lab environments."""

import inspect
import os

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.envs.utils.spaces import sample_space
from isaaclab.sim import SimulationContext
from isaaclab.utils.version import get_isaac_sim_version

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def setup_environment(
    include_play: bool = False,
    factory_envs: bool | None = None,
    multi_agent: bool | None = None,
) -> list[str]:
    """
    Acquire all registered Isaac environment task IDs with optional filters.

    Args:
        include_play: If True, include environments ending in 'Play-v0'.
        factory_envs:
            - True: include only Factory environments
            - False: exclude Factory environments
            - None: include both Factory and non-Factory environments
        multi_agent:
            - True: include only multi-agent environments
            - False: include only single-agent environments
            - None: include all environments regardless of agent type

    Returns:
        A sorted list of task IDs matching the selected filters.
    """
    # disable interactive mode for wandb for automate environments
    os.environ["WANDB_DISABLED"] = "true"

    # acquire all Isaac environment names
    registered_tasks = []
    for task_spec in gym.registry.values():
        # only consider Isaac environments
        if "Isaac" not in task_spec.id:
            continue

        # filter Play environments, if needed
        if not include_play and task_spec.id.endswith("Play-v0"):
            continue

        # TODO: factory environments cause tests to fail if run together with other envs,
        # so we collect these environments separately to run in a separate unit test.
        # apply factory filter
        if (factory_envs is True and ("Factory" not in task_spec.id and "Forge" not in task_spec.id)) or (
            factory_envs is False and ("Factory" in task_spec.id or "Forge" in task_spec.id)
        ):
            continue
        # if None: no filter

        # apply multi agent filter
        if multi_agent is not None:
            # parse config
            env_cfg = parse_env_cfg(task_spec.id)
            if (multi_agent is True and not hasattr(env_cfg, "possible_agents")) or (
                multi_agent is False and hasattr(env_cfg, "possible_agents")
            ):
                continue
        # if None: no filter

        registered_tasks.append(task_spec.id)

    # sort environments alphabetically
    registered_tasks.sort()

    # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running many environments
    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    print(">>> All registered environments:", registered_tasks)

    return registered_tasks


def _run_environments(
    task_name,
    device,
    num_envs,
    num_steps=100,
    multi_agent=False,
    create_stage_in_memory=False,
    disable_clone_in_fabric=False,
):
    """Run all environments and check environments return valid signals.

    Args:
        task_name: Name of the environment.
        device: Device to use (e.g., 'cuda').
        num_envs: Number of environments.
        num_steps: Number of simulation steps.
        multi_agent: Whether the environment is multi-agent.
        create_stage_in_memory: Whether to create stage in memory.
        disable_clone_in_fabric: Whether to disable fabric cloning.
    """

    # skip test if stage in memory is not supported
    if get_isaac_sim_version().major < 5 and create_stage_in_memory:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # skip suction gripper environments as they require CPU simulation and cannot be run with GPU simulation
    if "Suction" in task_name and device != "cpu":
        return

    # skip these environments as they cannot be run with 32 environments within reasonable VRAM
    if num_envs == 32 and task_name in [
        "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
        "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0",
        "Isaac-Stack-Cube-Instance-Randomize-Franka-v0",
    ]:
        return

    # these environments are using SingleArticulation class, which need to be updated
    if "RmpFlow" in task_name or "Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor" in task_name:
        return

    # skip these environments as they cannot be run with 32 environments within reasonable VRAM
    if "Visuomotor" in task_name and num_envs == 32:
        return

    # skip automate environments as they require cuda installation
    if task_name in ["Isaac-AutoMate-Assembly-Direct-v0", "Isaac-AutoMate-Disassembly-Direct-v0"]:
        return

    # Check if this is the teddy bear environment and if it's being called from the right test file
    if task_name == "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0":
        # Get the calling frame to check which test file is calling this function
        frame = inspect.currentframe()
        while frame:
            filename = frame.f_code.co_filename
            if "test_lift_teddy_bear.py" in filename:
                # Called from the dedicated test file, allow it to run
                break
            frame = frame.f_back

        # If not called from the dedicated test file, skip it
        if not frame:
            return

    print(f""">>> Running test for environment: {task_name}""")
    _check_random_actions(
        task_name,
        device,
        num_envs,
        num_steps=num_steps,
        multi_agent=multi_agent,
        create_stage_in_memory=create_stage_in_memory,
        disable_clone_in_fabric=disable_clone_in_fabric,
    )
    print(f""">>> Closing environment: {task_name}""")
    print("-" * 80)


def _check_random_actions(
    task_name: str,
    device: str,
    num_envs: int,
    num_steps: int = 100,
    multi_agent: bool = False,
    create_stage_in_memory: bool = False,
    disable_clone_in_fabric: bool = False,
):
    """Run random actions and check environments return valid signals.

    Args:
        task_name: Name of the environment.
        device: Device to use (e.g., 'cuda').
        num_envs: Number of environments.
        num_steps: Number of simulation steps.
        multi_agent: Whether the environment is multi-agent.
        create_stage_in_memory: Whether to create stage in memory.
        disable_clone_in_fabric: Whether to disable fabric cloning.
    """
    # create a new context stage, if stage in memory is not enabled
    if not create_stage_in_memory:
        sim_utils.create_new_stage()

    # reset the rtx sensors setting to False
    get_settings_manager().set_bool("/isaaclab/render/rtx_sensors", False)
    env = None
    try:
        # parse config
        env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        # set config args
        env_cfg.sim.create_stage_in_memory = create_stage_in_memory
        if disable_clone_in_fabric:
            env_cfg.scene.clone_in_fabric = False

        # filter based off multi agents mode and create env
        if multi_agent:
            if not hasattr(env_cfg, "possible_agents"):
                print(f"[INFO]: Skipping {task_name} as it is not a multi-agent task")
                return
            env = gym.make(task_name, cfg=env_cfg)
        else:
            if hasattr(env_cfg, "possible_agents"):
                print(f"[INFO]: Skipping {task_name} as it is a multi-agent task")
                return
            env = gym.make(task_name, cfg=env_cfg)

        # disable control on stop
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

        # override action space if set to inf for `Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0`
        if task_name == "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0":
            for i in range(env.unwrapped.single_action_space.shape[0]):
                if env.unwrapped.single_action_space.low[i] == float("-inf"):
                    env.unwrapped.single_action_space.low[i] = -1.0
                if env.unwrapped.single_action_space.high[i] == float("inf"):
                    env.unwrapped.single_action_space.low[i] = 1.0

        # reset environment
        obs, _ = env.reset()

        # check signal
        assert _check_valid_tensor(obs)

        # simulate environment for num_steps
        with torch.inference_mode():
            for _ in range(num_steps):
                # sample actions according to the defined space
                if multi_agent:
                    actions = {
                        agent: sample_space(
                            env.unwrapped.action_spaces[agent], device=env.unwrapped.device, batch_size=num_envs
                        )
                        for agent in env.unwrapped.possible_agents
                    }
                else:
                    actions = sample_space(
                        env.unwrapped.single_action_space, device=env.unwrapped.device, batch_size=num_envs
                    )
                # apply actions
                transition = env.step(actions)
                # check signals
                for data in transition[:-1]:  # exclude info
                    if multi_agent:
                        for agent, agent_data in data.items():
                            assert _check_valid_tensor(agent_data), f"Invalid data ('{agent}'): {agent_data}"
                    else:
                        assert _check_valid_tensor(data), f"Invalid data: {data}"

    finally:
        # Always ensure cleanup happens, regardless of success or failure
        if env is not None:
            env.close()

        # Clear the simulation context singleton (also closes the USD context stage)
        SimulationContext.clear_instance()


def _check_valid_tensor(data: torch.Tensor | dict) -> bool:
    """Checks if given data does not have corrupted values.

    Args:
        data: Data buffer.

    Returns:
        True if the data is valid.
    """
    if isinstance(data, torch.Tensor):
        return not torch.any(torch.isnan(data))
    elif isinstance(data, (tuple, list)):
        return all(_check_valid_tensor(value) for value in data)
    elif isinstance(data, dict):
        return all(_check_valid_tensor(value) for value in data.values())
    else:
        raise ValueError(f"Input data of invalid type: {type(data)}.")
