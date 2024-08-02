# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for the hydra configuration system."""


import functools
from collections.abc import Callable

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab.utils import replace_slices_with_strings, replace_strings_with_slices

from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry


def register_task_to_hydra(
    task_name: str, agent_cfg_entry_point: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, dict]:
    """Register the task configuration to the Hydra configuration store.

    This function resolves the configuration file for the environment and agent based on the task's name.
    It then registers the configurations to the Hydra configuration store.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        The parsed environment and agent configurations.
    """
    # load the configurations
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)
    # convert the configs to dictionary
    env_cfg_dict = env_cfg.to_dict()
    if isinstance(agent_cfg, dict):
        agent_cfg_dict = agent_cfg
    else:
        agent_cfg_dict = agent_cfg.to_dict()
    cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}
    # replace slices with strings because OmegaConf does not support slices
    cfg_dict = replace_slices_with_strings(cfg_dict)
    # store the configuration to Hydra
    ConfigStore.instance().store(name=task_name, node=cfg_dict)
    return env_cfg, agent_cfg


def hydra_task_config(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Decorator to handle the Hydra configuration for a task.

    This decorator registers the task to Hydra and updates the environment and agent configurations from Hydra parsed
    command line arguments.

    Args:
        task_name: The name of the task.
        agent_cfg_entry_point: The entry point key to resolve the agent's configuration file.

    Returns:
        The decorated function with the envrionment's and agent's configurations updated from command line arguments.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # register the task to Hydra
            env_cfg, agent_cfg = register_task_to_hydra(task_name, agent_cfg_entry_point)

            # define thr new Hydra main function
            @hydra.main(config_path=None, config_name=task_name, version_base="1.3")
            def hydra_main(hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg):
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])
                if isinstance(agent_cfg, dict):
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main()

        return wrapper

    return decorator
