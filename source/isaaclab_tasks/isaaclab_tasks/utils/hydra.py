# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for the hydra configuration system."""


import functools
import gymnasium as gym
from collections.abc import Callable, Mapping

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.")

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


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
        A tuple containing the parsed environment and agent configuration objects.
    """
    # load the configurations
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    agent_cfg = None
    if agent_cfg_entry_point:
        agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)
    # replace gymnasium spaces with strings because OmegaConf does not support them.
    # this must be done before converting the env configs to dictionary to avoid internal reinterpretations
    env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
    # convert the configs to dictionary
    env_cfg_dict = env_cfg.to_dict()
    if isinstance(agent_cfg, dict) or agent_cfg is None:
        agent_cfg_dict = agent_cfg
    else:
        agent_cfg_dict = agent_cfg.to_dict()
    cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}
    # replace slices with strings because OmegaConf does not support slices
    cfg_dict = replace_slices_with_strings(cfg_dict)
    config_store = ConfigStore.instance()

    has_hydra_group_configuration = "configurable_entry_point" in gym.spec(task_name).kwargs
    if has_hydra_group_configuration:
        hydra_groups_cfg = load_cfg_from_registry(task_name, "configurable_entry_point")
        hydra_groups_cfg = replace_env_cfg_spaces_with_strings(hydra_groups_cfg)
        configurables_dict = hydra_groups_cfg.to_dict()
        configurables_dict = replace_slices_with_strings(configurables_dict)
        default_groups = []
        for root_config_name, root_config_dict in configurables_dict.items():
            for group_name, group_option_dict in root_config_dict.items():
                group_path = f"{root_config_name}.{group_name}"
                default_groups.append(group_path)
                config_store.store(group=group_path, name="default", node=getattr_nested(cfg_dict, group_path))
                for option_name, option_val in group_option_dict.items():
                    config_store.store(group=group_path, name=option_name, node=option_val)

        root_defaults = ["_self_"] + [{grp: "default"} for grp in default_groups]
        cfg_dict = {"defaults": root_defaults, **cfg_dict}
    else:
        # no predefined-group cfg override, only field overrides
        hydra_groups_cfg = None
    config_store.store(name=task_name, node=OmegaConf.create(cfg_dict), group=None)

    return env_cfg, agent_cfg, hydra_groups_cfg


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
            env_cfg, agent_cfg, configurables = register_task_to_hydra(task_name.split(":")[-1], agent_cfg_entry_point)

            # define the new Hydra main function
            @hydra.main(config_path=None, config_name=task_name.split(":")[-1], version_base="1.3")
            def hydra_main(
                hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg, configurables=configurables
            ):
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # update the group configs with Hydra command line arguments
                has_hydra_group_configuration = "configurable_entry_point" in gym.spec(task_name).kwargs
                if has_hydra_group_configuration and hasattr(configurables, "env"):
                    hydra_cfg = HydraConfig.get()
                    configurables = replace_strings_with_slices(configurables)
                    for key in configurables.env.keys():
                        cmd_group_choice = hydra_cfg.runtime.choices[f"env.{key}"]
                        if cmd_group_choice != "default":
                            setattr_nested(env_cfg, key, configurables.env[key][cmd_group_choice])
                            setattr_nested(
                                hydra_env_cfg["env"], key, configurables.env[key][cmd_group_choice].to_dict()
                            )
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])
                # replace strings that represent gymnasium spaces because OmegaConf does not support them.
                # this must be done after converting the env configs from dictionary to avoid internal reinterpretations
                env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
                # get agent configs
                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    if has_hydra_group_configuration and hasattr(configurables, "agent"):
                        for key in configurables.agent.keys():
                            cmd_group_choice = hydra_cfg.runtime.choices[f"agent.{key}"]
                            if cmd_group_choice != "default":
                                setattr_nested(agent_cfg, key, configurables.agent[key][cmd_group_choice])
                    agent_cfg.from_dict(hydra_env_cfg["agent"])
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main()

        return wrapper

    return decorator


def setattr_nested(obj: object, attr_path: str, value: object) -> None:
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:
        if isinstance(obj, Mapping):
            obj = obj[attr]
        else:
            obj = getattr(obj, attr)
    last = attrs[-1]
    if isinstance(obj, Mapping):
        obj[last] = value
    else:
        setattr(obj, last, value)


def getattr_nested(obj: object, attr_path: str) -> object:
    attrs = attr_path.split(".")
    for attr in attrs:
        if isinstance(obj, Mapping):
            obj = obj[attr]
        else:
            obj = getattr(obj, attr)
    return obj
