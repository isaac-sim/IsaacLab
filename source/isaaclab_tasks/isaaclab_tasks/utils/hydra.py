# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for the hydra configuration system."""

import functools
from collections.abc import Callable

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    raise ImportError("Hydra is not installed. Please install it by running 'pip install hydra-core'.")

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_tasks.utils.render_config_store import register_render_configs


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
    cfg_dict = {
        "defaults": ["_self_", {"render": "isaac_rtx"}],
        "env": env_cfg_dict,
        "agent": agent_cfg_dict,
    }
    # replace slices with strings because OmegaConf does not support slices
    cfg_dict = replace_slices_with_strings(cfg_dict)
    # register render config presets and store the configuration to Hydra
    register_render_configs()
    ConfigStore.instance().store(name=task_name, node=cfg_dict)
    return env_cfg, agent_cfg


def process_hydra_config(
    hydra_cfg: DictConfig | dict,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    agent_cfg: dict | object,
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, dict | object]:
    """Process composed Hydra config and update env/agent configs in place.

    Shared by hydra_task_config and tests. Applies render config to cameras,
    updates env/agent from dict, restores gymnasium spaces.
    """
    if not isinstance(hydra_cfg, dict):
        hydra_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)
    hydra_cfg = replace_strings_with_slices(hydra_cfg)

    if "render" in hydra_cfg and hydra_cfg["render"]:
        renderer_dict = hydra_cfg["render"]
        if isinstance(renderer_dict, dict):
            env_dict = hydra_cfg.get("env", {})

            def apply_to_cameras(d: dict) -> None:
                for v in d.values():
                    if isinstance(v, dict):
                        if "renderer_cfg" in v:
                            v["renderer_cfg"] = renderer_dict
                        apply_to_cameras(v)

            apply_to_cameras(env_dict)

    env_cfg.from_dict(hydra_cfg["env"])
    env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)

    if isinstance(agent_cfg, dict) or agent_cfg is None:
        agent_cfg = hydra_cfg["agent"]
    else:
        agent_cfg.from_dict(hydra_cfg["agent"])

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
            env_cfg, agent_cfg = register_task_to_hydra(task_name.split(":")[-1], agent_cfg_entry_point)

            # define the new Hydra main function
            @hydra.main(config_path=None, config_name=task_name.split(":")[-1], version_base="1.3")
            def hydra_main(hydra_env_cfg: DictConfig, env_cfg=env_cfg, agent_cfg=agent_cfg):
                env_cfg, agent_cfg = process_hydra_config(hydra_env_cfg, env_cfg, agent_cfg)
                func(env_cfg, agent_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main()

        return wrapper

    return decorator
