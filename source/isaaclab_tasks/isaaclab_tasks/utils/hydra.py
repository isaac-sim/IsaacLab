# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for the hydra configuration system."""


import functools
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
    # --- ENV variants → register groups + record defaults
    register_hydra_group(cfg_dict)
    # store the configuration to Hydra
    ConfigStore.instance().store(name=task_name, node=OmegaConf.create(cfg_dict), group=None)
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
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # update the group configs with Hydra command line arguments
                runtime_choice = HydraConfig.get().runtime.choices
                resolve_hydra_group_runtime_override(env_cfg, agent_cfg, hydra_env_cfg, runtime_choice)
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])
                # replace strings that represent gymnasium spaces because OmegaConf does not support them.
                # this must be done after converting the env configs from dictionary to avoid internal reinterpretations
                env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
                # get agent configs
                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main()

        return wrapper

    return decorator


def register_hydra_group(cfg_dict: dict) -> None:
    """Register Hydra config groups for variant entries and prime defaults.

    The helper inspects the ``env`` and ``agent`` sections of ``cfg_dict`` for ``variants`` mappings,
    registers each group/variant pair with Hydra's :class:`~hydra.core.config_store.ConfigStore`, and
    records a ``defaults`` list so Hydra selects the ``default`` variant unless overridden.

    Args:
        cfg_dict: Mutable configuration dictionary generated for Hydra consumption.
    """
    cs = ConfigStore.instance()
    default_groups: list[str] = []

    for section in ("env", "agent"):
        section_dict = cfg_dict.get(section, {})
        if isinstance(section_dict, dict) and "variants" in section_dict:
            for root_name, root_dict in section_dict["variants"].items():
                group_path = f"{section}.{root_name}"
                default_groups.append(group_path)
                # register the default node pointing at cfg_dict[section][root_name]
                cs.store(group=group_path, name="default", node=getattr_nested(cfg_dict, group_path))
                # register each variant under that group
                for variant_name, variant_node in root_dict.items():
                    cs.store(group=group_path, name=variant_name, node=variant_node)

    cfg_dict["defaults"] = ["_self_"] + [{g: "default"} for g in default_groups]


def resolve_hydra_group_runtime_override(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    agent_cfg: dict | object,
    hydra_cfg: dict,
    choices_runtime: dict = {},
) -> None:
    """Resolve runtime Hydra overrides for registered variant groups.

    Hydra tracks user-selected variants under ``HydraConfig.get().runtime.choices``. Given the original
    environment and agent configuration objects plus the Hydra-parsed dictionary, this function replaces
    the default variant nodes with the selected ones (excluding explicit ``default``) so downstream code
    consumes the correct configuration objects and dictionaries.

    This function also works in contexts without ``hydra.main`` (e.g., tests using ``hydra.compose``):
    it falls back to reading choices from ``hydra_cfg['hydra']['runtime']['choices']`` if
    ``HydraConfig.get()`` is not initialized.

    Args:
        env_cfg: Environment configuration object, typically a dataclass with optional ``variants`` mapping.
        agent_cfg: Agent configuration, either a mutable mapping or object exposing ``variants`` entries.
        hydra_cfg: Native dictionary that mirrors the Hydra config tree, including the ``hydra`` section.
    """
    # Try to read choices from HydraConfig; fall back to hydra_cfg dict if unavailable.
    vrnt = "variants"
    get_variants = lambda c: getattr(c, vrnt, None) or (c.get(vrnt) if isinstance(c, Mapping) else None)  # noqa: E731
    is_group_variant = lambda k, v: k.startswith(pref) and k[cut:] in var and v != "default"  # noqa: E731
    for sec, cfg in (("env", env_cfg), ("agent", agent_cfg)):
        var = get_variants(cfg)
        if not var:
            continue
        pref, cut = f"{sec}.", len(sec) + 1
        choices = {k[cut:]: v for k, v in choices_runtime.items() if is_group_variant(k, v)}
        for key, choice in choices.items():
            node = var[key][choice]
            setattr_nested(cfg, key, node)
            setattr_nested(hydra_cfg[sec], key, node.to_dict() if hasattr(node, "to_dict") else node)
        delattr_nested(cfg, vrnt)
        delattr_nested(hydra_cfg, f"{sec}.variants")


def setattr_nested(obj: object, attr_path: str, value: object) -> None:
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:
        obj = obj[attr] if isinstance(obj, Mapping) else getattr(obj, attr)
    if isinstance(obj, Mapping):
        obj[attrs[-1]] = value
    else:
        setattr(obj, attrs[-1], value)


def getattr_nested(obj: object, attr_path: str) -> object:
    for attr in attr_path.split("."):
        obj = obj[attr] if isinstance(obj, Mapping) else getattr(obj, attr)
    return obj


def delattr_nested(obj: object, attr_path: str) -> None:
    """Delete a nested attribute/key strictly (raises on missing path).

    Uses dict indexing and getattr for traversal, mirroring getattr_nested's strictness.
    """
    if "." in attr_path:
        parent_path, leaf = attr_path.rsplit(".", 1)
        parent = getattr_nested(obj, parent_path)  # may raise KeyError/AttributeError
    else:
        parent, leaf = obj, attr_path
    if isinstance(parent, Mapping):
        del parent[leaf]
    else:
        delattr(parent, leaf)
