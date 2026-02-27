# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for the hydra configuration system."""

import functools
import logging
from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

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
from isaaclab_tasks.utils.render_config_store import register_render_configs

# Renderer type options for Hydra config groups; default when missing.
RENDERER_TYPE_OPTIONS = ("isaac_rtx", "newton_warp")
DEFAULT_RENDERER_TYPE = "isaac_rtx"


def _normalize_renderer_type_in_dict(d: dict) -> None:
    """In-place: where renderer_type is a dict from a Hydra group (single key isaac_rtx/newton_warp), replace with that string."""
    for key, value in list(d.items()):
        if key == "renderer_type" and isinstance(value, Mapping):
            keys_in = [k for k in value.keys() if k in RENDERER_TYPE_OPTIONS]
            if len(keys_in) == 1:
                d["renderer_type"] = keys_in[0]
                continue
        if isinstance(value, Mapping) and not isinstance(value, type):
            _normalize_renderer_type_in_dict(value)


def _instantiate_renderer_cfg_at(obj: object, _seen: set | None = None) -> None:
    """Recursively walk config and replace renderer_cfg with an instance; default to RTX when renderer_type missing."""
    _seen = _seen or set()
    obj_id = id(obj)
    if obj_id in _seen:
        return
    _seen.add(obj_id)

    if not hasattr(obj, "renderer_cfg"):
        pass
    else:
        from isaaclab.renderers import renderer_cfg_from_type

        rt = getattr(obj, "renderer_type", None) or DEFAULT_RENDERER_TYPE
        cfg = renderer_cfg_from_type(rt)
        if hasattr(obj, "data_types") and getattr(obj, "data_types", None) is not None:
            cfg.data_types = list(obj.data_types)
        setattr(obj, "renderer_cfg", cfg)
        logger.info(
            "Env config: passing concrete renderer config (not string) — %s for renderer_type=%s",
            type(cfg).__name__,
            rt,
        )

    def recurse(v):
        if isinstance(v, Mapping):
            for val in v.values():
                _instantiate_renderer_cfg_at(val, _seen)
        elif hasattr(v, "__dict__") and not callable(v) and not isinstance(v, type):
            _instantiate_renderer_cfg_at(v, _seen)

    if isinstance(obj, Mapping):
        for val in obj.values():
            recurse(val)
    else:
        fields = getattr(obj, "__dataclass_fields__", None)
        keys = list(fields.keys()) if fields is not None else (list(vars(obj).keys()) if hasattr(obj, "__dict__") else ())
        for key in keys:
            if key.startswith("_"):
                continue
            try:
                v = getattr(obj, key)
            except (AttributeError, KeyError):
                continue
            if isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Mapping) or (hasattr(item, "__dict__") and not callable(item) and not isinstance(item, type)):
                        recurse(item)
            else:
                recurse(v)


def instantiate_renderer_cfg_in_env(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> None:
    """Replace renderer_type with instantiated renderer_cfg everywhere in env config.

    After Hydra applies overrides (e.g. env.scene.base_camera.renderer_type=newton_warp),
    call this so that env_cfg.scene.base_camera.renderer_cfg is a concrete RendererCfg instance
    (e.g. NewtonWarpRendererCfg) instead of relying on string resolution later in TiledCamera.
    """
    logger.info(
        "Instantiating renderer config in env: replacing renderer_type strings with concrete RendererCfg instances."
    )
    _instantiate_renderer_cfg_at(env_cfg)


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
    # register render config presets and store the configuration to Hydra
    register_render_configs()
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
                # log Hydra overrides that set renderer (e.g. env.scene.base_camera.renderer_type=newton_warp)
                hydra_cfg = HydraConfig.get()
                overrides_list = getattr(getattr(hydra_cfg, "overrides", None), "task", None) or []
                renderer_overrides = [o for o in overrides_list if "renderer_type" in o]
                if renderer_overrides:
                    logger.info("Hydra overrides overriding renderer config: %s", renderer_overrides)
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # apply renderer config to all cameras (in scene and at env level, e.g. tiled_camera)
                if "renderer" in hydra_env_cfg and hydra_env_cfg["renderer"]:
                    renderer_dict = hydra_env_cfg["renderer"]
                    if isinstance(renderer_dict, dict):
                        env_dict = hydra_env_cfg.get("env", {})

                        def apply_to_cameras(d: dict) -> None:
                            for v in d.values():
                                if isinstance(v, dict):
                                    if "renderer_cfg" in v:
                                        v["renderer_cfg"] = renderer_dict
                                    apply_to_cameras(v)

                        apply_to_cameras(env_dict)
                # normalize renderer_type: Hydra config groups can leave it as a dict {option: node}; flatten to string
                _normalize_renderer_type_in_dict(hydra_env_cfg["env"])
                # update the group configs with Hydra command line arguments
                runtime_choice = hydra_cfg.runtime.choices
                resolve_hydra_group_runtime_override(env_cfg, agent_cfg, hydra_env_cfg, runtime_choice)
                # update the configs with the Hydra command line arguments (strict=False: skip keys from replaced group nodes)
                env_cfg.from_dict(hydra_env_cfg["env"], strict=False)
                # instantiate renderer_cfg from renderer_type so cameras get a concrete RendererCfg
                instantiate_renderer_cfg_in_env(env_cfg)
                # replace strings that represent gymnasium spaces because OmegaConf does not support them.
                # this must be done after converting the env configs from dictionary to avoid internal reinterpretations
                env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
                # get agent configs
                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"], strict=False)
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

            # call the new Hydra main function
            hydra_main()

        return wrapper

    return decorator


def _find_renderer_cfg_paths(node: object, prefix: str = "env") -> list[str]:
    """Recursively find paths under env where the value has renderer_type or renderer_cfg (TiledCameraCfg-like)."""
    paths: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, Mapping):
                if "renderer_type" in value or "renderer_cfg" in value:
                    paths.append(path)
                paths.extend(_find_renderer_cfg_paths(value, path))
            else:
                if hasattr(value, "renderer_type") or hasattr(value, "renderer_cfg"):
                    paths.append(path)
                if hasattr(value, "__dict__") and not isinstance(value, type):
                    paths.extend(_find_renderer_cfg_paths(vars(value), path))
    return paths


def _register_renderer_type_groups(cfg_dict: dict, cs: ConfigStore) -> list[dict]:
    """Register Hydra config groups for renderer_type (isaac_rtx, newton_warp); return default entries for defaults list."""
    default_entries: list[dict] = []

    def add_group(group_path: str) -> None:
        for opt in RENDERER_TYPE_OPTIONS:
            cs.store(group=group_path, name=opt, node=opt)
        default_entries.append({group_path: DEFAULT_RENDERER_TYPE})

    env = cfg_dict.get("env") if isinstance(cfg_dict, Mapping) else getattr(cfg_dict, "env", None)
    if env is None:
        return default_entries
    for path in _find_renderer_cfg_paths(env, "env"):
        add_group(f"{path}.renderer_type")
    return default_entries


def register_hydra_group(cfg_dict: dict) -> None:
    """Register Hydra config groups for variant entries and prime defaults.

    Also registers config groups for renderer type (env.scene.base_camera.renderer_type, etc.)
    with options isaac_rtx and newton_warp. Composed group output is normalized to a string
    before from_dict via _normalize_renderer_type_in_dict().
    """
    cs = ConfigStore.instance()
    default_groups: list[str] = []

    for section in ("env", "agent"):
        section_dict = cfg_dict.get(section, {})
        if isinstance(section_dict, dict) and "variants" in section_dict:
            for root_name, root_dict in section_dict["variants"].items():
                group_path = f"{section}.{root_name}"
                default_groups.append(group_path)
                cs.store(group=group_path, name="default", node=getattr_nested(cfg_dict, group_path))
                for variant_name, variant_node in root_dict.items():
                    cs.store(group=group_path, name=variant_name, node=variant_node)

    renderer_defaults = _register_renderer_type_groups(cfg_dict, cs)
    cfg_dict["defaults"] = ["_self_", {"renderer": "isaac_rtx"}] + [{g: "default"} for g in default_groups] + renderer_defaults


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
            # Do not overwrite hydra_cfg[sec][key]: Hydra already composed the variant with
            # overrides (e.g. env.scene.base_camera.renderer_type=newton_warp). Keeping the
            # composed value ensures from_dict() later applies those overrides to the config object.
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
