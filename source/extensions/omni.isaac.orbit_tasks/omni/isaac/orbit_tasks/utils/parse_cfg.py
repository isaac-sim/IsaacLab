# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for parsing and loading configurations."""

from __future__ import annotations

import gymnasium as gym
import importlib
import inspect
import os
import re
import yaml

from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.utils import update_class_from_dict, update_dict


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict | RLTaskEnvCfg:
    """Load default configuration given its entry point from the gym registry.

    This function loads the configuration object from the gym registry for the given task name.
    It supports both YAML and Python configuration files.

    It expects the configuration to be registered in the gym registry as:

    .. code-block:: python

        gym.register(
            id="My-Awesome-Task-v0",
            ...
            kwargs={"env_entry_point_cfg": "path.to.config:ConfigClass"},
        )

    The parsed configuration object for above example can be obtained as:

    .. code-block:: python

        from omni.isaac.orbit_tasks.utils.parse_cfg import load_cfg_from_registry

        cfg = load_cfg_from_registry("My-Awesome-Task-v0", "env_entry_point_cfg")

    Args:
        task_name: The name of the environment.
        entry_point_key: The entry point key to resolve the configuration file.

    Returns:
        The parsed configuration object. This is either a dictionary or a class object.

    Raises:
        ValueError: If the entry point key is not available in the gym registry for the task.
    """
    # obtain the configuration entry point
    cfg_entry_point = gym.spec(task_name).kwargs.get(entry_point_key)
    # check if entry point exists
    if cfg_entry_point is None:
        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'."
            f" Please check that the gym registry has the entry point: '{entry_point_key}'."
        )
    # parse the default config file
    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            # absolute path for the config file
            config_file = cfg_entry_point
        else:
            # resolve path to the module location
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            # obtain the configuration file path
            config_file = os.path.join(mod_path, file_name)
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.full_load(f)
    else:
        if callable(cfg_entry_point):
            # resolve path to the module location
            mod_path = inspect.getfile(cfg_entry_point)
            # load the configuration
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            # resolve path to the module location
            mod_name, attr_name = cfg_entry_point.split(":")
            mod = importlib.import_module(mod_name)
            cfg_cls = getattr(mod, attr_name)
        else:
            cfg_cls = cfg_entry_point
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
        if callable(cfg_cls):
            cfg = cfg_cls()
        else:
            cfg = cfg_cls
    return cfg


def parse_env_cfg(
    task_name: str, use_gpu: bool | None = None, num_envs: int | None = None, use_fabric: bool | None = None
) -> dict | RLTaskEnvCfg:
    """Parse configuration for an environment and override based on inputs.

    Args:
        task_name: The name of the environment.
        use_gpu: Whether to use GPU/CPU pipeline. Defaults to None, in which case it is left unchanged.
        num_envs: Number of environments to create. Defaults to None, in which case it is left unchanged.
        use_fabric: Whether to enable/disable fabric interface. If false, all read/write operations go through USD.
            This slows down the simulation but allows seeing the changes in the USD through the USD stage.
            Defaults to None, in which case it is left unchanged.

    Returns:
        The parsed configuration object. This is either a dictionary or a class object.

    Raises:
        ValueError: If the task name is not provided, i.e. None.
    """
    # check if a task name is provided
    if task_name is None:
        raise ValueError("Please provide a valid task name. Hint: Use --task <task_name>.")
    # create a dictionary to update from
    args_cfg = {"sim": {"physx": dict()}, "scene": dict()}
    # resolve pipeline to use (based on input)
    if use_gpu is not None:
        if not use_gpu:
            args_cfg["sim"]["use_gpu_pipeline"] = False
            args_cfg["sim"]["physx"]["use_gpu"] = False
            args_cfg["sim"]["device"] = "cpu"
        else:
            args_cfg["sim"]["use_gpu_pipeline"] = True
            args_cfg["sim"]["physx"]["use_gpu"] = True
            args_cfg["sim"]["device"] = "cuda:0"

    # disable fabric to read/write through USD
    if use_fabric is not None:
        args_cfg["sim"]["use_fabric"] = use_fabric

    # number of environments
    if num_envs is not None:
        args_cfg["scene"]["num_envs"] = num_envs

    # load the default configuration
    cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    # update the main configuration
    if isinstance(cfg, dict):
        cfg = update_dict(cfg, args_cfg)
    else:
        update_class_from_dict(cfg, args_cfg)

    return cfg


def get_checkpoint_path(
    log_path: str, run_dir: str = ".*", checkpoint: str = ".*", other_dirs: list[str] = None, sort_alpha: bool = True
) -> str:
    """Get path to the model checkpoint in input directory.

    The checkpoint file is resolved as: ``<log_path>/<run_dir>/<*other_dirs>/<checkpoint>``, where the
    :attr:`other_dirs` are intermediate folder names to concatenate. These cannot be regex expressions.

    If :attr:`run_dir` and :attr:`checkpoint` are regex expressions then the most recent (highest alphabetical order)
    run and checkpoint are selected. To disable this behavior, set the flag :attr:`sort_alpha` to False.

    Args:
        log_path: The log directory path to find models in.
        run_dir: The regex expression for the name of the directory containing the run. Defaults to the most
            recent directory created inside :attr:`log_path`.
        other_dirs: The intermediate directories between the run directory and the checkpoint file. Defaults to
            None, which implies that checkpoint file is directly under the run directory.
        checkpoint: The regex expression for the model checkpoint file. Defaults to the most recent
            torch-model saved in the :attr:`run_dir` directory.
        sort_alpha: Whether to sort the runs by alphabetical order. Defaults to True.
            If False, the folders in :attr:`run_dir` are sorted by the last modified time.

    Raises:
        ValueError: When no runs are found in the input directory.
        ValueError: When no checkpoints are found in the input directory.

    Returns:
        The path to the model checkpoint.

    Reference:
        https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/utils/helpers.py#L103
    """
    # check if runs present in directory
    try:
        # find all runs in the directory that math the regex expression
        runs = [
            os.path.join(log_path, run) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)
        ]
        # sort matched runs by alphabetical order (latest run should be last)
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)
        # create last run file path
        if other_dirs is not None:
            run_path = os.path.join(runs[-1], *other_dirs)
        else:
            run_path = runs[-1]
    except IndexError:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.")

    # list all model checkpoints in the directory
    model_checkpoints = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
    # check if any checkpoints are present
    if len(model_checkpoints) == 0:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")
    # sort alphabetically while ensuring that *_10 comes after *_9
    model_checkpoints.sort(key=lambda m: f"{m:0>15}")
    # get latest matched checkpoint file
    checkpoint_file = model_checkpoints[-1]

    return os.path.join(run_path, checkpoint_file)
