# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for environments."""


import os

# submodules
from .parse_cfg import load_default_env_cfg, parse_env_cfg

__all__ = ["load_default_env_cfg", "parse_env_cfg", "get_checkpoint_path"]


def get_checkpoint_path(log_path: str, run_dir: str = "*", checkpoint: str = None) -> str:
    """Get path to the model checkpoint in input directory.

    The checkpoint file is resolved as: <log_path>/<run_dir>/<checkpoint>.

    Args:
        log_path: The log directory path to find models in.
        run_dir: The name of the directory containing the run. Defaults to the most
            recent directory created inside :obj:`log_dir`.
        checkpoint: The model checkpoint file or directory name. Defaults to the most recent
            recent torch-model saved in the :obj:`run_dir` directory.

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
        # find all runs in the directory
        runs = [os.path.join(log_path, run) for run in os.scandir(log_path)]
        runs = [run for run in runs if os.path.isdir(run)]
        # sort by date to handle change of month
        runs = sorted(runs, key=os.path.getmtime)
        # create last run file path
        last_run_path = runs[-1]
    except IndexError:
        raise ValueError(f"No runs present in the directory: {log_path}")
    # path to the directory containing the run
    if run_dir.startswith("*"):
        # check if there are other paths. Example: "*/nn"
        run_path = run_dir.replace("*", last_run_path)
    else:
        run_path = os.path.join(log_path, run_dir)
    # name of model checkpoint
    if checkpoint is None:
        # list all model checkpoints in the directory
        model_checkpoints = [f for f in os.listdir(run_path) if ".pt" in f]
        # check if any checkpoints are present
        if len(model_checkpoints) == 0:
            raise ValueError(f"No checkpoints present in the directory: {run_path}")
        # sort by date
        model_checkpoints.sort(key=lambda m: f"{m:0>15}")
        # get latest model checkpoint file
        checkpoint_file = model_checkpoints[-1]
    else:
        checkpoint_file = checkpoint

    return os.path.join(run_path, checkpoint_file)
