# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab_tasks.utils.wrappers.torchrl import OnPolicyPPORunnerCfg


def add_torchrl_args(parser: argparse.ArgumentParser):
    """Add TorchRL arguments to the parser.

    Adds the following fields to argparse:
        - "--experiment_name" : Name of the experiment folder where logs will be stored (default: None).
        - "--run_name" : Run name suffix to the log directory (default: None).
        - "--resume" : Whether to resume from a checkpoint (default: None).
        - "--load_run" : Name of the run folder to resume from (default: None).
        - "--checkpoint" : Checkpoint file to resume from (default: None).
        - "--logger" : Logger module to use (default: None).
        - "--log_project_name" : Name of the logging project when using wandb or neptune (default: None).
    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("torchrl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    arg_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name suffix to the log directory.",
    )
    # -- load arguments
    arg_group.add_argument(
        "--resume",
        type=bool,
        default=None,
        help="Whether to resume from a checkpoint.",
    )
    arg_group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Name of the run folder to resume from.",
    )
    arg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file to resume from.",
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Name of the logging project when using wandb or neptune.",
    )


def parse_torchrl_cfg(task_name: str, args_cli: argparse.Namespace) -> OnPolicyPPORunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    torchrl_cfg: OnPolicyPPORunnerCfg = load_cfg_from_registry(task_name, "torchrl_cfg_entry_point")

    # override the default configuration with CLI arguments
    torchrl_cfg.device = "cpu" if args_cli.cpu else f"cuda:{args_cli.physics_gpu}"

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        torchrl_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        torchrl_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        torchrl_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        torchrl_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        torchrl_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        torchrl_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if torchrl_cfg.logger == "wandb" and args_cli.log_project_name:
        torchrl_cfg.wandb_project = args_cli.log_project_name

    return torchrl_cfg


def update_torchrl_cfg(agent_cfg: OnPolicyPPORunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for torchrl agent based on inputs.

    Args:
        agent_cfg: The configuration for torchrl agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for torchrl agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name

    return agent_cfg
