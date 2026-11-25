# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Arguments and training checkpoint policy shared by RSL-RL entrypoints.

Keep training checkpoint selection here; benchmarks must not import another entrypoint
or add a separate resume policy. Explicit references must not overwrite filename patterns.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.string import string_to_callable

from isaaclab_tasks.utils import get_checkpoint_path

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlDistillationRunnerCfg
from ...utils.wandb import is_wandb_checkpoint, resolve_wandb_checkpoint
from ..common import CHECKPOINT_SELECTORS, resolve_checkpoint_selector, resolve_seed


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path, latest/best, pretrained for play, or a Weights & Biases run"
            " (https://wandb.ai/<entity>/<project>/runs/<run_id>, optionally with a '?checkpoint=<iteration>' query,"
            " or the wandb:<entity>/<project>/<run_id> shorthand)."
        ),
    )
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def register_external_tasks(argv: list[str]) -> list[str] | None:
    """Run the ``--external_callback`` named in *argv* and return the arguments it did not consume.

    Downstream code registers its tasks in the callback, so it has to run before the preset setup
    reads the Gym metadata of those tasks.

    Args:
        argv: Command-line arguments excluding the executable name.

    Returns:
        The arguments left for Hydra, or None when no callback was requested.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--external_callback")
    args, _ = parser.parse_known_args(argv)
    if not args.external_callback:
        return None
    return string_to_callable(args.external_callback, separator=".")()


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Load the registered RSL-RL agent configuration of a task and apply the command-line overrides.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The updated RSL-RL agent configuration.
    """
    # the task registry is only needed by callers that bypass the Hydra task resolution
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    return update_rsl_rl_cfg(agent_cfg, args_cli)


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Override an RSL-RL agent configuration with the command-line arguments.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated RSL-RL agent configuration.
    """
    agent_cfg.device = args_cli.device
    if getattr(args_cli, "seed", None) is not None:
        args_cli.seed = resolve_seed(args_cli.seed)
        agent_cfg.seed = args_cli.seed
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    return agent_cfg


def resolve_rsl_rl_training_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, log_root_path: str
) -> str | None:
    """Select the initial checkpoint for training and training benchmarks.

    An explicit reference takes precedence over configured resume. RSL-RL decides whether
    the checkpoint initializes a distillation teacher or restores a student.

    Args:
        args_cli: Training arguments containing checkpoint, task, and agent selections.
        agent_cfg: Runner configuration containing the resume policy and filename pattern.
        log_root_path: Experiment directory searched only for configured resume or selectors.

    Returns:
        The checkpoint path, or None for fresh on-policy or off-policy training.
    """
    if args_cli.checkpoint and is_wandb_checkpoint(args_cli.checkpoint):
        return resolve_wandb_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=args_cli.task,
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint:
        run_dir = Path(args_cli.checkpoint)
        if not run_dir.is_dir():
            return retrieve_file_path(args_cli.checkpoint)
        checkpoints = [
            path for path in run_dir.iterdir() if path.is_file() and re.fullmatch(r"model_.*\.pt", path.name)
        ]
        if not checkpoints:
            raise ValueError(f"No checkpoints in the directory: '{run_dir}'.")
        checkpoints.sort(
            key=lambda path: [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", path.name)]
        )
        return str(checkpoints[-1].resolve())
    if agent_cfg.resume:
        return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    if isinstance(agent_cfg, RslRlDistillationRunnerCfg):
        raise ValueError("Distillation training requires --checkpoint.")
    return None
