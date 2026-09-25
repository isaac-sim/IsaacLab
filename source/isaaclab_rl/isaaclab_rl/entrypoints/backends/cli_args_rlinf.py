# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line arguments and configuration discovery shared by the RLinf entrypoints."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from ..common import CHECKPOINT_SELECTORS, resolve_checkpoint_selector

_BACKENDS_DIR = str(Path(__file__).parent.absolute())


def add_rlinf_args(parser: argparse.ArgumentParser) -> None:
    """Add RLinf arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task (overrides the config if set).")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment (overrides the config).")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the pretrained base model (optional; can be set in config).",
    )
    arg_group = parser.add_argument_group("rlinf", description="Arguments for RLinf agent.")
    arg_group.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "Path to the RLinf configuration directory (for Hydra). "
            "If omitted, the isaaclab_tasks package is searched automatically."
        ),
    )
    arg_group.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Name of the RLinf configuration file (without .yaml extension).",
    )
    arg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="RL-finetuned checkpoint path, or latest/best.",
    )
    arg_group.add_argument(
        "--only_eval", action="store_true", default=False, help="Only run evaluation without training."
    )


def resolve_config_dir(config_name: str, explicit_path: str | None) -> str:
    """Return the directory that contains ``<config_name>.yaml``.

    Resolution order:

    1. *explicit_path* if provided (``--config_path``).
    2. The ``isaaclab_tasks`` package tree, searched for a matching YAML.
    3. The directory containing the RLinf entrypoint implementation.
    """
    if explicit_path is not None:
        return explicit_path
    spec = importlib.util.find_spec("isaaclab_tasks")
    if spec is not None and spec.origin is not None:
        matches = list(Path(spec.origin).parent.rglob(f"{config_name}.yaml"))
        if matches:
            return str(matches[0].parent)
    return _BACKENDS_DIR


def configure_rlinf_environment(config_name: str, config_path: str | None) -> str:
    """Export the environment variables RLinf reads at import time and return the config directory.

    Args:
        config_name: Name of the RLinf configuration file without extension.
        config_path: Explicit configuration directory, or None to search for it.
    """
    config_dir = resolve_config_dir(config_name, config_path)
    # required for RLinf to register Isaac Lab tasks and converters
    os.environ.setdefault("RLINF_EXT_MODULE", "isaaclab_contrib.rl.rlinf.extension")
    os.environ["RLINF_CONFIG_FILE"] = str(Path(config_dir) / f"{config_name}.yaml")
    # Ray rollout workers resolve data_config_class references like "gr00t_config:IsaacLabDataConfig"
    # from the config directory
    if config_dir not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = config_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    return config_dir


def resolve_rlinf_checkpoint(checkpoint: str, *, log_root_path: str, task: str, config_name: str) -> str:
    """Resolve an RLinf checkpoint selector or local path.

    Raises:
        ValueError: If a published pre-trained checkpoint is requested; RLinf has none.
    """
    if checkpoint == "pretrained":
        raise ValueError("Pre-trained checkpoints are not available for RLinf.")
    if checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            checkpoint,
            library="rlinf",
            task=task,
            checkpoint_pattern=r"full_weights[.]pt",
            metadata={"config_name": config_name},
            recursive=True,
        )
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "full_weights.pt"
    return str(checkpoint_path)


def print_rlinf_banner(title: str, fields: dict[str, object]) -> None:
    """Print a framed summary of the resolved RLinf configuration."""
    rule = "=" * 60
    print(f"\n{rule}\n{title}\n{rule}")
    for name, value in fields.items():
        print(f"  {name}: {value}")
    print(f"{rule}\n")
