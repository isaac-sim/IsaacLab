# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command line argument utilities for RLinf integration with IsaacLab."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).parent.absolute())


def resolve_config_dir(config_name: str, explicit_path: str | None) -> str:
    """Return the directory that contains ``<config_name>.yaml``.

    Resolution order:
    1. *explicit_path* if provided (``--config_path``).
    2. Walk the ``isaaclab_tasks`` package tree looking for a matching YAML.
    3. Fall back to the package directory containing the RLinf entrypoint implementation.
    """
    if explicit_path is not None:
        return explicit_path

    spec = importlib.util.find_spec("isaaclab_tasks")
    if spec is not None and spec.origin is not None:
        tasks_root = Path(spec.origin).parent
        matches = list(tasks_root.rglob(f"{config_name}.yaml"))
        if matches:
            return str(matches[0].parent)

    return _SCRIPT_DIR


def _resolve_rlinf_checkpoint(checkpoint: str) -> str:
    """Return the absolute ``full_weights.pt`` path: ``checkpoint`` is the file or a directory holding exactly one.

    Absolute because the Ray workers do not share the launcher's working directory.
    """
    path = Path(checkpoint).expanduser().resolve()
    if path.is_dir():
        matches = list(path.rglob("full_weights.pt"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one full_weights.pt under {path}, found {len(matches)}.")
        path = matches[0]
    if not path.is_file():
        raise FileNotFoundError(f"RLinf checkpoint does not exist: {path}")
    return str(path)


def _resolve_rlinf_resume_dir(weights_path: str) -> str:
    """Return the enclosing ``global_step_<N>`` directory; RLinf resumes from it and parses the step from its name."""
    for parent in Path(weights_path).parents:
        if parent.name.startswith("global_step_"):
            return str(parent)
    raise ValueError(f"{weights_path} is not inside a global_step_<N> directory.")


def add_rlinf_args(parser: argparse.ArgumentParser) -> None:
    """Add RLinf arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
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
        help=(
            "RL-finetuned RLinf checkpoint: the full_weights.pt file, or its global_step_<N> directory "
            "(or any directory between the two)."
        ),
    )
    arg_group.add_argument(
        "--only_eval", action="store_true", default=False, help="Only run evaluation without training."
    )
