# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CLI, graph metadata, and recurrent-state helpers for LEAPP policy export."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence

import torch
from leapp import GraphConfigs

# TorchScript must be disabled before importing task or environment modules because
# ``@torch.jit.script`` compiles at decoration time.
torch.jit._state.disable()

from isaaclab.app import AppLauncher
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedEnvCfg

from isaaclab_tasks.utils import setup_preset_cli


def add_common_export_args(parser: argparse.ArgumentParser, *, agent_default: str) -> None:
    """Add CLI arguments shared by all LEAPP export backends.

    Args:
        parser: Argument parser to extend.
        agent_default: Default Hydra agent configuration entry point for the backend.
    """

    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--agent",
        type=str,
        default=agent_default,
        help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path, or 'pretrained'. Omit for automatic local discovery.",
    )
    parser.add_argument(
        "--export_task_name",
        type=str,
        default=None,
        help="Name of the exported graph. Defaults to the task name.",
    )
    parser.add_argument(
        "--export_method",
        type=str,
        default=None,
        choices=["onnx-dynamo", "onnx-torchscript", "jit-script", "jit-trace", "pt2"],
        help=(
            "Select the backend based on the artifact format you need. Defaults to onnx-dynamo, which is "
            "recommended unless you have a specific reason to use another backend. If one backend does not "
            "support your model, try another."
        ),
    )
    parser.add_argument(
        "--export_save_path",
        type=str,
        default=None,
        help="Path to save the exported model",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5,
        help="Number of steps to validate the exported model",
    )
    parser.add_argument(
        "--disable_graph_visualization",
        action="store_true",
        default=False,
        help="Disable LEAPP graph visualization during compile_graph().",
    )
    AppLauncher.add_app_launcher_args(parser)


def finalize_export_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments with preset support and force headless mode."""

    args_cli, hydra_args = setup_preset_cli(parser, argv)
    args_cli.headless = True
    return args_cli, hydra_args


def get_checkpoint_path(
    log_path: str,
    run_dir: str = ".*",
    checkpoint: str = ".*",
    other_dirs: list[str] | None = None,
    sort_alpha: bool = True,
    preferred_checkpoint: str | None = None,
) -> str:
    """Resolve a model checkpoint from a run directory.

    The checkpoint is selected from ``<log_path>/<run_dir>/<*other_dirs>``.
    Run and checkpoint names may be regular expressions. The latest matching
    run and naturally sorted checkpoint are returned.

    Args:
        log_path: Log directory containing training runs.
        run_dir: Regular expression matching a run directory.
        checkpoint: Regular expression matching checkpoint files.
        other_dirs: Literal intermediate directories below the run directory.
        sort_alpha: Sort runs alphabetically instead of by modification time.
        preferred_checkpoint: Optional checkpoint expression to try first.

    Returns:
        Path to the selected checkpoint.

    Raises:
        ValueError: If no matching run or checkpoint exists.
    """
    try:
        runs = [
            os.path.join(log_path, run.name)
            for run in os.scandir(log_path)
            if run.is_dir() and re.match(run_dir, run.name)
        ]
        runs.sort() if sort_alpha else runs.sort(key=os.path.getmtime)
        run_path = os.path.join(runs[-1], *other_dirs) if other_dirs is not None else runs[-1]
    except (IndexError, FileNotFoundError):
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.")

    model_checkpoints = []
    if preferred_checkpoint is not None:
        model_checkpoints = [name for name in os.listdir(run_path) if re.match(preferred_checkpoint, name)]
    if not model_checkpoints:
        model_checkpoints = [name for name in os.listdir(run_path) if re.match(checkpoint, name)]
    if not model_checkpoints:
        patterns = f"'{checkpoint}'"
        if preferred_checkpoint is not None:
            patterns = f"'{preferred_checkpoint}' nor '{checkpoint}'"
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match {patterns}.")

    model_checkpoints.sort(
        key=lambda name: [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", name)]
    )
    return os.path.join(run_path, model_checkpoints[-1])


def create_graph_configs(env_cfg: ManagerBasedEnvCfg | DirectRLEnvCfg) -> GraphConfigs:
    """Create LEAPP graph metadata from an Isaac Lab environment configuration.

    Args:
        env_cfg: Environment configuration that defines the policy period.

    Returns:
        Graph metadata containing the policy frequency [Hz].
    """

    policy_frequency = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    return GraphConfigs(frequency=policy_frequency)


def is_two_tensor_lstm_state(states: object) -> bool:
    """Return whether *states* looks like an LSTM ``[hidden, cell]`` state."""

    return (
        isinstance(states, (list, tuple))
        and len(states) == 2
        and all(isinstance(state, torch.Tensor) for state in states)
    )


def state_dict_from_sequence(states: Sequence[torch.Tensor], prefix: str = "actor_state") -> dict[str, torch.Tensor]:
    """Convert an ordered recurrent-state sequence to a LEAPP named-state mapping."""
    return {f"{prefix}_{index}": state for index, state in enumerate(states)}


def state_sequence_from_registered(
    registered_state: object,
    names: Sequence[str],
    original_states: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Restore registered LEAPP state to the ordered sequence expected by an RL framework."""
    if isinstance(registered_state, dict):
        return [registered_state[name] for name in names]
    if isinstance(registered_state, (list, tuple)):
        return list(registered_state)
    if len(original_states) == 1:
        return [registered_state]
    raise TypeError(f"Expected registered recurrent state for {list(names)}, got {type(registered_state).__name__}.")
