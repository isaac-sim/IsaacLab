# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CLI and recurrent-state helpers for LEAPP policy export."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def add_common_export_args(parser: argparse.ArgumentParser, *, agent_default: str) -> None:
    """Add CLI arguments shared by all LEAPP export backends.

    Args:
        parser: Argument parser to extend.
        agent_default: Default Hydra agent configuration entry point for the backend.
    """
    from isaaclab.app import AppLauncher

    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--agent",
        type=str,
        default=agent_default,
        help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument(
        "--use_pretrained_checkpoint",
        action="store_true",
        help="Use the pre-trained checkpoint from Nucleus.",
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
        default="onnx-dynamo",
        choices=["onnx-dynamo", "onnx-torchscript", "jit-script", "jit-trace"],
        help="Method to export the policy",
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
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
    *,
    agent_library: str | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments with preset support and force headless mode."""
    from isaaclab_tasks.utils import setup_preset_cli

    args_cli, hydra_args = setup_preset_cli(parser, argv, agent_library=agent_library)
    args_cli.headless = True
    return args_cli, hydra_args


def disable_torchscript_for_export() -> None:
    """Disable TorchScript compilation so ``@torch.jit.script`` helpers stay traceable.

    LEAPP traces the observation and action pipeline in Python. A compiled
    :class:`torch.jit.ScriptFunction` is opaque to the tracer, so any environment
    quantity flowing through one (for example the quaternion helpers in
    ``isaaclab.utils.math``) is folded into the graph as a constant and the exported
    policy fails validation once that quantity changes.

    Call this before importing task or environment modules: :func:`torch.jit.script`
    compiles at decoration time, so disabling afterwards has no effect on helpers that
    were already imported.
    """
    import torch

    torch.jit._state.disable()


def is_two_tensor_lstm_state(states: object) -> bool:
    """Return whether *states* looks like an LSTM ``[hidden, cell]`` state."""
    import torch

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
