# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CLI, LEAPP session, and recurrent-state helpers for policy export."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch

# LEAPP traces Isaac Lab's Python tensor operations, so TorchScript is disabled before importing task
# or environment modules that compile decorated helpers at import time.
torch.jit._state.disable()

from isaaclab.app import AppLauncher, launch_simulation

from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ..common import normalize_task_name

if TYPE_CHECKING:
    from leapp import GraphConfigs

    from isaaclab.envs import DirectRLEnvCfg, ManagerBasedEnvCfg

__all__ = [
    "add_common_export_args",
    "create_graph_configs",
    "finalize_export_args",
    "get_checkpoint_path",
    "is_two_tensor_lstm_state",
    "leapp_capture",
    "prepare_export_env",
    "resolve_export_save_path",
    "run_export",
    "state_dict_from_sequence",
    "state_sequence_from_registered",
]

_EXPORT_METHOD_ERROR = (
    "--export_method is only supported for manager-based environments. For direct environments, "
    "set export_with directly in the annotate.output_tensors() call instead."
)


def add_common_export_args(parser: argparse.ArgumentParser, *, agent_default: str) -> None:
    """Add CLI arguments shared by all LEAPP export backends.

    Args:
        parser: Argument parser to extend.
        agent_default: Default Hydra agent configuration entry point for the backend.
    """
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default=agent_default, help="Name of the RL agent configuration entry point."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path, or 'pretrained'. Omit for automatic local discovery.",
    )
    parser.add_argument(
        "--export_task_name", type=str, default=None, help="Name of the exported graph. Defaults to the task name."
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
    parser.add_argument("--export_save_path", type=str, default=None, help="Path to save the exported model.")
    parser.add_argument(
        "--validation_steps", type=int, default=5, help="Number of steps to validate the exported model."
    )
    parser.add_argument(
        "--validation_rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for LEAPP output parity validation",
    )
    parser.add_argument(
        "--validation_atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for LEAPP output parity validation",
    )
    parser.add_argument(
        "--disable_graph_visualization",
        action="store_true",
        default=False,
        help="Disable LEAPP graph visualization during compile_graph().",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--limit_cpu_threads", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)


def finalize_export_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> tuple[argparse.Namespace, list[str]]:
    """Parse export arguments with preset support and force headless mode.

    The remainder carries the typed preset selectors (``physics=``, ``renderer=``, ``presets=``) verbatim
    for Hydra.
    """
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    args_cli.headless = True
    return args_cli, hydra_args


def run_export(
    args_cli: argparse.Namespace,
    hydra_args: list[str],
    export_agent: Callable[[argparse.Namespace, Any, Any], bool],
) -> int:
    """Resolve the task configuration, launch the simulation, and export one policy.

    Args:
        args_cli: Parsed export arguments.
        hydra_args: Hydra overrides and preset selectors left over from parsing.
        export_agent: Backend export function receiving the arguments, environment config, and agent config.

    Returns:
        Process exit code, ``0`` when a policy was exported.
    """
    # Hydra reads the preset tokens from sys.argv directly
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + hydra_args
    try:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
        with launch_simulation(env_cfg, args_cli):
            exported = export_agent(args_cli, env_cfg, agent_cfg)
    finally:
        sys.argv = original_argv
    return 0 if exported else 1


def prepare_export_env(
    env: Any, args_cli: argparse.Namespace, *, required_obs_groups: set[str]
) -> tuple[str, Any | None]:
    """Patch a Gymnasium environment for LEAPP tracing and return its policy node and patcher.

    Manager-based environments are patched to export only the observation groups the actor consumes.
    Direct environments annotate their own tensors and therefore reject ``--export_method``.

    Args:
        env: Environment created for the export.
        args_cli: Parsed export arguments.
        required_obs_groups: Observation groups consumed by the actor policy.

    Raises:
        ValueError: If ``--export_method`` is passed for a direct environment.
    """
    # concrete environment classes and the LEAPP runtime load simulation modules, so import them
    # only after launch_simulation has initialized the selected backend
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.leapp import patch_env_for_export
    from isaaclab.utils.leapp.utils import ensure_env_spec_id

    policy_node_name = ensure_env_spec_id(env)
    patcher = None
    if isinstance(env.unwrapped, ManagerBasedRLEnv):
        export_method = args_cli.export_method or "onnx-dynamo"
        patcher = patch_env_for_export(env, export_method=export_method, required_obs_groups=required_obs_groups)
    elif args_cli.export_method is not None:
        raise ValueError(_EXPORT_METHOD_ERROR)
    return policy_node_name, patcher


def resolve_export_save_path(args_cli: argparse.Namespace, library: str, log_dir: str) -> str:
    """Return the directory the exported graph is written to.

    Published checkpoints are exported to a predictable path independent of the Nucleus mirror layout.
    """
    if args_cli.export_save_path is not None:
        return args_cli.export_save_path
    if args_cli.checkpoint == "pretrained":
        return os.path.join(".pretrained_checkpoints", library, normalize_task_name(args_cli.task))
    return log_dir


@contextlib.contextmanager
def leapp_capture(args_cli: argparse.Namespace, *, save_path: str, env_cfg: Any, patcher: Any = None) -> Iterator[int]:
    """Record policy steps with LEAPP and compile the graph once the block completes.

    Args:
        args_cli: Parsed export arguments.
        save_path: Directory the exported graph is written to.
        env_cfg: Environment config providing the policy frequency.
        patcher: Manager-based export patcher, if one was installed.

    Yields:
        The number of policy steps the block has to run; at least two so LEAPP observes a state update.
    """
    # the LEAPP runtime loads simulation modules, so import it only after the launch
    import leapp

    graph_name = args_cli.export_task_name or args_cli.task.split(":")[-1]
    num_steps = max(args_cli.validation_steps, 2)
    leapp.start(graph_name, save_path=save_path, max_cached_io=num_steps)
    try:
        yield num_steps
    except BaseException:
        with contextlib.suppress(Exception):
            leapp.stop()
        raise
    leapp.stop()
    leapp.compile_graph(
        visualize=not args_cli.disable_graph_visualization,
        validate=args_cli.validation_steps > 0,
        rtol=args_cli.validation_rtol,
        atol=args_cli.validation_atol,
        graph_configs=create_graph_configs(
            env_cfg,
            controller_owned_write_requirements=(patcher.controller_owned_write_requirements if patcher else ()),
        ),
    )


def create_graph_configs(
    env_cfg: ManagerBasedEnvCfg | DirectRLEnvCfg,
    *,
    controller_owned_write_requirements: Sequence[dict[str, Any]] = (),
) -> GraphConfigs:
    """Create LEAPP graph metadata from an Isaac Lab environment configuration.

    Args:
        env_cfg: Environment configuration that defines the policy period.
        controller_owned_write_requirements: Controller capabilities that must
            be provided outside the exported policy graph.

    Returns:
        Graph metadata containing the policy frequency [Hz] and deployment
        requirements.
    """
    from leapp import GraphConfigs

    policy_frequency = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    extra = None
    if controller_owned_write_requirements:
        extra = {
            "isaaclab": {
                "controller_owned_writes": {
                    "schema_version": 1,
                    "requirements": list(controller_owned_write_requirements),
                }
            }
        }
    return GraphConfigs(frequency=policy_frequency, extra=extra)


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
