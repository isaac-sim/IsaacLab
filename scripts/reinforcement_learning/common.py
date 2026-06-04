# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common utilities for reinforcement learning entrypoints."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym

from isaaclab.app import add_launcher_args
from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml


def dispatch_library_entrypoint(
    argv: list[str] | None,
    entrypoints: dict[str, Path],
    *,
    action: str,
    description: str,
    library_help: str,
    run_as_script: bool = False,
) -> int:
    """Dispatch a unified entrypoint to a library-specific implementation.

    Args:
        argv: Command-line arguments, excluding the script path.
        entrypoints: Mapping from library name to implementation path.
        action: Action name used to create a unique module name.
        description: Top-level parser description.
        library_help: Help text for the ``--rl_library`` argument.
        run_as_script: Whether to execute the selected implementation as a script.

    Returns:
        Process exit code.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rl_library", choices=sorted(entrypoints), required=True)
    args_cli, library_args = parser.parse_known_args(argv)

    if args_cli.rl_library is None:
        help_parser = argparse.ArgumentParser(description=description)
        help_parser.add_argument("--rl_library", choices=sorted(entrypoints), required=True, help=library_help)
        help_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected library.")
        help_parser.print_help()
        return 0 if "-h" in argv or "--help" in argv else 2

    module_path = entrypoints[args_cli.rl_library]
    if run_as_script:
        original_argv = sys.argv
        original_path = list(sys.path)
        try:
            sys.argv = [str(module_path)] + library_args
            sys.path.insert(0, str(module_path.parent))
            runpy.run_path(str(module_path), run_name="__main__")
        finally:
            sys.argv = original_argv
            sys.path[:] = original_path
        return 0

    module = import_local_module(f"isaaclab_rl_{action}_{args_cli.rl_library}", module_path)
    module.run(library_args)
    return 0


def add_common_train_args(
    parser: argparse.ArgumentParser,
    *,
    agent_default: str | None,
    agent_help: str,
    include_agent: bool = True,
    include_distributed: bool = True,
) -> None:
    """Add common Isaac Lab reinforcement learning training arguments.

    Args:
        parser: The parser to add arguments to.
        agent_default: Default agent config entry point.
        agent_help: Help text for the ``--agent`` argument.
        include_agent: Whether to include the ``--agent`` argument.
        include_distributed: Whether to include the ``--distributed`` argument.
    """
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--video_interval", type=int, default=2000, help="Interval between video recordings (in steps)."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    if include_agent:
        parser.add_argument("--agent", type=str, default=agent_default, help=agent_help)
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    if include_distributed:
        parser.add_argument(
            "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
        )
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
    parser.add_argument(
        "--ray-proc-id",
        "-rid",
        type=int,
        default=None,
        help="Automatically configured by Ray integration, otherwise None.",
    )


def add_isaaclab_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Add Isaac Lab simulation launcher arguments to a parser.

    Args:
        parser: The parser to add arguments to.
    """
    add_launcher_args(parser)


def enable_cameras_for_video(args_cli: argparse.Namespace) -> None:
    """Enable camera rendering when video recording is requested.

    Args:
        args_cli: Parsed command-line arguments.
    """
    if getattr(args_cli, "video", False):
        args_cli.enable_cameras = True


def set_hydra_args(hydra_args: list[str]) -> None:
    """Replace ``sys.argv`` with arguments intended for Hydra.

    Args:
        hydra_args: Remaining command-line arguments not consumed by argparse.
    """
    sys.argv = [sys.argv[0]] + hydra_args


def import_local_module(module_name: str, module_path: Path) -> ModuleType:
    """Import a module from an explicit file path.

    Args:
        module_name: Unique module name to use in ``sys.modules``.
        module_path: Path to the Python file to import.

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def apply_env_overrides(args_cli: argparse.Namespace, env_cfg: Any, *, apply_device: bool = True) -> None:
    """Apply common environment overrides from command-line arguments.

    Args:
        args_cli: Parsed command-line arguments.
        env_cfg: Isaac Lab environment config.
        apply_device: Whether to apply the ``--device`` override for non-distributed runs.
    """
    if getattr(args_cli, "num_envs", None) is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if apply_device and not getattr(args_cli, "distributed", False):
        device = getattr(args_cli, "device", None)
        env_cfg.sim.device = device if device is not None else env_cfg.sim.device


def validate_distributed_device(args_cli: argparse.Namespace) -> None:
    """Reject unsupported CPU distributed training configuration.

    Args:
        args_cli: Parsed command-line arguments.

    Raises:
        ValueError: If distributed training is requested with a CPU device.
    """
    device = getattr(args_cli, "device", None)
    if getattr(args_cli, "distributed", False) and device is not None and "cpu" in device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )


def configure_io_descriptors(env_cfg: Any, args_cli: argparse.Namespace, logger: logging.Logger) -> None:
    """Configure IO descriptor export on supported environment configs.

    Args:
        env_cfg: Isaac Lab environment config.
        args_cli: Parsed command-line arguments.
        logger: Logger used for unsupported environment warnings.
    """
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )


def create_isaaclab_env(
    task: str,
    env_cfg: Any,
    args_cli: argparse.Namespace,
    *,
    convert_marl_to_single_agent: bool,
):
    """Create the Isaac Lab Gymnasium environment.

    Args:
        task: Task name to instantiate.
        env_cfg: Isaac Lab environment config.
        args_cli: Parsed command-line arguments.
        convert_marl_to_single_agent: Whether to convert direct MARL environments to single-agent environments.

    Returns:
        The created Gymnasium environment.
    """
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if convert_marl_to_single_agent and isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
        from isaaclab.envs import multi_agent_to_single_agent

        env = multi_agent_to_single_agent(env)
    return env


def wrap_record_video(env, log_dir: str, args_cli: argparse.Namespace):
    """Wrap an environment with video recording when requested.

    Args:
        env: Gymnasium environment to wrap.
        log_dir: Training log directory.
        args_cli: Parsed command-line arguments.

    Returns:
        The original or video-wrapped environment.
    """
    if not args_cli.video:
        return env

    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "train"),
        "step_trigger": lambda step: step % args_cli.video_interval == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    return gym.wrappers.RecordVideo(env, **video_kwargs)


def dump_train_configs(log_dir: str, env_cfg: Any, agent_cfg: Any) -> None:
    """Dump training configuration files under a run log directory.

    Args:
        log_dir: Training log directory.
        env_cfg: Isaac Lab environment config.
        agent_cfg: Reinforcement learning agent config.
    """
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
