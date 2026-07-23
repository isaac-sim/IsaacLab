# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common utilities for reinforcement learning entrypoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import runpy
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import torch
from PIL import Image

from isaaclab.app import AppLauncher
from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.images import make_camera_output_grid, normalize_camera_output_for_display
from isaaclab.utils.io import dump_yaml

RUN_MANIFEST_FILENAME = "run.json"
RUN_MANIFEST_VERSION = 1
CHECKPOINT_SELECTORS = frozenset({"latest", "best"})
logger = logging.getLogger(__name__)


class CaptureEnvSensors(gym.Wrapper):
    """Capture image-like environment sensor outputs during training."""

    def __init__(
        self,
        env: gym.Env,
        output_dir: str,
        frame_count: int,
        capture_num_envs: int,
        interval: int,
        output_format: str = "tensorboard",
    ) -> None:
        """Initialize the sensor capture wrapper.

        Args:
            env: Gymnasium environment to wrap.
            output_dir: Directory where captured frames are written.
            frame_count: Number of frames to capture per interval.
            capture_num_envs: Number of environment views to capture from each sensor.
            interval: Number of environment steps between capture windows.
            output_format: Output format. Can be ``"tensorboard"`` or ``"file"``.
        """
        super().__init__(env)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_count = max(frame_count, 0)
        self.capture_num_envs = max(capture_num_envs, 0)
        self.interval = max(interval, 1)
        self._step_count = 0
        self._global_step_count = 0
        self._episode_index = 0
        self.writer = None

        if output_format not in {"tensorboard", "file"}:
            raise ValueError(f"Unsupported sensor capture output format: {output_format}")
        if output_format == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.output_dir)

    def reset(self, **kwargs) -> Any:
        """Reset the wrapped environment and capture the reset frame when scheduled."""
        result = self.env.reset(**kwargs)
        self._step_count = 0
        self._episode_index += 1
        self._save_frame()
        return result

    def step(self, action) -> Any:
        """Step the wrapped environment and capture the resulting frame when scheduled."""
        result = self.env.step(action)
        self._step_count += 1
        self._global_step_count += 1
        self._save_frame()
        return result

    def close(self) -> None:
        """Close the writer and wrapped environment."""
        if self.writer is not None:
            self.writer.close()
        super().close()

    def _save_frame(self) -> None:
        """Write the current sensor outputs when the current step is inside a capture window."""
        if self.frame_count == 0:
            return
        if self._step_count % self.interval >= self.frame_count:
            return

        sensors = getattr(getattr(self.unwrapped, "scene", None), "sensors", {})

        for sensor_name, sensor in sensors.items():
            camera_outputs = getattr(getattr(sensor, "data", None), "output", None)
            if not isinstance(camera_outputs, dict):
                continue

            for data_type, output in camera_outputs.items():
                if output is None:
                    continue
                tensor = output if isinstance(output, torch.Tensor) else output.torch
                tensor = tensor[: self.capture_num_envs].detach().clone()
                condition = torch.logical_or(torch.isinf(tensor), torch.isnan(tensor))
                corrected = torch.where(condition, torch.zeros_like(tensor), tensor)
                normalized = normalize_camera_output_for_display(corrected, data_type)
                grid = make_camera_output_grid(normalized)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                tag = f"{sensor_name}/{data_type}/episode_{self._episode_index:05d}"

                if self.writer is not None:
                    self.writer.add_image(tag, ndarr, global_step=self._global_step_count, dataformats="HWC")
                else:
                    file_path = os.path.join(
                        self.output_dir,
                        self._safe_path_name(sensor_name),
                        self._safe_path_name(data_type),
                        f"episode_{self._episode_index:05d}_step_{self._step_count:08d}.png",
                    )
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    result_image = Image.fromarray(ndarr)
                    result_image.save(file_path)

    @staticmethod
    def _safe_path_name(name: str) -> str:
        """Return a filesystem-safe path component."""
        return "".join(character if character.isalnum() or character in "._-" else "_" for character in name)


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
        Process exit code. Returns ``0`` after printing selector help when
        ``argv`` requests ``-h`` or ``--help`` without ``--rl_library``;
        otherwise returns ``2`` when no library is selected.
    """
    if argv is None:
        argv = sys.argv[1:]
    # the per-library scripts parse this explicit list (not sys.argv), so the sys.argv
    # fusing in AppLauncher.add_app_launcher_args never reaches it; normalize here too
    argv = AppLauncher._fuse_kit_args(argv)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rl_library", choices=sorted(entrypoints))
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


def resolve_play_checkpoint(checkpoint: str | None, framework: str, task: str) -> str:
    """Resolve an explicit or published checkpoint for a play workflow.

    Args:
        checkpoint: Local or Nucleus checkpoint path.
        framework: RL library name.
        task: Gym task id; namespaces and a trailing ``-Play`` are ignored for published lookups.

    Returns:
        Local checkpoint path.

    Raises:
        FileNotFoundError: If no explicit or published checkpoint is available.
    """
    if checkpoint:
        from isaaclab.utils.assets import retrieve_file_path

        return retrieve_file_path(checkpoint)

    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    logger.warning("No --checkpoint given; using the published checkpoint for %s / %s.", framework, task)
    published_task = task.split(":")[-1].replace("-Play", "")
    path = get_published_pretrained_checkpoint(framework, published_task)
    if path is None:
        raise FileNotFoundError(
            f"No checkpoint available for framework {framework!r} and task {task!r}; pass --checkpoint"
        )
    return path


def add_common_train_args(
    parser: argparse.ArgumentParser,
    *,
    agent_default: str | None,
    agent_help: str,
    include_agent: bool = True,
    include_distributed: bool = True,
    max_iterations_type: Callable[[str], int] = int,
) -> None:
    """Add common Isaac Lab reinforcement learning training arguments.

    Args:
        parser: The parser to add arguments to.
        agent_default: Default agent config entry point.
        agent_help: Help text for the ``--agent`` argument.
        include_agent: Whether to include the ``--agent`` argument.
        include_distributed: Whether to include the ``--distributed`` argument.
        max_iterations_type: Converter and validator for ``--max_iterations``.
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
    parser.add_argument(
        "--max_iterations", type=max_iterations_type, default=None, help="RL Policy training iterations."
    )
    parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
    parser.add_argument(
        "--ray-proc-id",
        "-rid",
        type=int,
        default=None,
        help="Automatically configured by Ray integration, otherwise None.",
    )

    parser.add_argument(
        "--capture_env_sensors",
        type=int,
        default=0,
        help="Number of environment views to capture from each image-like scene sensor.",
    )
    parser.add_argument(
        "--capture_env_sensors_length",
        type=int,
        default=200,
        help="Length of each captured sensor frame window (in steps).",
    )
    parser.add_argument(
        "--capture_env_sensors_interval",
        type=int,
        default=2000,
        help="Interval between captured sensor frame windows (in steps).",
    )
    parser.add_argument(
        "--capture_env_sensors_format",
        choices=["tensorboard", "file"],
        default="tensorboard",
        help="Format used to save the captured sensor frames.",
    )


def enable_cameras_for_video(args_cli: argparse.Namespace) -> None:
    """Enable camera rendering when video recording or sensor capture is requested.

    Args:
        args_cli: Parsed command-line arguments.
    """
    if getattr(args_cli, "video", False) or getattr(args_cli, "capture_env_sensors", 0) > 0:
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


def wrap_sensor_capture(env: gym.Env, log_dir: str, args_cli: argparse.Namespace):
    """Wrap an environment with sensor capture when requested.

    Args:
        env: Gymnasium environment to wrap.
        log_dir: Training log directory.
        args_cli: Parsed command-line arguments.

    Returns:
        The original or sensor-capture-wrapped environment.
    """
    if args_cli.capture_env_sensors <= 0:
        return env

    output_dir = os.path.join(log_dir, "sensor_frames", "train")
    sensor_capture_kwargs = {
        "output_dir": output_dir,
        "frame_count": args_cli.capture_env_sensors_length,
        "capture_num_envs": args_cli.capture_env_sensors,
        "interval": args_cli.capture_env_sensors_interval,
        "output_format": args_cli.capture_env_sensors_format,
    }
    print("[INFO] Capturing environment sensor frames during training.")
    print_dict(sensor_capture_kwargs, nesting=4)
    return CaptureEnvSensors(env, **sensor_capture_kwargs)


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


def wrap_training_capture(env: gym.Env, log_dir: str, args_cli: argparse.Namespace) -> gym.Env:
    """Apply optional video and sensor capture wrappers for training."""
    env = wrap_record_video(env, log_dir, args_cli)
    env = wrap_sensor_capture(env, log_dir, args_cli)
    return env


def dump_train_configs(log_dir: str, env_cfg: Any, agent_cfg: Any) -> None:
    """Dump training configuration files under a run log directory.

    Args:
        log_dir: Training log directory.
        env_cfg: Isaac Lab environment config.
        agent_cfg: Reinforcement learning agent config.
    """
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)


def write_run_manifest(
    log_dir: str,
    *,
    library: str,
    task: str,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write metadata used to discover checkpoints from a training run.

    Args:
        log_dir: Training run directory.
        library: Reinforcement learning library that owns the run.
        task: Task used for training.
        metadata: Additional fields used to distinguish compatible runs.
    """
    run_dir = Path(log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": RUN_MANIFEST_VERSION,
        "library": library,
        "task": _normalize_task_name(task),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    temporary_path = run_dir / f".{RUN_MANIFEST_FILENAME}.{os.getpid()}.tmp"
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, manifest_path)


def resolve_checkpoint_selector(
    log_root_path: str,
    selector: str,
    *,
    library: str,
    task: str,
    checkpoint_pattern: str,
    other_dirs: list[str] | None = None,
    preferred_checkpoint_pattern: str | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    """Resolve a checkpoint selector using manifests from new training runs.

    ``latest`` selects the naturally last checkpoint in the newest compatible
    run. ``best`` prefers the backend's canonical best or final checkpoint and
    falls back to the same checkpoint used by ``latest``.

    Args:
        log_root_path: Directory containing training run directories.
        selector: Checkpoint selector, either ``"latest"`` or ``"best"``.
        library: Reinforcement learning library expected in the run manifest.
        task: Task expected in the run manifest.
        checkpoint_pattern: Regular expression matching checkpoint filenames.
        other_dirs: Intermediate directories below each run directory.
        preferred_checkpoint_pattern: Regular expression for the backend's best or final checkpoint.
        metadata: Additional manifest metadata required for compatibility.

    Returns:
        Absolute path to the selected checkpoint.

    Raises:
        ValueError: If the selector is invalid or no compatible manifested run has a checkpoint.
    """
    if selector not in CHECKPOINT_SELECTORS:
        raise ValueError(f"Unknown checkpoint selector '{selector}'. Expected one of: {sorted(CHECKPOINT_SELECTORS)}.")

    log_root = Path(log_root_path)
    expected_task = _normalize_task_name(task)
    expected_metadata = metadata or {}
    runs: list[tuple[datetime, Path]] = []
    if log_root.is_dir():
        for run_dir in log_root.iterdir():
            if not run_dir.is_dir():
                continue
            manifest_path = run_dir / RUN_MANIFEST_FILENAME
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                created_at = datetime.fromisoformat(manifest["created_at"])
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if manifest.get("version") != RUN_MANIFEST_VERSION:
                continue
            if manifest.get("library") != library or manifest.get("task") != expected_task:
                continue
            manifest_metadata = manifest.get("metadata", {})
            if not isinstance(manifest_metadata, dict):
                continue
            if any(manifest_metadata.get(key) != value for key, value in expected_metadata.items()):
                continue
            runs.append((created_at, run_dir))

    for _, run_dir in sorted(runs, reverse=True):
        checkpoint_dir = run_dir.joinpath(*(other_dirs or []))
        if not checkpoint_dir.is_dir():
            continue
        checkpoints = [
            path for path in checkpoint_dir.iterdir() if path.is_file() and re.fullmatch(checkpoint_pattern, path.name)
        ]
        if not checkpoints:
            continue
        if selector == "best" and preferred_checkpoint_pattern is not None:
            preferred = [
                path for path in checkpoints if re.fullmatch(preferred_checkpoint_pattern, path.name) is not None
            ]
            if preferred:
                checkpoints = preferred
        checkpoints.sort(key=lambda path: _natural_sort_key(path.name))
        return str(checkpoints[-1].resolve())

    raise ValueError(
        f"No compatible manifested run with a checkpoint was found in '{log_root}'. "
        f"Run training with the current unified training entrypoint before using '--checkpoint {selector}'."
    )


def _normalize_task_name(task: str) -> str:
    """Normalize training and play variants to the same task name."""
    return task.split(":")[-1].removesuffix("-Play")


def _natural_sort_key(value: str) -> list[int | str]:
    """Return a key that sorts numeric filename components by value."""
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", value)]
