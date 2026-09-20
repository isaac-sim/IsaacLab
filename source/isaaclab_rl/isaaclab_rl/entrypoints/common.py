# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the reinforcement learning train and play entrypoints."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import re
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import warp as wp
from PIL import Image

from isaaclab.app import LoadingScreen, scan
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.images import make_camera_output_grid, normalize_camera_output_for_display
from isaaclab.utils.io import dump_yaml

logger = logging.getLogger(__name__)

RUN_MANIFEST_FILENAME = "run.json"
RUN_MANIFEST_VERSION = 1
CHECKPOINT_SELECTORS = frozenset({"latest", "best"})

_MISSING = object()
_PHYSICS_BACKEND_NAMES = {"PhysxCfg": "isaacsim_physx", "OvPhysxCfg": "ovphysx", "PhysxAutoCfg": "physx"}
_RENDERER_BACKEND_NAMES = {"isaac_rtx": "isaacsim_rtx", "newton_warp": "newton_renderer", "auto_rtx": "rtx"}
# streaming visualizers do not expose a local frame-capture API
_NO_CAPTURE_VISUALIZERS = frozenset({"rerun", "viser"})


"""
Scoped global state.
"""


@contextmanager
def preserve_attribute(target: object, name: str) -> Iterator[None]:
    """Restore an attribute after a scoped operation.

    The attribute is deleted on exit when it did not exist before entering the context.

    Args:
        target: Object containing the attribute.
        name: Name of the attribute to restore.
    """
    previous = getattr(target, name, _MISSING)
    try:
        yield
    finally:
        if previous is _MISSING:
            if hasattr(target, name):
                delattr(target, name)
        else:
            setattr(target, name, previous)


@contextmanager
def scoped_torch_backend_flags(
    *,
    cuda_matmul_allow_tf32: bool,
    cudnn_allow_tf32: bool,
    cudnn_deterministic: bool,
    cudnn_benchmark: bool,
) -> Iterator[None]:
    """Temporarily configure Torch backend flags.

    Args:
        cuda_matmul_allow_tf32: Whether CUDA matrix multiplication may use TF32.
        cudnn_allow_tf32: Whether cuDNN may use TF32.
        cudnn_deterministic: Whether cuDNN uses deterministic algorithms.
        cudnn_benchmark: Whether cuDNN benchmarks convolution algorithms.
    """
    settings = (
        (torch.backends.cuda.matmul, "allow_tf32", cuda_matmul_allow_tf32),
        (torch.backends.cudnn, "allow_tf32", cudnn_allow_tf32),
        (torch.backends.cudnn, "deterministic", cudnn_deterministic),
        (torch.backends.cudnn, "benchmark", cudnn_benchmark),
    )
    with ExitStack() as cleanup:
        for target, name, value in settings:
            cleanup.enter_context(preserve_attribute(target, name))
            setattr(target, name, value)
        yield


"""
Command-line arguments.
"""


def add_frontend_args(parser: argparse.ArgumentParser) -> None:
    """Add the environment-runtime selector argument.

    The flag is always registered so the CLI surface does not depend on optional packages; the
    warp runtime itself is imported only when selected (see :func:`create_isaaclab_env`).

    Args:
        parser: The parser to add the argument to.
    """
    parser.add_argument(
        "--frontend",
        type=str,
        choices=["torch", "warp"],
        default="torch",
        help=(
            "Runtime that constructs the environment. 'torch' uses the registered stable environment via"
            " gym.make. 'warp' (experimental) adapts a manager-based task config onto the Warp runtime, or"
            " dispatches a direct task to its registered Warp environment; requires isaaclab_experimental"
            " and `physics=newton_mjwarp`."
        ),
    )


def _add_video_args(parser: argparse.ArgumentParser, *, action: str) -> None:
    """Add the video recording arguments shared by training and playback."""
    parser.add_argument("--video", action="store_true", default=False, help=f"Record videos during {action}.")
    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="Length of each recorded video clip in env steps. Overrides the value in VideoRecorderCfg.",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=None,
        help="Interval between video clips in env steps. Overrides the value in VideoRecorderCfg.",
    )


def add_common_train_args(
    parser: argparse.ArgumentParser,
    *,
    agent_default: str | None,
    agent_help: str,
    include_agent: bool = True,
    include_distributed: bool = True,
    max_iterations_type: Callable[[str], int] = int,
) -> None:
    """Add the training arguments shared by all reinforcement learning backends.

    Args:
        parser: The parser to add arguments to.
        agent_default: Default agent config entry point.
        agent_help: Help text for the ``--agent`` argument.
        include_agent: Whether to include the ``--agent`` argument.
        include_distributed: Whether to include the ``--distributed`` argument.
        max_iterations_type: Converter and validator for ``--max_iterations``.
    """
    _add_video_args(parser, action="training")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    add_frontend_args(parser)
    if include_agent:
        parser.add_argument("--agent", type=str, default=agent_default, help=agent_help)
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    if include_distributed:
        parser.add_argument(
            "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
        )
    parser.add_argument(
        "--max_iterations", type=max_iterations_type, default=None, help="RL Policy training iterations."
    )
    parser.add_argument(
        "--export_io_descriptors",
        action="store_true",
        default=False,
        help="Deprecated: export IO descriptors (removed in Isaac Lab 3.2).",
    )
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


def add_common_play_args(parser: argparse.ArgumentParser, *, agent_default: str | None, agent_help: str) -> None:
    """Add the playback arguments shared by all reinforcement learning backends.

    Backends add their own ``--checkpoint`` argument since the accepted selectors differ.

    Args:
        parser: The parser to add arguments to.
        agent_default: Default agent config entry point.
        agent_help: Help text for the ``--agent`` argument.
    """
    _add_video_args(parser, action="play")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    add_frontend_args(parser)
    parser.add_argument("--agent", type=str, default=agent_default, help=agent_help)
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    parser.add_argument(
        "--train_env_cfg",
        action="store_true",
        default=False,
        help="Play with the training environment configuration as-is, skipping play-mode overrides.",
    )


def enable_cameras_for_video(args_cli: argparse.Namespace) -> None:
    """Enable camera rendering when video recording or sensor capture is requested.

    Args:
        args_cli: Parsed command-line arguments.
    """
    if getattr(args_cli, "video", False) or getattr(args_cli, "capture_env_sensors", 0) > 0:
        args_cli.enable_cameras = True


def set_hydra_args(hydra_args: list[str]) -> None:
    """Replace ``sys.argv`` with the arguments intended for Hydra.

    Args:
        hydra_args: Remaining command-line arguments not consumed by argparse.
    """
    sys.argv = [sys.argv[0]] + hydra_args


def resolve_seed(seed: int | None) -> int | None:
    """Return the requested seed, drawing a random one when ``-1`` is given.

    Args:
        seed: Seed from the command line, or None when not requested.
    """
    return random.randint(0, 10000) if seed == -1 else seed


def normalize_task_name(task: str) -> str:
    """Return the training task name without a namespace prefix or a ``-Play`` suffix.

    Args:
        task: Gym task id, possibly with a namespace prefix.
    """
    return task.split(":")[-1].removesuffix("-Play")


def resolve_play_task_name(task: str | None) -> str | None:
    """Redirect a retired ``-Play`` task id to its training task id.

    Task ids with a ``-Play`` suffix (before an optional ``-v<N>`` version) were removed in favor of
    play-mode overrides that play scripts apply to the training configuration (see ``play_mode`` on
    the environment configuration). When ``task`` carries the suffix, is not itself registered, and
    the corresponding training task id is registered, the training task id is returned (preserving
    any namespace prefix) along with a deprecation warning. Externally registered ``-Play`` tasks
    are returned unchanged.

    Args:
        task: Gym task id, possibly with a namespace prefix.

    Returns:
        The task id to use, or None if ``task`` is None.
    """
    if not task:
        return task
    namespace, _, name = task.rpartition(":")
    train_name = re.sub(r"-Play(-v\d+)?$", r"\1", name)
    if train_name == name or name in gym.registry or train_name not in gym.registry:
        return task
    warnings.warn(
        f"Task '{name}' was removed. Playing '{train_name}' with play-mode overrides instead. "
        "Pass --train_env_cfg to play the training configuration as-is.",
        FutureWarning,
        stacklevel=2,
    )
    return f"{namespace}:{train_name}" if namespace else train_name


"""
Environment configuration.
"""


def apply_env_overrides(args_cli: argparse.Namespace, env_cfg: Any, *, apply_device: bool = True) -> None:
    """Apply the common environment overrides from the command line.

    Every override is read with a default so parsers that omit an argument are supported.

    Args:
        args_cli: Parsed command-line arguments.
        env_cfg: Isaac Lab environment config.
        apply_device: Whether to apply the ``--device`` override for non-distributed runs.
    """
    if getattr(args_cli, "num_envs", None) is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if apply_device and not getattr(args_cli, "distributed", False):
        device = getattr(args_cli, "device", None)
        if device is not None:
            env_cfg.sim.device = device
    if getattr(args_cli, "disable_fabric", False):
        env_cfg.sim.use_fabric = False
    if getattr(args_cli, "export_io_descriptors", False):
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = True
        else:
            logger.warning("IO descriptors are only supported for manager-based RL environments; none are exported.")
    # --deterministic is an AppLauncher flag, so it only reaches carb settings on its own. Record the
    # request on the resolved physics config; each backend translates and validates it at startup.
    request_determinism(args_cli, env_cfg)


def request_determinism(args_cli: argparse.Namespace, env_cfg: Any) -> None:
    """Record a ``--deterministic`` request on the config tree and on Warp's global.

    Call this before the environment is created: Warp reads its setting at module build time and
    the scene BVH is built while the environment is constructed.

    Args:
        args_cli: Parsed command-line arguments.
        env_cfg: Isaac Lab environment config.
    """
    if not getattr(args_cli, "deterministic", False):
        return
    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if physics_cfg is not None:
        physics_cfg.deterministic = True
    request_warp_determinism(physics_cfg)


def request_warp_determinism(physics_cfg: Any) -> None:
    """Ask Warp for deterministic atomics process-wide, matching the configured guarantee.

    Newton's solvers apply their ``deterministic`` argument per module, so a solver-level request
    already covers the physics kernels. Its sensor and geometry modules fall back to
    ``warp.config.deterministic`` instead, which defaults to ``NOT_GUARANTEED``. The BVH shape
    compaction in ``newton._src.geometry.bvh`` claims output slots with ``wp.atomic_add``, so the
    primitive order of the scene BVH varies between processes and a tiled camera renders a handful
    of pixels differently from identical simulation state, which is enough to make an
    image-observation policy diverge.

    The strings accepted by :attr:`~isaaclab_newton.physics.NewtonCfg.deterministic_mode` are the
    ``warp.DeterministicMode`` members lowercased, so a stronger configured guarantee such as
    ``"gpu_to_gpu"`` is honored rather than weakened to ``RUN_TO_RUN``. The modes are ordered by
    strength and this only ever raises the setting: a guarantee already in place, whoever set it,
    is never weakened.

    Args:
        physics_cfg: Resolved physics config, or None when the config tree carries none.
    """
    requested = getattr(physics_cfg, "deterministic_mode", None)
    mode = getattr(wp.DeterministicMode, requested.upper(), None) if isinstance(requested, str) else None
    if mode is None or mode == wp.DeterministicMode.NOT_GUARANTEED:
        mode = wp.DeterministicMode.RUN_TO_RUN
    if wp.config.deterministic < mode:
        wp.config.deterministic = mode


def validate_distributed_device(args_cli: argparse.Namespace) -> None:
    """Reject distributed training on a CPU device.

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


"""
Startup reporting.
"""


def startup_screen(args_cli: argparse.Namespace, *, num_stages: int) -> LoadingScreen:
    """Create the loading screen shown while a run starts up.

    The live screen is used only for an interactive console, and only when the run did not ask
    for verbose logging; otherwise the startup output is the point and is left untouched.

    Args:
        args_cli: Parsed command-line arguments.
        num_stages: Number of stages the progress bar counts up to.

    Returns:
        An unopened loading screen.
    """
    verbose = getattr(args_cli, "verbose", False) or getattr(args_cli, "info", False)
    return LoadingScreen(num_stages, enabled=False if verbose else None)


def show_run_summary(
    screen: LoadingScreen,
    args_cli: argparse.Namespace,
    env_cfg: Any,
    *,
    library: str,
    action: str,
) -> None:
    """Print a summary of the backends and scale a run is about to use.

    Every row names the backend that will run. An automatic launcher choice is shown as
    ``<automatic> (<concrete>)``.

    Resolving automatic backend configurations mutates *env_cfg* in place, exactly as the following
    :func:`~isaaclab.app.launch_simulation` call would; call this after every other pre-launch
    config change, in particular :func:`pre_launch_video_config`.

    Args:
        screen: Loading screen that owns the console.
        args_cli: Parsed command-line arguments.
        env_cfg: Concrete Isaac Lab environment config.
        library: Reinforcement learning library running the workflow.
        action: Workflow name, either ``"train"`` or ``"play"``.
    """
    device = getattr(args_cli, "device", None) or env_cfg.sim.device
    num_envs = getattr(args_cli, "num_envs", None) or env_cfg.scene.num_envs

    # read the names before the scan resolves the automatic selectors so a row can report the
    # family the run asked for next to the backend that family resolved to
    requested_physics = _physics_backend_name(env_cfg.sim.physics)
    requested_renderer = _renderer_name(env_cfg)
    scan(env_cfg, args_cli)
    physics = _physics_backend_name(env_cfg.sim.physics)
    renderer = _renderer_name(env_cfg)

    screen.summary(
        f"Isaac Lab · {action}",
        {
            "Task": args_cli.task,
            "Workflow": _workflow_name(env_cfg),
            "RL library": library,
            "Physics": _backend_label(requested_physics, physics),
            "Renderer": (
                "n/a (no camera sensors)"
                if renderer is None
                else _backend_label(requested_renderer or renderer, renderer)
            ),
            "Visualizer": _visualizer_name(args_cli, env_cfg),
            "Device": str(device),
            "Environments": str(num_envs),
        },
    )


def _backend_label(requested: str, concrete: str) -> str:
    """Return the concrete backend name, prefixed by its automatic selector when they differ."""
    return concrete if requested == concrete else f"{requested} ({concrete})"


def _workflow_name(env_cfg: Any) -> str:
    """Return the task workflow *env_cfg* belongs to."""
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        return "manager-based"
    return "direct (multi-agent)" if isinstance(env_cfg, DirectMARLEnvCfg) else "direct"


def _physics_backend_name(physics_cfg: Any) -> str:
    """Return the backend name of a concrete physics config."""
    class_name = type(physics_cfg).__name__
    if class_name in _PHYSICS_BACKEND_NAMES:
        return _PHYSICS_BACKEND_NAMES[class_name]
    backend = class_name.removesuffix("Cfg").lower()
    solver_cfg = getattr(physics_cfg, "solver_cfg", None)
    if solver_cfg is None:
        return backend
    return f"{backend}_{type(solver_cfg).__name__.removesuffix('SolverCfg').lower()}"


def _renderer_name(env_cfg: Any) -> str | None:
    """Return the backend name of the renderer used by the first camera sensor of *env_cfg*.

    Only configs whose class declares ``renderer_cfg`` are read. Probing every attribute with
    :func:`getattr` instead would resolve lazily evaluated config values, notably the
    ``ResolvableString`` class handles, which imports Kit modules before
    :class:`~isaaclab.app.AppLauncher` starts and breaks the Isaac Sim runtime.
    """
    for container in (env_cfg, getattr(env_cfg, "scene", None)):
        for value in vars(container).values() if container is not None else ():
            if "renderer_cfg" not in getattr(type(value), "__dataclass_fields__", {}):
                continue
            renderer_cfg = value.renderer_cfg
            if isinstance(renderer_cfg, RendererCfg):
                return _RENDERER_BACKEND_NAMES.get(renderer_cfg.renderer_type, renderer_cfg.renderer_type)
    return None


def _visualizer_name(args_cli: argparse.Namespace, env_cfg: Any) -> str:
    """Return the visualizers selected on the command line or by *env_cfg*."""
    selected = getattr(args_cli, "visualizer", None)
    if isinstance(selected, str):
        selected = selected.split(",")
    if not selected:
        visualizer_cfgs = env_cfg.sim.visualizer_cfgs
        if not isinstance(visualizer_cfgs, list):
            visualizer_cfgs = [visualizer_cfgs]
        selected = [cfg.visualizer_type for cfg in visualizer_cfgs if cfg is not None]
    return ", ".join(str(name).strip() for name in selected) if selected else "none (headless)"


"""
Environment creation.
"""


def create_isaaclab_env(
    task: str,
    env_cfg: Any,
    args_cli: argparse.Namespace,
    *,
    convert_marl_to_single_agent: bool,
) -> gym.Env:
    """Create the Isaac Lab Gymnasium environment.

    Args:
        task: Task name to instantiate.
        env_cfg: Isaac Lab environment config.
        args_cli: Parsed command-line arguments.
        convert_marl_to_single_agent: Whether to convert direct MARL environments to single-agent environments.

    Returns:
        The created Gymnasium environment.
    """
    if args_cli.frontend == "torch":
        env = gym.make(task, cfg=env_cfg)
    else:
        # the warp frontend lives in the optional isaaclab_experimental package
        from isaaclab_experimental.envs.frontend import WarpFrontend

        env = WarpFrontend.build_env(env_cfg, task)
    if convert_marl_to_single_agent and isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
        # concrete environment modules load simulation modules, so import them after the launch
        from isaaclab.envs import multi_agent_to_single_agent

        env = multi_agent_to_single_agent(env)
    return env


"""
Checkpoints.
"""


def resolve_published_checkpoint(library: str, task: str, env_cfg: Any) -> str | None:
    """Fetch the published pre-trained checkpoint of a task for the configured backends.

    Args:
        library: RL library name.
        task: Gym task id; namespaces and a trailing ``-Play`` are ignored.
        env_cfg: Resolved environment config used to identify the active backends.

    Returns:
        Local checkpoint path, or None when no checkpoint is published for this combination.
    """
    # importing the checkpoint utilities registers every task, so defer it to first use
    from ..utils.pretrained_checkpoint import (
        get_pretrained_checkpoint_backend_names,
        get_published_pretrained_checkpoint,
    )

    backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
    return get_published_pretrained_checkpoint(library, normalize_task_name(task), *backend_names)


def resolve_play_checkpoint(
    checkpoint: str | None,
    framework: str,
    task: str,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg | None = None,
) -> str:
    """Resolve an explicit or published checkpoint for a play workflow.

    Args:
        checkpoint: Local or Nucleus checkpoint path.
        framework: RL library name.
        task: Gym task id; namespaces and a trailing ``-Play`` are ignored for published lookups.
        env_cfg: Resolved environment config used to identify the active backends.

    Returns:
        Local checkpoint path.

    Raises:
        FileNotFoundError: If no explicit or published checkpoint is available.
    """
    if checkpoint:
        return retrieve_file_path(checkpoint)
    # importing the checkpoint utilities registers every task, so defer it to first use
    from ..utils.pretrained_checkpoint import (
        get_pretrained_checkpoint_backend_names,
        get_published_pretrained_checkpoint,
    )

    logger.warning("No --checkpoint given; using the published checkpoint for %s / %s.", framework, task)
    backend_names = ()
    if env_cfg is not None:
        backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
    path = get_published_pretrained_checkpoint(framework, normalize_task_name(task), *backend_names)
    if path is None:
        raise FileNotFoundError(
            f"No checkpoint available for framework {framework!r} and task {task!r}; pass --checkpoint"
        )
    return path


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
        "task": normalize_task_name(task),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    # write through a temporary file so a concurrent reader never sees a partial manifest
    temporary_path = run_dir / f".{RUN_MANIFEST_FILENAME}.{os.getpid()}.tmp"
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, run_dir / RUN_MANIFEST_FILENAME)


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
    recursive: bool = False,
) -> str:
    """Resolve a checkpoint selector using the manifests written by training runs.

    ``latest`` selects the naturally last checkpoint in the newest compatible run. ``best`` prefers
    the backend's canonical best or final checkpoint and falls back to the same checkpoint used by
    ``latest``.

    Args:
        log_root_path: Directory containing training run directories.
        selector: Checkpoint selector, either ``"latest"`` or ``"best"``.
        library: Reinforcement learning library expected in the run manifest.
        task: Task expected in the run manifest.
        checkpoint_pattern: Regular expression matching checkpoint filenames.
        other_dirs: Intermediate directories below each run directory.
        preferred_checkpoint_pattern: Regular expression for the backend's best or final checkpoint.
        metadata: Additional manifest metadata required for compatibility.
        recursive: Whether to search recursively below each matching run directory.

    Returns:
        Absolute path to the selected checkpoint.

    Raises:
        ValueError: If the selector is invalid or no compatible manifested run has a checkpoint.
    """
    if selector not in CHECKPOINT_SELECTORS:
        raise ValueError(f"Unknown checkpoint selector '{selector}'. Expected one of: {sorted(CHECKPOINT_SELECTORS)}.")

    log_root = Path(log_root_path)
    runs = _compatible_runs(log_root, library=library, task=task, metadata=metadata or {})
    for _, run_dir in sorted(runs, reverse=True):
        checkpoint_dir = run_dir.joinpath(*(other_dirs or []))
        if not checkpoint_dir.is_dir():
            continue
        paths = checkpoint_dir.rglob("*") if recursive else checkpoint_dir.iterdir()
        checkpoints = [path for path in paths if path.is_file() and re.fullmatch(checkpoint_pattern, path.name)]
        if not checkpoints:
            continue
        if selector == "best" and preferred_checkpoint_pattern is not None:
            preferred = [path for path in checkpoints if re.fullmatch(preferred_checkpoint_pattern, path.name)]
            checkpoints = preferred or checkpoints
        checkpoints.sort(key=lambda path: _natural_sort_key(str(path.relative_to(checkpoint_dir))))
        return str(checkpoints[-1].resolve())

    raise ValueError(
        f"No compatible manifested run with a checkpoint was found in '{log_root}'. "
        f"Run training with the current unified training entrypoint before using '--checkpoint {selector}'."
    )


def _compatible_runs(
    log_root: Path, *, library: str, task: str, metadata: dict[str, str]
) -> list[tuple[datetime, Path]]:
    """Return the creation time and directory of every manifested run compatible with the request."""
    if not log_root.is_dir():
        return []
    expected_task = normalize_task_name(task)
    runs = []
    for run_dir in log_root.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            manifest = json.loads((run_dir / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(manifest["created_at"])
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            continue
        if manifest.get("version") != RUN_MANIFEST_VERSION:
            continue
        if manifest.get("library") != library or manifest.get("task") != expected_task:
            continue
        manifest_metadata = manifest.get("metadata", {})
        if not isinstance(manifest_metadata, dict):
            continue
        if any(manifest_metadata.get(key) != value for key, value in metadata.items()):
            continue
        runs.append((created_at, run_dir))
    return runs


def _natural_sort_key(value: str) -> list[int | str]:
    """Return a key that sorts numeric filename components by value."""
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", value)]


"""
Logging.
"""


def dump_train_configs(log_dir: str, env_cfg: Any, agent_cfg: Any) -> None:
    """Dump the training configuration files under a run log directory.

    Args:
        log_dir: Training log directory.
        env_cfg: Isaac Lab environment config.
        agent_cfg: Reinforcement learning agent config.
    """
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)


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

        Raises:
            ValueError: If the output format is not supported.
        """
        super().__init__(env)
        if output_format not in {"tensorboard", "file"}:
            raise ValueError(f"Unsupported sensor capture output format: {output_format}")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_count = max(frame_count, 0)
        self.capture_num_envs = max(capture_num_envs, 0)
        self.interval = max(interval, 1)
        self._step_count = 0
        self._global_step_count = 0
        self._episode_index = 0
        self.writer = None
        if output_format == "tensorboard":
            # tensorboard is an optional dependency of the file output format
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
        """Close the writer and the wrapped environment."""
        if self.writer is not None:
            self.writer.close()
        super().close()

    def _save_frame(self) -> None:
        """Write the current sensor outputs when the current step is inside a capture window."""
        if self.frame_count == 0 or self._step_count % self.interval >= self.frame_count:
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
                tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
                grid = make_camera_output_grid(normalize_camera_output_for_display(tensor, data_type))
                image = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                if self.writer is not None:
                    tag = f"{sensor_name}/{data_type}/episode_{self._episode_index:05d}"
                    self.writer.add_image(tag, image, global_step=self._global_step_count, dataformats="HWC")
                else:
                    file_path = os.path.join(
                        self.output_dir,
                        self._safe_path_name(sensor_name),
                        self._safe_path_name(data_type),
                        f"episode_{self._episode_index:05d}_step_{self._step_count:08d}.png",
                    )
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    Image.fromarray(image).save(file_path)

    @staticmethod
    def _safe_path_name(name: str) -> str:
        """Return a filesystem-safe path component."""
        return "".join(character if character.isalnum() or character in "._-" else "_" for character in name)


def wrap_sensor_capture(env: gym.Env, log_dir: str, args_cli: argparse.Namespace) -> gym.Env:
    """Wrap a training environment with sensor capture when requested.

    Args:
        env: Gymnasium environment to wrap.
        log_dir: Training log directory.
        args_cli: Parsed command-line arguments.

    Returns:
        The original or sensor-capture-wrapped environment.
    """
    if args_cli.capture_env_sensors <= 0:
        return env
    sensor_capture_kwargs = {
        "output_dir": os.path.join(log_dir, "sensor_frames", "train"),
        "frame_count": args_cli.capture_env_sensors_length,
        "capture_num_envs": args_cli.capture_env_sensors,
        "interval": args_cli.capture_env_sensors_interval,
        "output_format": args_cli.capture_env_sensors_format,
    }
    print("[INFO] Capturing environment sensor frames during training.")
    print_dict(sensor_capture_kwargs, nesting=4)
    return CaptureEnvSensors(env, **sensor_capture_kwargs)


"""
Video recording.
"""


def pre_launch_video_config(env_cfg: Any, args_cli: argparse.Namespace) -> None:
    """Pre-inject a headless Kit visualizer into *env_cfg* so the launcher includes the Kit runtime.

    Must be called before :func:`~isaaclab.app.launch_simulation`. Only acts when ``--video`` is set
    and neither the environment config nor the command line names a visualizer or a video recorder
    to record from; :func:`apply_video_recording` wires the recorder itself after the launch.

    Args:
        env_cfg: Isaac Lab environment config to modify in-place.
        args_cli: Parsed command-line arguments.
    """
    if not getattr(args_cli, "video", False) or getattr(env_cfg, "video_recorders", None):
        return
    if _cli_visualizers(args_cli):
        return
    sim_cfg = getattr(env_cfg, "sim", None)
    if sim_cfg is None or _configured_visualizer_cfgs(sim_cfg):
        return
    if _inject_headless_kit_visualizer(sim_cfg):
        print(
            "[INFO] pre_launch_video_config: pre-injecting a headless Kit visualizer so the launcher "
            "includes the Kit runtime. Pass --viz <type> to choose a different visualizer."
        )


def apply_video_recording(
    env_cfg: Any,
    log_dir: str,
    args_cli: argparse.Namespace,
    *,
    subdir: str = "train",
    checkpoint_path: str | None = None,
) -> None:
    """Configure internal video recording on the environment config.

    Recorders already declared by the environment config are kept, preserving user-set fields such
    as ``output_dir``, ``source`` and ``fps``; only the fields controlled by CLI flags are
    overwritten. Without declared recorders, a default one records from a visualizer (see
    :func:`_resolve_video_source`) into ``<log_dir>/videos/<subdir>`` every 2000 steps.

    Maps CLI flags:

    * ``--video`` enables recording,
    * ``--video_length`` overrides :attr:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg.video_length`
      on every recorder when passed,
    * ``--video_interval`` overrides :attr:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg.video_interval`
      on every recorder when passed.

    Args:
        env_cfg: Isaac Lab environment config to modify in-place.
        log_dir: Training or play log directory.
        args_cli: Parsed command-line arguments.
        subdir: Sub-directory below ``<log_dir>/videos/`` for the fallback output path. Use ``"train"``
            for training runs and ``"play"`` for evaluation.
        checkpoint_path: Checkpoint loaded by a play run. When set to a ``model_<N>.pt`` path with a
            numeric id, the checkpoint stem is appended to play video names.

    Raises:
        ValueError: If recording is requested with the warp frontend or without a visualizer to record from.
    """
    if not getattr(args_cli, "video", False):
        return
    frontend = getattr(args_cli, "frontend", "torch") or "torch"
    if frontend != "torch":
        raise ValueError(
            f"--video is not supported with --frontend {frontend!r}. "
            "Video recording requires the standard torch frontend. "
            "Remove --video or switch to --frontend torch."
        )

    if not getattr(env_cfg, "video_recorders", None):
        source = _resolve_video_source(env_cfg, args_cli)
        env_cfg.video_recorders = [VideoRecorderCfg(source=source, video_interval=2000)]

    video_length = getattr(args_cli, "video_length", None)
    video_interval = getattr(args_cli, "video_interval", None)
    label = _checkpoint_video_label(checkpoint_path) if subdir == "play" else None
    for cfg in env_cfg.video_recorders:
        if cfg.output_dir is None:
            cfg.output_dir = os.path.join(log_dir, "videos", subdir)
        if video_length is not None:
            cfg.video_length = video_length
        if video_interval is not None:
            cfg.video_interval = video_interval
        if label is not None:
            cfg.output_filename_prefix = _checkpoint_video_prefix(cfg.output_filename_prefix, label)

    print("[INFO] Video recording enabled.")
    for cfg in env_cfg.video_recorders:
        print_dict(
            {
                "source": cfg.source,
                "output_dir": cfg.output_dir,
                "video_length": cfg.video_length,
                "video_interval": cfg.video_interval,
            },
            nesting=4,
        )


def _resolve_video_source(env_cfg: Any, args_cli: argparse.Namespace) -> str:
    """Return the recorder source for a run that declares no video recorders.

    A visualizer requested with ``--viz`` wins, then a concrete visualizer configured on the
    environment; otherwise a headless Kit visualizer is injected so there is something to record.

    Raises:
        ValueError: If ``--viz none`` or only streaming visualizers were requested.
    """
    # _parse_visualizer_csv("none") yields None rather than ["none"], so an explicitly disabled
    # visualizer is only recognizable through the ExplicitAction sentinel
    if getattr(args_cli, "visualizer_explicit", False) and getattr(args_cli, "visualizer", "not_none") is None:
        raise ValueError(
            "--video is not compatible with --viz none: there is no active visualizer to record from. "
            "Remove --viz none so that video recording can auto-create a visualizer, "
            "pass --viz kit (or another capture-capable type), "
            "or add VideoRecorderCfg(source='sensor:<name>') to your env config."
        )
    cli_visualizers = _cli_visualizers(args_cli)
    if cli_visualizers:
        capture_capable = [name for name in cli_visualizers if name not in _NO_CAPTURE_VISUALIZERS]
        if not capture_capable:
            raise ValueError(_no_capture_visualizer_message(cli_visualizers))
        return f"visualizer:{capture_capable[0]}"

    sim_cfg = getattr(env_cfg, "sim", None)
    if sim_cfg is None:
        return "visualizer"
    configured = _configured_visualizer_cfgs(sim_cfg)
    if configured:
        # prefer capture-capable visualizers over streaming-only ones
        configured.sort(key=lambda cfg: cfg.visualizer_type in _NO_CAPTURE_VISUALIZERS)
        return f"visualizer:{configured[0].visualizer_type}"
    if _inject_headless_kit_visualizer(sim_cfg):
        print(
            "[INFO] --video specified without --viz: auto-creating a headless Kit visualizer "
            "for video recording. Pass --viz <type> to choose a different visualizer, or "
            "set video_recorders in your env config to record from a scene sensor instead."
        )
        return "visualizer:kit"
    return "visualizer"


def _cli_visualizers(args_cli: argparse.Namespace) -> list[str]:
    """Return the visualizers requested with ``--viz``, ignoring ``"none"`` entries."""
    selected = getattr(args_cli, "visualizer", None) or []
    if isinstance(selected, str):
        selected = [selected]
    return [name for name in selected if name != "none"]


def _configured_visualizer_cfgs(sim_cfg: Any) -> list[Any]:
    """Return the concrete visualizer configs of a simulation config.

    A base ``VisualizerCfg`` with ``visualizer_type=None`` is a hint-only placeholder that cannot
    create a visualizer, so it does not count.
    """
    cfgs = list(getattr(sim_cfg, "visualizer_cfgs", None) or [])
    default_cfg = getattr(sim_cfg, "default_visualizer_cfg", None)
    if default_cfg is not None:
        cfgs.append(default_cfg)
    return [cfg for cfg in cfgs if getattr(cfg, "visualizer_type", None) is not None]


def _inject_headless_kit_visualizer(sim_cfg: Any) -> bool:
    """Append a headless Kit visualizer to *sim_cfg*; returns False when the visualizers package is missing."""
    try:
        from isaaclab_visualizers.kit import KitVisualizerCfg
    except ImportError:
        return False
    if not isinstance(getattr(sim_cfg, "visualizer_cfgs", None), list):
        sim_cfg.visualizer_cfgs = []
    sim_cfg.visualizer_cfgs.append(KitVisualizerCfg(headless=True))
    return True


def _no_capture_visualizer_message(names: list[str]) -> str:
    """Explain why streaming-only visualizers cannot back ``--video`` and how to record anyway."""
    quoted = " and ".join(repr(name) for name in names)
    verb = "is a streaming visualizer" if len(names) == 1 else "are streaming visualizers"
    example_cfg = {"rerun": "RerunVisualizerCfg", "viser": "ViserVisualizerCfg"}.get(
        names[0], f"{names[0].title()}VisualizerCfg"
    )
    return (
        f"--video is not supported with --viz {quoted}: {quoted} {verb} "
        "and do not expose a local frame-capture API.\n\n"
        "Supported recording backends (all support headless mode for zero UI overhead):\n"
        "  --viz kit        Kit/Omniverse viewport\n"
        "  --viz newton_gl  Newton OpenGL viewport\n"
        "  --viz newton_rtx Newton OVRTX path-traced viewport\n\n"
        f"To run {quoted} alongside video recording, add a headless capture backend\n"
        "to sim.visualizer_cfgs in your environment config, for example:\n\n"
        "  sim_cfg.visualizer_cfgs = [\n"
        f"      {example_cfg}(...),\n"
        "      KitVisualizerCfg(headless=True),   # provides frames for --video\n"
        "  ]\n\n"
        "Frames can also be captured from a scene camera sensor without any visualizer:\n"
        "  VideoRecorderCfg(source='sensor:<name>')   # add to env_cfg.video_recorders\n\n"
        "See: https://isaac-sim.github.io/IsaacLab/main/source/features/record_video.html"
    )


def _checkpoint_video_label(checkpoint_path: str | None) -> str | None:
    """Return the ``model_<N>`` stem of a checkpoint path, or None for other checkpoint names."""
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if path.suffix != ".pt" or re.fullmatch(r"model_\d+", path.stem) is None:
        return None
    return path.stem


def _checkpoint_video_prefix(prefix: str, label: str) -> str:
    """Append the checkpoint label to a video filename prefix unless it already ends with it."""
    if prefix == label or prefix.endswith(f"_{label}"):
        return prefix
    return f"{prefix}_{label}"


def wrap_record_video(env: gym.Env, log_dir: str, args_cli: argparse.Namespace) -> gym.Env:
    """Deprecated no-op kept for backwards compatibility.

    Video recording is configured before the environment is created via :func:`apply_video_recording`
    and driven inside ``env.step()``, so wrapping the environment afterwards has no effect.
    """
    if getattr(args_cli, "video", False):
        logger.warning(
            "wrap_record_video() is no longer functional; recording is now driven inside env.step(). "
            "Call apply_video_recording(env_cfg, log_dir, args_cli) before creating the environment."
        )
    return env


"""
Playback.
"""


def run_playback(step: Callable[[], None], *, dt: float, args_cli: argparse.Namespace, env_cfg: Any) -> None:
    """Step a policy until interrupted, or until the requested video clip is complete.

    Args:
        step: Callable that infers one action and steps the environment. Runs under
            :func:`torch.inference_mode`.
        dt: Environment step duration [s], used to pace real-time playback.
        args_cli: Parsed command-line arguments providing ``video``, ``video_length`` and ``real_time``.
        env_cfg: Environment config whose first video recorder bounds the clip when ``--video_length`` is omitted.
    """
    max_steps = _video_playback_steps(args_cli, env_cfg)
    print("[INFO] Policy playback is running, press Ctrl+C to exit...")
    step_count = 0
    with contextlib.suppress(KeyboardInterrupt):
        while max_steps is None or step_count < max_steps:
            start_time = time.time()
            with torch.inference_mode():
                step()
            step_count += 1
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)


def _video_playback_steps(args_cli: argparse.Namespace, env_cfg: Any) -> int | None:
    """Return the number of steps needed to record the requested clip, or None to play unbounded."""
    if not getattr(args_cli, "video", False):
        return None
    if args_cli.video_length is not None:
        return args_cli.video_length
    recorders = getattr(env_cfg, "video_recorders", None) or []
    return recorders[0].video_length + recorders[0].step_offset if recorders else None
