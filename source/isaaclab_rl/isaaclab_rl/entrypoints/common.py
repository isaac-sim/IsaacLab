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
import signal
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import warp as wp
from PIL import Image

from isaaclab.app import LoadingScreen
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg, parse_video_source
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
_PHYSICS_BACKEND_NAMES = {"PhysxCfg": "isaacsim_physx", "OvPhysxCfg": "ovphysx"}
_RENDERER_BACKEND_NAMES = {"isaac_rtx": "isaacsim_rtx", "newton_warp": "newton_renderer"}


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


def add_video_args(parser: argparse.ArgumentParser, *, action: str) -> None:
    """Add the video recording arguments shared by training and playback.

    Args:
        parser: The parser to add the arguments to.
        action: Workflow name used in the ``--video`` help text.
    """
    parser.add_argument(
        "--video",
        nargs="?",
        const="viz",
        default=None,
        type=video_source,
        metavar="SOURCE",
        help=(
            f"Record videos during {action}. SOURCE defaults to 'viz': the first capture-capable visualizer --viz"
            " selects, else a headless newton_gl. 'viz:<type>' (kit, newton_gl, newton_rtx) records from that"
            " visualizer (a bare type is shorthand for 'viz:<type>'), added headless when --viz does not select it;"
            " 'sensor:<name>[:<channel>]' records from a scene camera. Recorders declared in the environment config"
            " take precedence."
        ),
    )
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


def video_source(value: str) -> str:
    """Validate a ``--video`` source, see :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg`.

    Args:
        value: The source given on the command line.

    Returns:
        The source unchanged.

    Raises:
        argparse.ArgumentTypeError: If *value* does not follow the source grammar.
    """
    try:
        parse_video_source(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


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
    add_video_args(parser, action="training")
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
            "--run_timestamp",
            type=str,
            default=None,
            help="Timestamp naming the run folder; train_multigpu passes one so that every rank shares the folder.",
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
    add_video_args(parser, action="play")
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
    """Enable camera rendering when sensor capture is requested.

    Video recording needs no flag: :func:`~isaaclab.app.launch_simulation` enables the rendering its source needs.

    Args:
        args_cli: Parsed command-line arguments.
    """
    if getattr(args_cli, "capture_env_sensors", 0) > 0:
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
    """Return the task name without a namespace prefix.

    Args:
        task: Gym task id, possibly with a namespace prefix.
    """
    return task.split(":")[-1]


"""
Environment configuration.
"""


def apply_env_overrides(args_cli: argparse.Namespace, env_cfg: Any) -> None:
    """Apply the common environment overrides from the command line.

    ``--disable_fabric`` (play parsers only) and ``--export_io_descriptors`` (train parsers only) are read
    with a default, since each parser defines only one of them. The ``--device`` override is applied by
    :func:`~isaaclab.app.launch_simulation`.

    Args:
        args_cli: Parsed command-line arguments.
        env_cfg: Isaac Lab environment config.
    """
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(args_cli, "disable_fabric", False):
        env_cfg.sim.use_fabric = False
    if getattr(args_cli, "export_io_descriptors", False):
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = True
        else:
            logger.warning("IO descriptors are only supported for manager-based RL environments; none are exported.")
    # --deterministic is a Kit launcher flag, so it only reaches carb settings on its own. Record the
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
    """Print a summary of the backends and scale a run uses.

    Call this inside :func:`~isaaclab.app.launch_simulation`, which resolves the backends, visualizers,
    and device into *env_cfg*.

    Args:
        screen: Loading screen that owns the console.
        args_cli: Parsed command-line arguments.
        env_cfg: Isaac Lab environment config resolved by the launch.
        library: Reinforcement learning library running the workflow.
        action: Workflow name, either ``"train"`` or ``"play"``.
    """
    renderer = _renderer_name(env_cfg)
    visualizers = [cfg.visualizer_type for cfg in env_cfg.sim.visualizer_cfgs if cfg.visualizer_type]
    screen.summary(
        f"Isaac Lab · {action}",
        {
            "Task": args_cli.task,
            "Workflow": _workflow_name(env_cfg),
            "RL library": library,
            "Physics": _physics_backend_name(env_cfg.sim.physics),
            "Renderer": "n/a (no camera sensors)" if renderer is None else renderer,
            "Visualizer": ", ".join(visualizers) or "headless",
            "Device": env_cfg.sim.device,
            "Environments": str(getattr(args_cli, "num_envs", None) or env_cfg.scene.num_envs),
        },
    )


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
    :class:`~isaaclab_physx.app.KitLauncher` starts and breaks the Isaac Sim runtime.
    """
    for container in (env_cfg, getattr(env_cfg, "scene", None)):
        for value in vars(container).values() if container is not None else ():
            if "renderer_cfg" not in getattr(type(value), "__dataclass_fields__", {}):
                continue
            renderer_cfg = value.renderer_cfg
            if isinstance(renderer_cfg, RendererCfg):
                return _RENDERER_BACKEND_NAMES.get(renderer_cfg.renderer_type, renderer_cfg.renderer_type)
    return None


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
    distributed = getattr(args_cli, "distributed", False)
    if distributed:
        _nccl_probe("before env")  # TEMP diagnostic (revert before review)
    if args_cli.frontend == "torch":
        env = gym.make(task, cfg=env_cfg)
    else:
        # the warp frontend lives in the optional isaaclab_experimental package
        from isaaclab_experimental.envs.frontend import WarpFrontend

        env = WarpFrontend.build_env(env_cfg, task)
    if convert_marl_to_single_agent and isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
        # Import the environment runtime only after simulation launch.
        from isaaclab.envs import multi_agent_to_single_agent

        env = multi_agent_to_single_agent(env)
    if distributed:
        _restore_rank_device(env.unwrapped.device)
        _nccl_probe("after env")  # TEMP diagnostic (revert before review)
    return env


def _nccl_probe(stage: str) -> None:
    """TEMP diagnostic: all-reduce one element over a fresh NCCL group, then tear the group down."""
    import torch.distributed as dist

    try:
        dist.init_process_group(backend="nccl")
        value = torch.ones(1, device=f"cuda:{torch.cuda.current_device()}")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        print(f"[NCCL-PROBE] {stage}: ok (cuda:{torch.cuda.current_device()})", flush=True)
    except Exception as err:  # noqa: BLE001 - diagnostic must not mask the run
        print(f"[NCCL-PROBE] {stage}: FAILED {type(err).__name__}: {str(err).splitlines()[-1][:200]}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _restore_rank_device(device: str) -> None:
    """Make ``device`` this rank's current CUDA device again once the environment exists.

    Creating the environment starts its renderers, which can leave another GPU current on this
    thread; the rank's first collective then runs NCCL against the wrong device.

    Args:
        device: The rank's simulation device, such as ``"cuda:1"``.
    """
    if not str(device).startswith("cuda"):
        return
    from isaaclab.utils.device import set_cuda_device

    expected = torch.device(device).index or 0
    current = torch.cuda.current_device()
    if current != expected:
        logger.warning("Creating the environment moved this rank from cuda:%d to cuda:%d.", expected, current)
    set_cuda_device(device)


def close_env(env: gym.Env) -> None:
    """Close the outermost environment wrapper without interrupting its teardown.

    Args:
        env: Environment to close on the entrypoint's main thread.
    """
    previous_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        env.close()
    finally:
        signal.signal(signal.SIGINT, previous_handler)


"""
Checkpoints.
"""


def resolve_published_checkpoint(library: str, task: str, env_cfg: Any) -> str | None:
    """Fetch the published pre-trained checkpoint of a task for the configured backends.

    Args:
        library: RL library name.
        task: Gym task id; namespaces are ignored.
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
        task: Gym task id; namespaces are ignored for published lookups.
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
    """Record video from the ``--video`` source unless the environment config declares video recorders.

    Must be called before :func:`~isaaclab.app.launch_simulation`, which resolves the recorder sources against
    the ``--viz`` selection and adds a headless visualizer for a source that ``--viz`` does not select.
    :func:`apply_video_recording` sets the output directory and clip schedule after the launch.

    Args:
        env_cfg: Isaac Lab environment config to modify in-place.
        args_cli: Parsed command-line arguments.

    Raises:
        ValueError: If recording is requested with the warp frontend.
    """
    if not getattr(args_cli, "video", None):
        return
    frontend = getattr(args_cli, "frontend", "torch") or "torch"
    if frontend != "torch":
        raise ValueError(
            f"--video is not supported with --frontend {frontend!r}. "
            "Video recording requires the standard torch frontend. "
            "Remove --video or switch to --frontend torch."
        )
    if not env_cfg.video_recorders:
        env_cfg.video_recorders = [VideoRecorderCfg(source=args_cli.video, video_interval=2000)]


def apply_video_recording(
    env_cfg: Any,
    log_dir: str,
    args_cli: argparse.Namespace,
    *,
    subdir: str = "train",
    checkpoint_path: str | None = None,
) -> None:
    """Apply the ``--video`` output directory and clip schedule to the environment's video recorders.

    Call this inside :func:`~isaaclab.app.launch_simulation`, after :func:`pre_launch_video_config` before it.

    Recorders keep user-set fields such as ``output_dir``, ``source`` and ``fps``; only the fields
    controlled by CLI flags are overwritten, and a recorder without ``output_dir`` writes into
    ``<log_dir>/videos/<subdir>``.

    Maps CLI flags:

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
    """
    if not getattr(args_cli, "video", None):
        return

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
    max_steps = video_playback_steps(args_cli, env_cfg)
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


def video_playback_steps(args_cli: argparse.Namespace, env_cfg: Any) -> int | None:
    """Return the env steps a ``--video`` playback needs for every recorder to finish its first clip.

    Call this after :func:`apply_video_recording`, which applies ``--video_length`` to every recorder.

    Args:
        args_cli: Parsed command-line arguments providing ``video``.
        env_cfg: Environment config whose video recorders define the clip schedules.

    Returns:
        The step budget, or None to play unbounded when ``--video`` is not set or no recorder is configured.
    """
    if not getattr(args_cli, "video", False):
        return None
    recorders = getattr(env_cfg, "video_recorders", None) or []
    return max((cfg.step_offset + cfg.video_length for cfg in recorders), default=None)
