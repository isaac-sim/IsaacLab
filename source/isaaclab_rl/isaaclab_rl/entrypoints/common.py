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
import warnings
from collections.abc import Callable, Container, Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import torch
from PIL import Image

from isaaclab.app import AppLauncher, LoadingScreen, scan
from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.images import make_camera_output_grid, normalize_camera_output_for_display
from isaaclab.utils.io import dump_yaml

# Preset selectors whose values name a preset, and the preset names that resolved
# configs map back to when their class or ``renderer_type`` differs from the name.
_PRESET_SELECTORS = frozenset({"presets", "physics", "renderer"})
_PHYSICS_PRESET_NAMES = {"PhysxCfg": "isaacsim_physx", "OvPhysxCfg": "ovphysx", "PhysxAutoCfg": "physx"}
_RENDERER_PRESET_NAMES = {"isaac_rtx": "isaacsim_rtx", "newton_warp": "newton_renderer", "auto_rtx": "rtx"}

RUN_MANIFEST_FILENAME = "run.json"
RUN_MANIFEST_VERSION = 1
CHECKPOINT_SELECTORS = frozenset({"latest", "best"})
logger = logging.getLogger(__name__)
_MISSING = object()


@contextmanager
def preserve_attribute(target: object, name: str) -> Iterator[None]:
    """Restore an attribute after a scoped operation.

    The attribute is deleted on exit when it did not exist before entering the
    context.

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


def resolve_play_task_name(task: str | None) -> str | None:
    """Redirect a retired ``-Play`` task id to its training task id.

    Task ids with a ``-Play`` suffix (before an optional ``-v<N>`` version) were removed in
    favor of play-mode overrides that play scripts apply to the training configuration (see
    ``play_mode`` on the environment configuration). When ``task`` carries the suffix,
    is not itself registered, and the corresponding training task id is registered, the
    training task id is returned (preserving any namespace prefix) along with a deprecation
    warning. Externally registered ``-Play`` tasks are returned unchanged.

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


def add_frontend_args(parser: argparse.ArgumentParser) -> None:
    """Add the environment-runtime selector argument.

    The flag is always registered so the CLI surface does not depend on
    optional packages; the warp runtime itself is imported only when selected
    (see :func:`create_isaaclab_env`).

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
            " and `presets=newton_mjwarp`."
        ),
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
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    add_frontend_args(parser)
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


def startup_screen(args_cli: argparse.Namespace, *, num_stages: int) -> LoadingScreen:
    """Create the loading screen shown while a run starts up.

    The live screen is used only for an interactive console, and only when the
    run did not ask for verbose logging -- otherwise the startup output is the
    point and is left untouched.

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

    Every row names the backend that will run, alongside the choice the run stopped at.
    A backend reached through a family the command line named -- ``physics=physx`` and
    ``renderer=rtx`` name a family that launch resolves -- is shown as
    ``<family> (<resolved>)``, and a backend the run named neither directly nor by
    family is shown as ``default (<resolved>)``.

    Resolving those selectors mutates *env_cfg* in place, exactly as the following
    :func:`~isaaclab.app.launch_simulation` call would; call this after every other
    pre-launch config change, in particular :func:`pre_launch_video_config`.

    Args:
        screen: Loading screen that owns the console.
        args_cli: Parsed command-line arguments.
        env_cfg: Isaac Lab environment config, with its presets already resolved.
        library: Reinforcement learning library running the workflow.
        action: Workflow name, either ``"train"`` or ``"play"``.
    """
    selected = _selected_preset_names()
    device = getattr(args_cli, "device", None) or env_cfg.sim.device
    num_envs = getattr(args_cli, "num_envs", None) or env_cfg.scene.num_envs

    # Names read before the scan resolves the automatic selectors, so a row can report
    # the family the run asked for next to the backend that family resolved to
    requested_physics = _physics_name(env_cfg.sim.physics)
    requested_renderer = _renderer_name(env_cfg)
    scan(env_cfg, args_cli)
    physics = _physics_name(env_cfg.sim.physics)
    renderer = _renderer_name(env_cfg)

    screen.summary(
        f"Isaac Lab · {action}",
        {
            "Task": args_cli.task,
            "Workflow": _workflow_name(env_cfg),
            "RL library": library,
            "Physics": _label(requested_physics, physics, selected=selected),
            "Renderer": (
                "n/a (no camera sensors)"
                if renderer is None
                else _label(requested_renderer or renderer, renderer, selected=selected)
            ),
            "Presets": _additional_preset_names({requested_physics, physics, requested_renderer, renderer}),
            "Visualizer": _visualizer_name(args_cli, env_cfg),
            "Device": str(device),
            "Environments": str(num_envs),
        },
    )


def _label(requested: str, resolved: str, *, selected: set[str] = frozenset()) -> str:
    """Name the backend a row reports, and where the run stopped choosing it.

    Args:
        requested: Preset name the config carried before launch resolved its automatic
            selectors, which names a backend family when it differs from *resolved*.
        resolved: Preset name of the backend that will run.
        selected: Preset names the command line asked for.

    Returns:
        The resolved name when the run asked for that backend, ``<family> (<resolved>)``
        when it asked for the family the backend was picked from, and
        ``default (<resolved>)`` when it asked for neither.
    """
    if resolved in selected:
        return resolved
    if requested in selected:
        return f"{requested} ({resolved})"
    return f"default ({resolved})"


def _selected_preset_names() -> set[str]:
    """Return the preset names the command line asked for.

    Reads the same ``sys.argv`` the preset resolver consumes (see
    :func:`~isaaclab_tasks.utils.hydra.register_task`), so the summary marks a
    backend as chosen exactly when a ``physics=`` / ``renderer=`` / ``presets=``
    token or a Hydra path override named it.
    """
    names: set[str] = set()
    for token in sys.argv[1:]:
        key, separator, value = token.partition("=")
        if separator and (key.lstrip("-") in _PRESET_SELECTORS or key.startswith(("env.", "agent."))):
            names.update(part.strip() for part in value.split(",") if part.strip())
    return names


def _additional_preset_names(shown: Container[str | None]) -> str:
    """Return the presets the command line asked for that no other row names.

    Domain presets such as ``presets=cube`` do not surface anywhere else in the
    summary, so they are listed here. A preset a row already reports is left out,
    whether it names the backend that will run or the family the row resolved it
    from -- ``renderer=rtx`` is reported by an ``rtx (ovrtx)`` renderer row.

    Args:
        shown: Preset names reported by the physics and renderer rows, including
            the families those rows resolved from.

    Returns:
        The remaining preset names in command-line order, comma separated, or
        ``"none"`` when the run named no other preset.
    """
    names: list[str] = []
    for token in sys.argv[1:]:
        key, separator, value = token.partition("=")
        if not separator or key.lstrip("-") not in _PRESET_SELECTORS:
            continue
        for part in (part.strip() for part in value.split(",")):
            if part and part not in shown and part not in names:
                names.append(part)
    return ", ".join(names) if names else "none"


def _workflow_name(env_cfg: Any) -> str:
    """Return the task workflow *env_cfg* belongs to."""
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        return "manager-based"
    return "direct (multi-agent)" if isinstance(env_cfg, DirectMARLEnvCfg) else "direct"


def _physics_name(physics_cfg: Any) -> str:
    """Return the preset name of a resolved physics config."""
    class_name = type(physics_cfg).__name__
    if class_name in _PHYSICS_PRESET_NAMES:
        return _PHYSICS_PRESET_NAMES[class_name]
    solver_cfg = getattr(physics_cfg, "solver_cfg", None)
    backend = class_name.removesuffix("Cfg").lower()
    if solver_cfg is None:
        return backend
    return f"{backend}_{type(solver_cfg).__name__.removesuffix('SolverCfg').lower()}"


def _renderer_name(env_cfg: Any) -> str | None:
    """Return the preset name of the renderer used by the first camera sensor of *env_cfg*.

    Only configs whose class declares ``renderer_cfg`` are read. Probing every
    attribute with :func:`getattr` instead would resolve lazily evaluated config
    values -- notably the ``ResolvableString`` class handles -- which imports
    Kit modules before :class:`~isaaclab.app.AppLauncher` starts and breaks the
    Isaac Sim runtime.
    """
    for container in (env_cfg, getattr(env_cfg, "scene", None)):
        for value in vars(container).values() if container is not None else ():
            if "renderer_cfg" not in getattr(type(value), "__dataclass_fields__", {}):
                continue
            renderer_cfg = value.renderer_cfg
            if isinstance(renderer_cfg, RendererCfg):
                renderer_type = renderer_cfg.renderer_type
                return _RENDERER_PRESET_NAMES.get(renderer_type, renderer_type)
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
    if args_cli.frontend == "torch":
        env = gym.make(task, cfg=env_cfg)
    else:
        # Imported lazily: the warp frontend lives in the optional
        # isaaclab_experimental package, and the torch path must work without it.
        from isaaclab_experimental.envs.frontend import WarpFrontend

        env = WarpFrontend.build_env(env_cfg, task)
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


def pre_launch_video_config(env_cfg: Any, log_dir: str | None = None, args_cli: Any = None) -> None:
    """Pre-inject a recording visualizer into env_cfg before launch_simulation().

    Must be called BEFORE :func:`~isaaclab.app.AppLauncher.launch_simulation` so the
    launcher can include the Kit runtime requirement in its backend scan.

    Only acts when:

    * ``--video`` is set,
    * no ``video_recorders`` are pre-configured,
    * no explicit ``--viz`` was passed (or only ``"none"`` entries were passed),
    * no concrete visualizer is already in ``sim.visualizer_cfgs``, and
    * ``--viz none`` was not passed.

    Args:
        env_cfg: Isaac Lab environment config to modify in-place.
        log_dir: Unused; kept for API symmetry with :func:`apply_video_recording`.
        args_cli: Parsed command-line arguments.
    """
    if not getattr(args_cli, "video", False):
        return

    existing: list = getattr(env_cfg, "video_recorders", []) or []
    if existing:
        return

    cli_visualizers: list[str] = getattr(args_cli, "visualizer", None) or []
    if isinstance(cli_visualizers, str):
        cli_visualizers = [cli_visualizers]
    active_cli_visualizers = [v for v in cli_visualizers if v != "none"]
    if active_cli_visualizers:
        # User already chose a visualizer; apply_video_recording() will wire the source.
        return

    sim_cfg = getattr(env_cfg, "sim", None)
    if sim_cfg is None:
        return

    existing_viz_cfg = getattr(sim_cfg, "default_visualizer_cfg", None)
    existing_viz_cfgs = getattr(sim_cfg, "visualizer_cfgs", []) or []
    has_concrete_visualizer = any(getattr(c, "visualizer_type", None) is not None for c in existing_viz_cfgs) or (
        existing_viz_cfg is not None and getattr(existing_viz_cfg, "visualizer_type", None) is not None
    )
    if has_concrete_visualizer:
        return

    if not isinstance(getattr(sim_cfg, "visualizer_cfgs", None), list):
        sim_cfg.visualizer_cfgs = []

    try:
        from isaaclab_visualizers.kit import KitVisualizerCfg as _KitCfg

        sim_cfg.visualizer_cfgs.append(_KitCfg(headless=True))
        print(
            "[INFO] pre_launch_video_config: pre-injecting a headless Kit visualizer so the launcher "
            "includes the Kit runtime. Pass --viz <type> to choose a different visualizer."
        )
    except (ImportError, ModuleNotFoundError):
        pass


def apply_video_recording(env_cfg: Any, log_dir: str, args_cli: argparse.Namespace, *, subdir: str = "train") -> None:
    """Configure internal video recording on the environment config.

    Enables recording by ensuring ``env_cfg.video_recorders`` is non-empty, then applies
    any CLI overrides.  If the env config already declares recorders, those are kept as-is
    (preserving user-set fields such as ``output_dir``, ``source``, and ``fps``); only the
    fields explicitly controlled by CLI flags are overwritten.  If no recorders are declared,
    a default one is created with ``source="visualizer"`` and
    ``output_dir=<log_dir>/videos/<subdir>``.

    Maps CLI flags:

    * ``--video``            → enables recording
    * ``--video_length``     → overrides :attr:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg.video_length`
      on every recorder when explicitly passed (``None`` = use the cfg default)
    * ``--video_interval``   → overrides :attr:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg.video_interval`
      on every recorder when explicitly passed (``None`` = use the cfg default)

    Args:
        env_cfg: Isaac Lab environment config to modify in-place.
        log_dir: Training or play log directory.  When no recorders are pre-configured,
            a default :class:`~isaaclab.envs.utils.video_recorder_cfg.VideoRecorderCfg` is
            created and its ``output_dir`` is set to ``<log_dir>/videos/<subdir>``.
        args_cli: Parsed command-line arguments.
        subdir: Sub-directory name appended to ``<log_dir>/videos/`` for the fallback output
            path.  Use ``"train"`` for training runs and ``"play"`` for evaluation.
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

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    existing: list = getattr(env_cfg, "video_recorders", []) or []
    if not existing:
        # No recorders pre-configured: create a default Kit-viewport recorder, mirroring the
        # historical --video behaviour (record immediately, then every 2 000 steps).
        # If the user did not pass --viz, we auto-create a headless Kit visualizer exactly as
        # teleop and other headless-capable modes do — cameras are enabled by
        # enable_cameras_for_video() before launch_simulation() is called.
        sim_cfg = getattr(env_cfg, "sim", None)
        visualizer_source = "visualizer"  # fallback when no specific backend is available

        # Check whether the user explicitly requested a visualizer via --viz.  If so,
        # that visualizer will be created by the launcher and we should record from it
        # without injecting a second one.
        cli_visualizers: list[str] = getattr(args_cli, "visualizer", None) or []
        if isinstance(cli_visualizers, str):
            cli_visualizers = [cli_visualizers]

        # Reject the explicit --viz none --video combination: "none" disables all visualizers,
        # so there is nothing to record from.  The user must either remove --viz none, pick a
        # capture-capable visualizer type, or configure video_recorders in the env cfg to use
        # a sensor source instead.
        # NOTE: _parse_visualizer_csv("none") returns None (not ["none"]), so cli_visualizers
        # is [] even when --viz none was passed.  Use the ExplicitAction sentinel instead.
        viz_explicitly_none = (
            getattr(args_cli, "visualizer_explicit", False) and getattr(args_cli, "visualizer", "not_none") is None
        )
        if viz_explicitly_none:
            raise ValueError(
                "--video is not compatible with --viz none: there is no active visualizer to record from. "
                "Remove --viz none so that video recording can auto-create a visualizer, "
                "pass --viz kit (or another capture-capable type), "
                "or add VideoRecorderCfg(source='sensor:<name>') to your env config."
            )

        # Filter out "none" entries — "none" disables all visualizers, so treat it the
        # same as "no visualizer specified" for the purpose of picking a recorder source.
        active_cli_visualizers = [v for v in cli_visualizers if v != "none"]

        # Streaming visualizers do not expose a local frame-capture API.
        _NO_CAPTURE = {"rerun", "viser"}

        if active_cli_visualizers:
            # Partition into capture-capable and no-capture lists.
            capture_capable = [v for v in active_cli_visualizers if v not in _NO_CAPTURE]
            no_capture = [v for v in active_cli_visualizers if v in _NO_CAPTURE]

            if no_capture and not capture_capable:
                names = " and ".join(repr(v) for v in no_capture)
                example_cfg = {
                    "rerun": "RerunVisualizerCfg",
                    "viser": "ViserVisualizerCfg",
                }.get(no_capture[0], f"{no_capture[0].title()}VisualizerCfg")
                raise ValueError(
                    f"--video is not supported with --viz {names}: {names} "
                    f"{'is a streaming visualizer' if len(no_capture) == 1 else 'are streaming visualizers'} "
                    "and do not expose a local frame-capture API.\n\n"
                    "Supported recording backends (all support headless mode for zero UI overhead):\n"
                    "  --viz kit        Kit/Omniverse viewport\n"
                    "  --viz newton_gl  Newton OpenGL viewport\n"
                    "  --viz newton_rtx Newton OVRTX path-traced viewport\n\n"
                    f"To run {names} alongside video recording, add a headless capture backend\n"
                    "to sim.visualizer_cfgs in your environment config, for example:\n\n"
                    f"  sim_cfg.visualizer_cfgs = [\n"
                    f"      {example_cfg}(...),\n"
                    f"      KitVisualizerCfg(headless=True),   # provides frames for --video\n"
                    f"  ]\n\n"
                    "Frames can also be captured from a scene camera sensor without any visualizer:\n"
                    "  VideoRecorderCfg(source='sensor:<name>')   # add to env_cfg.video_recorders\n\n"
                    "See: https://isaac-sim.github.io/IsaacLab/main/source/how-to/record_video.html"
                )

            # Use the first capture-capable visualizer as the recording source.
            visualizer_source = f"visualizer:{capture_capable[0]}"
        else:
            # No --viz: determine whether a concrete visualizer is already configured in
            # the env cfg.  A base VisualizerCfg with visualizer_type=None is a hint-only
            # placeholder that cannot create a real visualizer; treat it as "not configured".
            existing_viz_cfg = getattr(sim_cfg, "default_visualizer_cfg", None) if sim_cfg is not None else None
            existing_viz_cfgs = getattr(sim_cfg, "visualizer_cfgs", []) if sim_cfg is not None else []
            has_concrete_visualizer = bool(existing_viz_cfgs) or (
                existing_viz_cfg is not None and getattr(existing_viz_cfg, "visualizer_type", None) is not None
            )

            if sim_cfg is not None and not has_concrete_visualizer:
                # Inject a headless Kit visualizer so there is something to record from.
                # pre_launch_video_config() already injected one before launch_simulation()
                # so this branch is normally not reached; it handles edge cases where that
                # pre-launch hook was skipped or failed (e.g. isaaclab_visualizers not installed).
                if not isinstance(getattr(sim_cfg, "visualizer_cfgs", None), list):
                    sim_cfg.visualizer_cfgs = []
                try:
                    from isaaclab_visualizers.kit import KitVisualizerCfg as _KitCfg

                    sim_cfg.visualizer_cfgs.append(_KitCfg(headless=True))
                    visualizer_source = "visualizer:kit"
                    print(
                        "[INFO] --video specified without --viz: auto-creating a headless Kit visualizer "
                        "for video recording. Pass --viz <type> to choose a different visualizer, or "
                        "set video_recorders in your env config to record from a scene sensor instead."
                    )
                except (ImportError, ModuleNotFoundError):
                    pass
            elif has_concrete_visualizer:
                # A concrete visualizer is already configured in the env cfg: record from it.
                # Prefer capture-capable types (kit, newton_gl) over streaming-only ones.
                all_viz_cfgs = list(existing_viz_cfgs or [])
                if existing_viz_cfg is not None and getattr(existing_viz_cfg, "visualizer_type", None) is not None:
                    all_viz_cfgs.append(existing_viz_cfg)
                # Two-pass: capture-capable first, then fall back to whatever is present.
                for vcfg in sorted(all_viz_cfgs, key=lambda c: getattr(c, "visualizer_type", None) in _NO_CAPTURE):
                    vtype = getattr(vcfg, "visualizer_type", None)
                    if vtype:
                        visualizer_source = f"visualizer:{vtype}"
                        break
        env_cfg.video_recorders = [VideoRecorderCfg(source=visualizer_source, video_interval=2000)]

    fallback_output_dir = os.path.join(log_dir, "videos", subdir)
    video_length = getattr(args_cli, "video_length", None)
    video_interval = getattr(args_cli, "video_interval", None)
    for cfg in env_cfg.video_recorders:
        # Fill in output_dir only when the user left it unset (None = use log_dir).
        if cfg.output_dir is None:
            cfg.output_dir = fallback_output_dir
        if video_length is not None:
            cfg.video_length = video_length
        if video_interval is not None:
            cfg.video_interval = video_interval

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


def wrap_record_video(env, log_dir: str, args_cli: argparse.Namespace):
    """No-op stub kept for backwards compatibility.

    Video recording is now configured before env creation via
    :func:`apply_video_recording`.  Calling this function after env
    creation has no effect.
    """
    if getattr(args_cli, "video", False):
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "wrap_record_video() is no longer functional — recording is now driven inside env.step(). "
            "Call apply_video_recording(env_cfg, log_dir, args_cli) before creating the environment."
        )
    return env


def wrap_training_capture(env: gym.Env, log_dir: str, args_cli: argparse.Namespace) -> gym.Env:
    """Apply optional sensor capture wrappers for training.

    Video recording is no longer applied here — it is configured before env creation
    via :func:`apply_video_recording`.
    """
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
    recursive: bool = False,
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
        recursive: Whether to search recursively below each matching run directory.

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
        paths = checkpoint_dir.rglob("*") if recursive else checkpoint_dir.iterdir()
        checkpoints = [path for path in paths if path.is_file() and re.fullmatch(checkpoint_pattern, path.name)]
        if not checkpoints:
            continue
        if selector == "best" and preferred_checkpoint_pattern is not None:
            preferred = [
                path for path in checkpoints if re.fullmatch(preferred_checkpoint_pattern, path.name) is not None
            ]
            if preferred:
                checkpoints = preferred
        checkpoints.sort(key=lambda path: _natural_sort_key(str(path.relative_to(checkpoint_dir))))
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
